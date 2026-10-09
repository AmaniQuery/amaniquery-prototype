"""
User Authentication Provider
Handles user registration, login, password management
"""

import secrets
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
from sqlalchemy.orm import Session
from sqlalchemy import and_
import bcrypt
from loguru import logger

from ..models.auth_models import User
from ..config import config
from ..services.password_validator import PasswordValidator
from ..services.security_audit import SecurityAudit

# IP-based brute force tracking (in-memory, backed by shared dict)
_ip_failures: Dict[str, list] = {}
_ip_lockout: Dict[str, float] = {}
PROGRESSIVE_DELAYS = [0, 1, 2, 5, 10, 30, 60]  # seconds delay per attempt


class UserAuthProvider:
    """Handles user authentication operations"""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(
                password.encode("utf-8"), password_hash.encode("utf-8")
            )
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False

    @staticmethod
    def create_user(
        db: Session,
        email: str,
        password: str,
        name: Optional[str] = None,
        phone_number: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> User:
        """Create a new user"""
        existing_user = db.query(User).filter(User.email == email.lower()).first()
        if existing_user:
            SecurityAudit.suspicious_activity(
                None, ip_address, f"duplicate_registration:{email}"
            )
            raise ValueError("User with this email already exists")

        password_validation = PasswordValidator.validate(password)
        if not password_validation:
            raise ValueError(f"Weak password: {'; '.join(password_validation.errors)}")

        normalized_phone = None
        if phone_number:
            phone = phone_number.strip().replace(" ", "").replace("-", "")
            if phone.startswith("0"):
                normalized_phone = "+254" + phone[1:]
            elif not phone.startswith("+"):
                normalized_phone = "+254" + phone
            else:
                normalized_phone = phone

        password_hash = UserAuthProvider.hash_password(password)
        verification_token = secrets.token_urlsafe(32)

        user = User(
            email=email.lower(),
            password_hash=password_hash,
            name=name,
            phone_number=normalized_phone,
            phone_verified=False,
            email_verification_token=verification_token,
            status="pending_verification",
            last_login_ip=ip_address,
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        logger.info(f"Created user: {user.email}")
        return user

    @staticmethod
    def _check_ip_rate_limit(ip_address: Optional[str]) -> Tuple[bool, float]:
        """Check if IP is rate-limited due to too many failures.
        Returns (blocked, wait_seconds).
        """
        if not ip_address:
            return False, 0

        now = time.time()
        locked_until = _ip_lockout.get(ip_address, 0)
        if locked_until > now:
            return True, locked_until - now

        failures = _ip_failures.get(ip_address, [])
        failures = [t for t in failures if now - t < 900]
        _ip_failures[ip_address] = failures

        if len(failures) >= config.MAX_LOGIN_ATTEMPTS * 3:
            _ip_lockout[ip_address] = now + 600
            logger.warning(
                f"IP {ip_address} locked out for 10 min ({len(failures)} failures)"
            )
            return True, 600

        attempt = len(failures)
        if attempt < len(PROGRESSIVE_DELAYS):
            delay = PROGRESSIVE_DELAYS[attempt]
        else:
            delay = PROGRESSIVE_DELAYS[-1]
        if delay > 0:
            return True, delay

        return False, 0

    @staticmethod
    def authenticate_user(
        db: Session, email: str, password: str, ip_address: Optional[str] = None
    ) -> Optional[User]:
        """Authenticate user with email and password"""
        blocked, wait = UserAuthProvider._check_ip_rate_limit(ip_address)
        if blocked:
            raise ValueError(
                f"Too many attempts from this IP. Please wait {wait:.0f} seconds."
            )

        user = db.query(User).filter(User.email == email.lower()).first()

        if not user:
            if ip_address:
                _ip_failures.setdefault(ip_address, []).append(time.time())
            return None

        if user.locked_until and user.locked_until > datetime.utcnow():
            remaining = (user.locked_until - datetime.utcnow()).total_seconds()
            raise ValueError(
                f"Account is locked. Try again in {remaining:.0f} seconds."
            )

        if not UserAuthProvider.verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= config.MAX_LOGIN_ATTEMPTS:
                user.locked_until = datetime.utcnow() + timedelta(
                    minutes=config.LOCKOUT_DURATION_MINUTES
                )
                logger.warning(
                    f"Account locked for {user.email} ({user.failed_login_attempts} failures)"
                )
            if ip_address:
                _ip_failures.setdefault(ip_address, []).append(time.time())
            db.commit()
            return None

        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        user.last_login_ip = ip_address
        db.commit()

        if ip_address and ip_address in _ip_failures:
            _ip_failures.pop(ip_address, None)
            _ip_lockout.pop(ip_address, None)

        logger.info(f"User authenticated: {user.email}")
        return user

    @staticmethod
    def request_password_reset(db: Session, email: str) -> Optional[str]:
        """Request password reset - returns reset token"""
        user = db.query(User).filter(User.email == email.lower()).first()

        if not user:
            # Don't reveal if user exists
            return None

        # Generate reset token
        reset_token = secrets.token_urlsafe(32)
        user.password_reset_token = reset_token
        user.password_reset_expires = datetime.utcnow() + timedelta(
            hours=config.PASSWORD_RESET_TOKEN_EXPIRE_HOURS
        )

        db.commit()

        logger.info(f"Password reset requested for: {user.email}")
        return reset_token

    @staticmethod
    def reset_password(db: Session, token: str, new_password: str) -> bool:
        """Reset password using token"""
        user = (
            db.query(User)
            .filter(
                and_(
                    User.password_reset_token == token,
                    User.password_reset_expires > datetime.utcnow(),
                )
            )
            .first()
        )

        if not user:
            return False

        # Update password
        user.password_hash = UserAuthProvider.hash_password(new_password)
        user.password_reset_token = None
        user.password_reset_expires = None
        user.failed_login_attempts = 0
        user.locked_until = None

        db.commit()

        logger.info(f"Password reset for: {user.email}")
        return True

    @staticmethod
    def change_password(
        db: Session, user: User, current_password: str, new_password: str
    ) -> bool:
        """Change password for authenticated user"""
        if not UserAuthProvider.verify_password(current_password, user.password_hash):
            return False

        user.password_hash = UserAuthProvider.hash_password(new_password)
        db.commit()

        logger.info(f"Password changed for: {user.email}")
        return True

    @staticmethod
    def verify_email(db: Session, token: str) -> bool:
        """Verify email using token"""
        user = db.query(User).filter(User.email_verification_token == token).first()

        if not user:
            return False

        user.email_verified = True
        user.email_verification_token = None
        if user.status == "pending_verification":
            user.status = "active"

        db.commit()

        logger.info(f"Email verified for: {user.email}")
        return True

    @staticmethod
    def request_email_verification(db: Session, user: User) -> str:
        """Request new email verification token"""
        verification_token = secrets.token_urlsafe(32)
        user.email_verification_token = verification_token
        db.commit()

        return verification_token

    @staticmethod
    def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email"""
        return db.query(User).filter(User.email == email.lower()).first()
