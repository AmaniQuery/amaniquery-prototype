"""
Security Audit Logger
Tracks security events for compliance and threat detection
"""

import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
from loguru import logger


class SecurityEvent:
    LOGIN_SUCCESS = "login.success"
    LOGIN_FAILURE = "login.failure"
    LOGIN_BLOCKED = "login.blocked"
    PASSWORD_RESET = "password.reset"
    PASSWORD_CHANGE = "password.change"
    EMAIL_VERIFIED = "email.verified"
    SESSION_CREATED = "session.created"
    SESSION_REVOKED = "session.revoked"
    SESSION_EXPIRED = "session.expired"
    API_KEY_CREATED = "api_key.created"
    API_KEY_REVOKED = "api_key.revoked"
    OAUTH_CLIENT_CREATED = "oauth.client.created"
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"
    ACCOUNT_LOCKED = "account.locked"
    ACCOUNT_UNLOCKED = "account.unlocked"
    SUSPICIOUS_ACTIVITY = "suspicious.activity"
    MFA_FAILURE = "mfa.failure"
    TOKEN_REFRESHED = "token.refreshed"


class SecurityAudit:
    """Logs security events to structured JSON for analysis"""

    LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
    LOG_FILE = os.path.join(LOG_DIR, "security_audit.jsonl")

    @classmethod
    def _ensure_log_dir(cls):
        os.makedirs(cls.LOG_DIR, exist_ok=True)

    @classmethod
    def log(
        cls,
        event: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log a security event"""
        cls._ensure_log_dir()
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "user_id": user_id,
            "ip_address": ip_address,
            "details": details or {},
        }
        try:
            with open(cls.LOG_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write audit log: {e}")

        logger.info(f"[SECURITY] {event} | user={user_id} | ip={ip_address}")

    @classmethod
    def login_success(cls, user_id: str, ip_address: Optional[str] = None):
        cls.log(SecurityEvent.LOGIN_SUCCESS, user_id, ip_address)

    @classmethod
    def login_failure(
        cls,
        email: str,
        ip_address: Optional[str] = None,
        reason: str = "invalid_password",
    ):
        cls.log(
            SecurityEvent.LOGIN_FAILURE,
            None,
            ip_address,
            {"email": email, "reason": reason},
        )

    @classmethod
    def account_locked(
        cls, user_id: str, ip_address: Optional[str] = None, attempts: int = 0
    ):
        cls.log(
            SecurityEvent.ACCOUNT_LOCKED,
            user_id,
            ip_address,
            {"failed_attempts": attempts},
        )

    @classmethod
    def rate_limit_exceeded(
        cls, user_id: Optional[str], ip_address: Optional[str], endpoint: str
    ):
        cls.log(
            SecurityEvent.RATE_LIMIT_EXCEEDED,
            user_id,
            ip_address,
            {"endpoint": endpoint},
        )

    @classmethod
    def suspicious_activity(
        cls, user_id: Optional[str], ip_address: Optional[str], reason: str
    ):
        cls.log(
            SecurityEvent.SUSPICIOUS_ACTIVITY, user_id, ip_address, {"reason": reason}
        )
