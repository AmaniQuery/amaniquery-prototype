"""
User Authentication Router
Handles user registration, login, profile management
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request, UploadFile, File
from sqlalchemy.orm import Session
from typing import Optional
from pathlib import Path

from ..models.pydantic_models import (
    UserRegister,
    UserLogin,
    UserResponse,
    UserProfileUpdate,
    PasswordChange,
    PasswordResetRequest,
    PasswordReset,
    PasswordResetRequestResponse,
    EmailVerificationRequest,
    SessionResponse,
    MFARequiredResponse,
    MFAChallengeRequest,
)
from ..providers.user_auth_provider import UserAuthProvider
from ..providers.session_provider import SessionProvider
from ..dependencies import get_db, get_current_user
from ..models.auth_models import User
from ..services.mfa_service import get_mfa_service

# In-memory MFA challenge store: mfa_token -> {user_id, expires_at}
_mfa_challenges: dict = {}

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


@router.post(
    "/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def register(
    user_data: UserRegister, request: Request, db: Session = Depends(get_db)
):
    """Register a new user (phone verification required after registration)"""
    try:
        user = UserAuthProvider.create_user(
            db=db,
            email=user_data.email,
            password=user_data.password,
            name=user_data.name,
            phone_number=user_data.phone_number,
        )

        # Send email verification email
        if user.email_verification_token:
            from ..services.email_service import get_email_service

            try:
                email_service = get_email_service()
                email_service.send_verification_email(
                    to_email=user.email,
                    verification_token=user.email_verification_token,
                )
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Failed to send verification email: {e}")
                # Don't fail registration if email sending fails

        return UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            status=user.status,
            email_verified=user.email_verified,
            last_login=user.last_login,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/login", response_model=None)
async def login(login_data: UserLogin, request: Request, db: Session = Depends(get_db)):
    """Login user and create session"""
    try:
        user = UserAuthProvider.authenticate_user(
            db=db,
            email=login_data.email,
            password=login_data.password,
            ip_address=request.client.host if request.client else None,
        )

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        # Check if MFA is required
        if user.mfa_enabled:
            import secrets

            mfa_token = secrets.token_urlsafe(32)
            _mfa_challenges[mfa_token] = {
                "user_id": user.id,
                "expires_at": __import__("datetime").datetime.utcnow()
                + __import__("datetime").timedelta(minutes=5),
            }
            return MFARequiredResponse(
                mfa_token=mfa_token,
                message="MFA verification required. Submit TOTP code to /api/v1/auth/mfa/challenge",
            )

        # Create session
        try:
            session_token, session = SessionProvider.create_session(
                db=db,
                user=user,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                f"Failed to create session for user {user.id}: {e}", exc_info=True
            )
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create session. Please try again.",
            )

        # Get user roles
        from ..authorization.role_manager import RoleManager

        user_roles = RoleManager.get_user_roles(db, user.id)
        role_names = [role.name for role in user_roles]

        # Create user response with roles
        user_response = UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            status=user.status,
            email_verified=user.email_verified,
            last_login=user.last_login,
            profile_image_url=user.profile_image_url,
            created_at=user.created_at,
            updated_at=user.updated_at,
            roles=role_names,  # Include roles directly in the model
        )

        return SessionResponse(
            session_token=session_token,
            refresh_token=None,  # Session-based, no separate refresh token
            expires_at=session.expires_at,
            user=user_response,  # UserResponse now includes roles
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


@router.post("/login/mfa-challenge", response_model=SessionResponse)
async def mfa_challenge(
    challenge_data: MFAChallengeRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Complete MFA challenge and create session"""
    from datetime import datetime, timedelta

    challenge = _mfa_challenges.pop(challenge_data.mfa_token, None)
    if not challenge:
        raise HTTPException(
            status_code=400, detail="Invalid or expired MFA token. Please login again."
        )

    if datetime.utcnow() > challenge["expires_at"]:
        raise HTTPException(
            status_code=400, detail="MFA token expired. Please login again."
        )

    user = db.query(User).filter(User.id == challenge["user_id"]).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    mfa = get_mfa_service()
    if not mfa.verify_token(user.mfa_secret, challenge_data.totp_code):
        from ..services.security_audit import SecurityAudit

        SecurityAudit.suspicious_activity(
            user.id,
            request.client.host if request.client else None,
            "mfa_challenge_failed",
        )
        raise HTTPException(status_code=400, detail="Invalid TOTP code")

    session_token, session = SessionProvider.create_session(
        db=db,
        user=user,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    from ..authorization.role_manager import RoleManager

    user_roles = RoleManager.get_user_roles(db, user.id)
    role_names = [role.name for role in user_roles]

    user_response = UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        status=user.status,
        email_verified=user.email_verified,
        last_login=user.last_login,
        profile_image_url=user.profile_image_url,
        created_at=user.created_at,
        updated_at=user.updated_at,
        roles=role_names,
    )

    return SessionResponse(
        session_token=session_token,
        expires_at=session.expires_at,
        user=user_response,
    )


@router.post("/logout")
async def logout(request: Request, db: Session = Depends(get_db)):
    """Logout user and invalidate session"""
    session_token = request.cookies.get("session_token") or request.headers.get(
        "X-Session-Token"
    )

    if session_token:
        SessionProvider.revoke_session(db, session_token)

    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Get current user profile"""
    from ..authorization.role_manager import RoleManager

    user_roles = RoleManager.get_user_roles(db, user.id)
    role_names = [role.name for role in user_roles]

    # Create response with roles
    response = UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        status=user.status,
        email_verified=user.email_verified,
        last_login=user.last_login,
        profile_image_url=user.profile_image_url,
        created_at=user.created_at,
        updated_at=user.updated_at,
        roles=role_names,  # Include roles directly
    )
    return response.model_dump()


@router.put("/me", response_model=UserResponse)
async def update_user_profile(
    profile_data: UserProfileUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update user profile"""
    if profile_data.name is not None:
        user.name = profile_data.name
    if profile_data.email is not None:
        # Check if email already exists
        existing = (
            db.query(User).filter(User.email == profile_data.email.lower()).first()
        )
        if existing and existing.id != user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Email already in use"
            )
        user.email = profile_data.email.lower()
    if profile_data.profile_image_url is not None:
        user.profile_image_url = profile_data.profile_image_url

    db.commit()
    db.refresh(user)

    # Get user roles for response
    from ..authorization.role_manager import RoleManager

    user_roles = RoleManager.get_user_roles(db, user.id)
    role_names = [role.name for role in user_roles]

    user_response = UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        status=user.status,
        email_verified=user.email_verified,
        last_login=user.last_login,
        profile_image_url=user.profile_image_url,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )
    # Add roles to response
    response_dict = user_response.model_dump()
    response_dict["roles"] = role_names
    return response_dict


@router.post("/me/profile-image", response_model=UserResponse)
async def upload_profile_image(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload profile image to Cloudinary"""
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}",
        )

    # Validate file size (5MB limit)
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 5MB limit",
        )

    try:
        # Upload to Cloudinary
        from Module4_NiruAPI.services.cloudinary_service import CloudinaryService

        cloudinary_service = CloudinaryService()

        # Get file extension
        file_ext = Path(file.filename).suffix or ".jpg"
        filename = f"profile_{user.id}{file_ext}"

        # Upload to Cloudinary using upload_bytes
        result = cloudinary_service.upload_bytes(
            file_content=file_content,
            filename=filename,
            session_id=user.id,
            resource_type="image",
            folder="user_profiles",
        )

        cloudinary_url = result.get("secure_url") or result.get("url")

        # Update user profile
        user.profile_image_url = cloudinary_url
        db.commit()
        db.refresh(user)

        # Get user roles for response
        from ..authorization.role_manager import RoleManager

        user_roles = RoleManager.get_user_roles(db, user.id)
        role_names = [role.name for role in user_roles]

        user_response = UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            status=user.status,
            email_verified=user.email_verified,
            last_login=user.last_login,
            profile_image_url=user.profile_image_url,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )
        # Add roles to response
        response_dict = user_response.model_dump()
        response_dict["roles"] = role_names
        return response_dict

    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloudinary service not available",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload profile image: {str(e)}",
        )


@router.post("/password/change")
async def change_password(
    password_data: PasswordChange,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Change password"""
    success = UserAuthProvider.change_password(
        db=db,
        user=user,
        current_password=password_data.current_password,
        new_password=password_data.new_password,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    return {"message": "Password changed successfully"}


@router.post("/password/reset-request", response_model=PasswordResetRequestResponse)
async def request_password_reset(
    reset_data: PasswordResetRequest, db: Session = Depends(get_db)
):
    """Request password reset - returns masked phone number for OTP verification"""
    # Get user by email
    user = UserAuthProvider.get_user_by_email(db, reset_data.email)

    if not user:
        # Don't reveal if email exists for security, but still return success message
        return PasswordResetRequestResponse(
            message="If the email exists, a password reset OTP will be sent to your phone number",
            phone_number=None,
        )

    # Generate password reset token
    token = UserAuthProvider.request_password_reset(db, reset_data.email)

    # Send OTP to user's phone number if available
    if user.phone_number:
        from Module8_NiruAuth.services.otp_service import get_otp_service
        from Module4_NiruAPI.config_manager import ConfigManager

        try:
            config_manager = ConfigManager()
            otp_service = get_otp_service(config_manager)
            otp_service.send_otp_sms(
                phone_number=user.phone_number, purpose="password_reset"
            )
        except Exception as e:
            # Log error but don't fail the request
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to send OTP for password reset: {e}")

    # Also send password reset email with token
    from ..services.email_service import get_email_service

    try:
        email_service = get_email_service()
        email_service.send_password_reset_email(to_email=user.email, reset_token=token)
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Failed to send password reset email: {e}")
        # Don't fail the request if email sending fails

    # Mask phone number for security (show only last 4 digits)
    masked_phone = None
    if user.phone_number:
        phone = user.phone_number
        if len(phone) > 4:
            masked_phone = "*" * (len(phone) - 4) + phone[-4:]
        else:
            masked_phone = "*" * len(phone)

    return PasswordResetRequestResponse(
        message="Password reset OTP will be sent to your phone number",
        phone_number=masked_phone,
    )


@router.post("/password/reset")
async def reset_password(reset_data: PasswordReset, db: Session = Depends(get_db)):
    """Reset password with token"""
    success = UserAuthProvider.reset_password(
        db=db, token=reset_data.token, new_password=reset_data.new_password
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    return {"message": "Password reset successfully"}


@router.post("/email/verify-request")
async def request_email_verification(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Request email verification"""
    token = UserAuthProvider.request_email_verification(db, user)

    # Send email with verification token
    from ..services.email_service import get_email_service

    try:
        email_service = get_email_service()
        success = email_service.send_verification_email(
            to_email=user.email, verification_token=token
        )
        if success:
            return {"message": "Verification email sent successfully"}
        else:
            return {
                "message": "Verification email requested, but sending failed. Please try again."
            }
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Failed to send verification email: {e}")
        return {
            "message": "Verification email requested, but sending failed. Please try again."
        }


@router.get("/email/verify")
async def verify_email(token: str, db: Session = Depends(get_db)):
    """Verify email with token"""
    # Get user before token is cleared (verify_email clears the token)
    user = db.query(User).filter(User.email_verification_token == token).first()

    success = UserAuthProvider.verify_email(db, token)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token",
        )

    # Send welcome email after successful verification
    if user:
        from ..services.email_service import get_email_service

        try:
            email_service = get_email_service()
            email_service.send_welcome_email(
                to_email=user.email, name=user.name or "User"
            )
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to send welcome email: {e}")
            # Don't fail verification if welcome email fails

    return {"message": "Email verified successfully"}
