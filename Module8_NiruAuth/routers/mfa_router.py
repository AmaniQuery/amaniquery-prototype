"""
MFA/TOTP Router - Setup, verify, and disable multi-factor authentication
"""

import json
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List

from ..dependencies import get_db, get_current_user
from ..models.auth_models import User
from ..services.mfa_service import get_mfa_service
from ..services.security_audit import SecurityAudit

router = APIRouter(prefix="/api/v1/auth/mfa", tags=["MFA"])


class MFASetupResponse(BaseModel):
    secret: str
    qr_code: str
    backup_codes: List[str]


class MFAVerifyRequest(BaseModel):
    token: str = Field(..., min_length=6, max_length=6)


class MFAVerifyResponse(BaseModel):
    success: bool
    message: str


class MFADisableRequest(BaseModel):
    password: str


@router.get("/setup", response_model=MFASetupResponse)
async def setup_mfa(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate TOTP secret and QR code for authenticator app setup"""
    mfa = get_mfa_service()
    secret = mfa.generate_secret()
    qr_code = mfa.generate_qr_code(user.email, secret)
    backup_codes = mfa.generate_backup_codes()

    user.mfa_secret = secret
    user.mfa_backup_codes = json.dumps(backup_codes)
    db.commit()

    logger = __import__("logging").getLogger(__name__)
    logger.info(f"MFA setup initiated for user {user.id}")

    return MFASetupResponse(
        secret=secret,
        qr_code=qr_code,
        backup_codes=backup_codes,
    )


@router.post("/verify", response_model=MFAVerifyResponse)
async def verify_mfa(
    request: MFAVerifyRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Verify TOTP token and enable MFA"""
    if not user.mfa_secret:
        raise HTTPException(
            status_code=400, detail="MFA not set up. Call GET /setup first."
        )

    mfa = get_mfa_service()
    if mfa.verify_token(user.mfa_secret, request.token):
        user.mfa_enabled = True
        db.commit()
        logger = __import__("logging").getLogger(__name__)
        logger.info(f"MFA enabled for user {user.id}")
        return MFAVerifyResponse(
            success=True, message="MFA has been enabled successfully."
        )
    else:
        SecurityAudit.suspicious_activity(
            user.id, getattr(user, "last_login_ip", None), f"mfa_verify_failed"
        )
        raise HTTPException(status_code=400, detail="Invalid token. Please try again.")


@router.post("/disable", response_model=MFAVerifyResponse)
async def disable_mfa(
    request: MFADisableRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Disable MFA (requires password confirmation)"""
    from ..providers.user_auth_provider import UserAuthProvider

    if not UserAuthProvider.verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=403, detail="Invalid password")
    user.mfa_secret = None
    user.mfa_enabled = False
    user.mfa_backup_codes = None
    db.commit()
    logger = __import__("logging").getLogger(__name__)
    logger.info(f"MFA disabled for user {user.id}")
    return MFAVerifyResponse(success=True, message="MFA has been disabled.")


@router.post("/token", response_model=MFAVerifyResponse)
async def verify_mfa_token(
    request: MFAVerifyRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Verify a TOTP token (for challenge during login)"""
    if not user.mfa_enabled or not user.mfa_secret:
        raise HTTPException(status_code=400, detail="MFA is not enabled")

    mfa = get_mfa_service()
    if mfa.verify_token(user.mfa_secret, request.token):
        return MFAVerifyResponse(success=True, message="Token verified.")
    raise HTTPException(status_code=400, detail="Invalid token")
