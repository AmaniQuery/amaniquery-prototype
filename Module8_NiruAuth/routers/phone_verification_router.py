"""
Phone Verification Router
Handles OTP sending and verification via TALKSASA
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional

from ..dependencies import get_db, get_current_user
from ..models.auth_models import User
from ..services.otp_service import get_otp_service
from ..providers.user_auth_provider import UserAuthProvider
from Module4_NiruAPI.config_manager import ConfigManager

router = APIRouter(prefix="/api/v1/auth/phone", tags=["Phone Verification"])


class SendOTPRequest(BaseModel):
    """Request to send OTP"""

    phone_number: str = Field(
        ..., description="Phone number in format +254712345678 or 0712345678"
    )
    purpose: str = Field(
        "verification", description="Purpose: verification, password_reset"
    )


class VerifyOTPRequest(BaseModel):
    """Request to verify OTP"""

    phone_number: str = Field(..., description="Phone number")
    otp: str = Field(..., min_length=6, max_length=6, description="6-digit OTP code")
    purpose: str = Field(
        "verification", description="Purpose: verification, password_reset"
    )


class ResendOTPRequest(BaseModel):
    """Request to resend OTP"""

    phone_number: str = Field(..., description="Phone number")
    purpose: str = Field(
        "verification", description="Purpose: verification, password_reset"
    )


def normalize_phone_number(phone: str) -> str:
    """Normalize phone number to +254 format"""
    phone = phone.strip().replace(" ", "").replace("-", "")

    # If starts with 0, replace with +254
    if phone.startswith("0"):
        phone = "+254" + phone[1:]
    # If doesn't start with +, add +254
    elif not phone.startswith("+"):
        phone = "+254" + phone

    return phone


def validate_phone_number(phone: str) -> bool:
    """Validate Kenyan phone number format"""
    normalized = normalize_phone_number(phone)
    # Kenyan numbers: +254 followed by 9 digits
    # Formats: +2547XXXXXXX or +2541XXXXXXX
    if normalized.startswith("+254") and len(normalized) == 13:
        # Check if remaining digits are valid
        digits = normalized[4:]
        # First digit can be 7 (mobile) or 1 (landline/special)
        return digits.isdigit() and digits[0] in ["7", "1"]
    return False


@router.post("/send-otp")
async def send_otp(request: SendOTPRequest, db: Session = Depends(get_db)):
    """Send OTP to phone number via TALKSASA"""
    # Normalize and validate phone number
    normalized_phone = normalize_phone_number(request.phone_number)

    if not validate_phone_number(normalized_phone):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid phone number format. Use format: +254712345678, +2541XXXXXXX, 0712345678, or 01XXXXXXX",
        )

    # Get OTP service
    config_manager = ConfigManager()
    otp_service = get_otp_service(config_manager)

    # Send OTP
    result = otp_service.send_otp_sms(normalized_phone, request.purpose)

    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("message", "Failed to send OTP"),
        )

    return {
        "status": "success",
        "message": "OTP sent successfully",
        "expires_in_minutes": result.get("expires_in_minutes", 5),
    }


@router.post("/verify-otp")
async def verify_otp(request: VerifyOTPRequest, db: Session = Depends(get_db)):
    """Verify OTP code"""
    # Normalize phone number
    normalized_phone = normalize_phone_number(request.phone_number)

    if not validate_phone_number(normalized_phone):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid phone number format",
        )

    # Get OTP service
    config_manager = ConfigManager()
    otp_service = get_otp_service(config_manager)

    # Verify OTP
    is_valid = otp_service.verify_otp(normalized_phone, request.otp, request.purpose)

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP code",
        )

    # If verification purpose, update user's phone_verified status
    if request.purpose == "verification":
        user = db.query(User).filter(User.phone_number == normalized_phone).first()
        if user:
            user.phone_verified = True
            if user.status == "pending_verification":
                user.status = "active"
            db.commit()

    # If password reset purpose, return the password reset token
    reset_token = None
    if request.purpose == "password_reset":
        user = db.query(User).filter(User.phone_number == normalized_phone).first()
        if user and user.password_reset_token:
            reset_token = user.password_reset_token

    response = {"status": "success", "message": "OTP verified successfully"}

    if reset_token:
        response["reset_token"] = reset_token

    return response


@router.post("/resend-otp")
async def resend_otp(request: ResendOTPRequest, db: Session = Depends(get_db)):
    """Resend OTP to phone number"""
    # Normalize phone number
    normalized_phone = normalize_phone_number(request.phone_number)

    if not validate_phone_number(normalized_phone):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid phone number format",
        )

    # Get OTP service
    config_manager = ConfigManager()
    otp_service = get_otp_service(config_manager)

    # Send OTP
    result = otp_service.send_otp_sms(normalized_phone, request.purpose)

    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("message", "Failed to resend OTP"),
        )

    return {
        "status": "success",
        "message": "OTP resent successfully",
        "expires_in_minutes": result.get("expires_in_minutes", 5),
    }


@router.post("/verify-phone")
async def verify_user_phone(
    request: VerifyOTPRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Verify phone number for authenticated user"""
    # Normalize phone number
    normalized_phone = normalize_phone_number(request.phone_number)

    if not validate_phone_number(normalized_phone):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid phone number format",
        )

    # Verify OTP
    config_manager = ConfigManager()
    otp_service = get_otp_service(config_manager)

    is_valid = otp_service.verify_otp(normalized_phone, request.otp, "verification")

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP code",
        )

    # Update user's phone number and verification status
    user.phone_number = normalized_phone
    user.phone_verified = True
    db.commit()

    return {"status": "success", "message": "Phone number verified successfully"}
