"""
OTP Service
Handles OTP generation, storage, validation, and SMS delivery via TALKSASA
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict
from sqlalchemy.orm import Session
from sqlalchemy import and_
from loguru import logger

from ..models.auth_models import User
from ..config import config
from Module4_NiruAPI.services.talksasa_service import TalksasaNotificationService


class OTPService:
    """Service for OTP generation and validation"""

    OTP_EXPIRY_MINUTES = 5
    OTP_LENGTH = 6
    MAX_OTP_ATTEMPTS_PER_HOUR = 3

    def __init__(self, config_manager=None):
        """Initialize OTP service with TALKSASA integration"""
        # Initialize TALKSASA service
        api_token = None
        sender_id = None
        if config_manager:
            api_token = config_manager.get_config("TALKSASA_API_TOKEN")
            sender_id = config_manager.get_config("TALKSASA_SENDER_ID")

        self.talksasa_service = TalksasaNotificationService(
            api_token=api_token, sender_id=sender_id
        )

        # In-memory OTP storage (in production, use Redis or database)
        self._otp_store: Dict[str, Dict] = {}

    def generate_otp(self) -> str:
        """Generate a 6-digit OTP code"""
        return f"{secrets.randbelow(1000000):06d}"

    def hash_otp(self, otp: str) -> str:
        """Hash OTP for storage"""
        return hashlib.sha256(otp.encode("utf-8")).hexdigest()

    def store_otp(
        self,
        identifier: str,  # phone number or email
        otp: str,
        purpose: str = "verification",  # verification, password_reset, etc.
    ) -> bool:
        """Store OTP with expiration"""
        otp_hash = self.hash_otp(otp)
        expires_at = datetime.utcnow() + timedelta(minutes=self.OTP_EXPIRY_MINUTES)

        key = f"{identifier}:{purpose}"

        # Check rate limiting
        if key in self._otp_store:
            last_sent = self._otp_store[key].get("last_sent")
            if last_sent:
                time_since_last = datetime.utcnow() - last_sent
                if time_since_last < timedelta(hours=1):
                    attempts = self._otp_store[key].get("attempts_in_hour", 0)
                    if attempts >= self.MAX_OTP_ATTEMPTS_PER_HOUR:
                        logger.warning(f"Rate limit exceeded for {identifier}")
                        return False

        self._otp_store[key] = {
            "otp_hash": otp_hash,
            "expires_at": expires_at,
            "purpose": purpose,
            "verified": False,
            "attempts": 0,
            "last_sent": datetime.utcnow(),
            "attempts_in_hour": self._otp_store.get(key, {}).get("attempts_in_hour", 0)
            + 1,
        }

        # Reset attempts counter if more than an hour has passed
        if key in self._otp_store:
            last_sent = self._otp_store[key].get("last_sent")
            if last_sent and (datetime.utcnow() - last_sent) >= timedelta(hours=1):
                self._otp_store[key]["attempts_in_hour"] = 1

        return True

    def verify_otp(
        self, identifier: str, otp: str, purpose: str = "verification"
    ) -> bool:
        """Verify OTP code"""
        key = f"{identifier}:{purpose}"

        if key not in self._otp_store:
            return False

        stored_data = self._otp_store[key]

        # Check if expired
        if datetime.utcnow() > stored_data["expires_at"]:
            # Clean up expired OTP
            del self._otp_store[key]
            return False

        # Check if already verified
        if stored_data.get("verified", False):
            return False

        # Increment attempts
        stored_data["attempts"] = stored_data.get("attempts", 0) + 1

        # Verify OTP
        otp_hash = self.hash_otp(otp)
        if stored_data["otp_hash"] == otp_hash:
            stored_data["verified"] = True
            return True

        return False

    def send_otp_sms(self, phone_number: str, purpose: str = "verification") -> Dict:
        """Generate and send OTP via TALKSASA SMS"""
        # Generate OTP
        otp = self.generate_otp()

        # Store OTP
        if not self.store_otp(phone_number, otp, purpose):
            return {
                "status": "error",
                "message": "Rate limit exceeded. Please try again later.",
            }

        # Create message based on purpose
        if purpose == "verification":
            message = f"Your AmaniQuery verification code is: {otp}. Valid for {self.OTP_EXPIRY_MINUTES} minutes."
        elif purpose == "password_reset":
            message = f"Your AmaniQuery password reset code is: {otp}. Valid for {self.OTP_EXPIRY_MINUTES} minutes."
        else:
            message = f"Your AmaniQuery code is: {otp}. Valid for {self.OTP_EXPIRY_MINUTES} minutes."

        # Send via TALKSASA
        result = self.talksasa_service.send_sms(phone_number, message)

        if result.get("status") == "success":
            logger.info(f"OTP sent to {phone_number}")
            return {
                "status": "success",
                "message": "OTP sent successfully",
                "expires_in_minutes": self.OTP_EXPIRY_MINUTES,
            }
        else:
            # Remove stored OTP if sending failed
            key = f"{phone_number}:{purpose}"
            if key in self._otp_store:
                del self._otp_store[key]

            return {
                "status": "error",
                "message": result.get("message", "Failed to send OTP"),
            }

    def cleanup_expired_otps(self):
        """Clean up expired OTPs from storage"""
        now = datetime.utcnow()
        expired_keys = [
            key for key, data in self._otp_store.items() if now > data["expires_at"]
        ]
        for key in expired_keys:
            del self._otp_store[key]

    def get_otp_info(
        self, identifier: str, purpose: str = "verification"
    ) -> Optional[Dict]:
        """Get OTP information without verifying"""
        key = f"{identifier}:{purpose}"
        if key not in self._otp_store:
            return None

        data = self._otp_store[key]
        if datetime.utcnow() > data["expires_at"]:
            return None

        return {
            "expires_at": data["expires_at"],
            "verified": data.get("verified", False),
            "attempts": data.get("attempts", 0),
        }


# Global OTP service instance
_otp_service: Optional[OTPService] = None


def get_otp_service(config_manager=None) -> OTPService:
    """Get or create OTP service instance"""
    global _otp_service
    if _otp_service is None:
        _otp_service = OTPService(config_manager)
    return _otp_service
