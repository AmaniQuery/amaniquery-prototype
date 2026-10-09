"""
MFA/TOTP Service for Time-based One-Time Password authentication
"""

import os
import io
import base64
from typing import Optional, Tuple
from loguru import logger
import pyotp
import qrcode


class MFAService:
    """TOTP-based Multi-Factor Authentication service"""

    def __init__(self, issuer_name: str = "AmaniQuery"):
        self.issuer_name = os.getenv("MFA_ISSUER_NAME", issuer_name)

    def generate_secret(self) -> str:
        """Generate a new TOTP secret"""
        return pyotp.random_base32()

    def get_provisioning_uri(self, email: str, secret: str) -> str:
        """Get otpauth URI for QR code generation"""
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=email, issuer_name=self.issuer_name)

    def generate_qr_code(self, email: str, secret: str) -> str:
        """Generate QR code as base64 PNG for authenticator app setup"""
        uri = self.get_provisioning_uri(email, secret)
        img = qrcode.make(uri)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def verify_token(self, secret: str, token: str) -> bool:
        """Verify a TOTP token against the secret"""
        if not secret or not token:
            return False
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)

    def generate_backup_codes(self, count: int = 8) -> list:
        """Generate one-time backup recovery codes"""
        import secrets

        return [secrets.token_hex(4).upper() for _ in range(count)]


_mfa_service: Optional[MFAService] = None


def get_mfa_service() -> MFAService:
    global _mfa_service
    if _mfa_service is None:
        _mfa_service = MFAService()
    return _mfa_service
