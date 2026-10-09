"""
JWT Provider
Handles JWT token generation and validation
"""

import jwt
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from ..config import config


class JWTProvider:
    """Handles JWT token operations"""

    @staticmethod
    def encode_token(
        payload: Dict[str, Any], expires_minutes: Optional[int] = None
    ) -> str:
        """Encode a JWT token"""
        if expires_minutes is None:
            expires_minutes = config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES

        # Add expiration
        exp = datetime.utcnow() + timedelta(minutes=expires_minutes)
        payload["exp"] = exp
        payload["iat"] = datetime.utcnow()

        # Encode token
        token = jwt.encode(
            payload, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM
        )

        return token

    @staticmethod
    def decode_token(token: str) -> Dict[str, Any]:
        """Decode and validate a JWT token"""
        try:
            payload = jwt.decode(
                token, config.JWT_SECRET_KEY, algorithms=[config.JWT_ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {str(e)}")

    @staticmethod
    def generate_access_token(
        user_id: Optional[str] = None,
        integration_id: Optional[str] = None,
        scopes: Optional[list] = None,
    ) -> str:
        """Generate a JWT access token"""
        payload = {
            "type": "access_token",
            "user_id": user_id,
            "integration_id": integration_id,
            "scopes": scopes or [],
        }
        return JWTProvider.encode_token(
            payload, expires_minutes=config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        )

    @staticmethod
    def generate_refresh_token(
        user_id: Optional[str] = None, integration_id: Optional[str] = None
    ) -> str:
        """Generate a JWT refresh token"""
        payload = {
            "type": "refresh_token",
            "user_id": user_id,
            "integration_id": integration_id,
        }
        return JWTProvider.encode_token(
            payload, expires_minutes=config.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60
        )

    @staticmethod
    def validate_token(token: str) -> Optional[Dict[str, Any]]:
        """Validate a JWT token and return payload"""
        try:
            return JWTProvider.decode_token(token)
        except:
            return None
