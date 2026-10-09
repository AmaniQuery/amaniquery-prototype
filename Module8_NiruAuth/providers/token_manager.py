"""
Token Manager
Manages token lifecycle (generation, validation, refresh, revocation)
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from ..models.auth_models import OAuthToken
from .jwt_provider import JWTProvider
from ..config import config


class TokenManager:
    """Manages token lifecycle"""

    @staticmethod
    def create_token_pair(
        user_id: Optional[str] = None,
        integration_id: Optional[str] = None,
        scopes: Optional[list] = None,
    ) -> Dict[str, str]:
        """Create access and refresh token pair"""
        access_token = JWTProvider.generate_access_token(
            user_id, integration_id, scopes
        )
        refresh_token = JWTProvider.generate_refresh_token(user_id, integration_id)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        }

    @staticmethod
    def validate_access_token(token: str) -> Optional[Dict[str, Any]]:
        """Validate access token"""
        payload = JWTProvider.validate_token(token)
        if payload and payload.get("type") == "access_token":
            return payload
        return None

    @staticmethod
    def validate_refresh_token(token: str) -> Optional[Dict[str, Any]]:
        """Validate refresh token"""
        payload = JWTProvider.validate_token(token)
        if payload and payload.get("type") == "refresh_token":
            return payload
        return None

    @staticmethod
    def refresh_tokens(refresh_token: str) -> Optional[Dict[str, str]]:
        """Refresh access token using refresh token"""
        payload = TokenManager.validate_refresh_token(refresh_token)
        if not payload:
            return None

        # Generate new token pair
        return TokenManager.create_token_pair(
            user_id=payload.get("user_id"),
            integration_id=payload.get("integration_id"),
            scopes=payload.get("scopes"),
        )

    @staticmethod
    def revoke_token(db: Session, token: str) -> bool:
        """Revoke a token (for database-stored tokens)"""
        oauth_token = (
            db.query(OAuthToken).filter(OAuthToken.access_token == token).first()
        )

        if oauth_token:
            oauth_token.revoked = True
            oauth_token.revoked_at = datetime.utcnow()
            db.commit()
            return True

        return False
