"""
API Key Provider
Handles API key generation, validation, and management
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..models.auth_models import APIKey
from ..config import config


class APIKeyProvider:
    """Handles API key operations"""

    @staticmethod
    def hash_key(key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    @staticmethod
    def generate_api_key(prefix: str = None) -> Tuple[str, str]:
        """Generate a new API key"""
        if prefix is None:
            prefix = config.API_KEY_PREFIX_LIVE

        # Generate random key
        random_part = secrets.token_urlsafe(config.API_KEY_LENGTH)
        full_key = f"{prefix}{random_part}"

        # Get prefix for identification
        key_prefix = full_key[:12]  # First 12 characters

        return full_key, key_prefix

    @staticmethod
    def create_api_key(
        db: Session,
        user_id: Optional[str] = None,
        integration_id: Optional[str] = None,
        name: Optional[str] = None,
        scopes: Optional[list] = None,
        rate_limit_per_minute: int = None,
        rate_limit_per_hour: int = None,
        rate_limit_per_day: int = None,
        expires_at: Optional[datetime] = None,
        prefix: str = None,
    ) -> Tuple[str, APIKey]:
        """Create a new API key"""
        if user_id is None and integration_id is None:
            raise ValueError("Either user_id or integration_id must be provided")

        # Generate key
        full_key, key_prefix = APIKeyProvider.generate_api_key(prefix)
        key_hash = APIKeyProvider.hash_key(full_key)

        # Set defaults
        if rate_limit_per_minute is None:
            rate_limit_per_minute = config.RATE_LIMIT_DEFAULT_PER_MINUTE
        if rate_limit_per_hour is None:
            rate_limit_per_hour = config.RATE_LIMIT_DEFAULT_PER_HOUR
        if rate_limit_per_day is None:
            rate_limit_per_day = config.RATE_LIMIT_DEFAULT_PER_DAY

        # Create API key record
        api_key = APIKey(
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=name,
            user_id=user_id,
            integration_id=integration_id,
            scopes=scopes or [],
            rate_limit_per_minute=rate_limit_per_minute,
            rate_limit_per_hour=rate_limit_per_hour,
            rate_limit_per_day=rate_limit_per_day,
            expires_at=expires_at,
        )

        db.add(api_key)
        db.commit()
        db.refresh(api_key)

        return full_key, api_key

    @staticmethod
    def validate_api_key(db: Session, api_key: str) -> Optional[APIKey]:
        """Validate API key and return the key record"""
        # Extract prefix for faster lookup
        if len(api_key) < 12:
            return None

        key_prefix = api_key[:12]
        key_hash = APIKeyProvider.hash_key(api_key)

        # Query by prefix first (indexed), then verify hash
        api_key_record = (
            db.query(APIKey)
            .filter(
                and_(
                    APIKey.key_prefix == key_prefix,
                    APIKey.is_active == True,
                    (APIKey.expires_at == None)
                    | (APIKey.expires_at > datetime.utcnow()),
                )
            )
            .first()
        )

        if not api_key_record:
            return None

        # Verify hash
        if api_key_record.key_hash != key_hash:
            return None

        # Update last used
        api_key_record.last_used = datetime.utcnow()
        db.commit()

        return api_key_record

    @staticmethod
    def revoke_api_key(db: Session, api_key_id: str) -> bool:
        """Revoke an API key"""
        api_key = db.query(APIKey).filter(APIKey.id == api_key_id).first()

        if api_key:
            api_key.is_active = False
            db.commit()
            return True

        return False

    @staticmethod
    def rotate_api_key(db: Session, api_key_id: str) -> Tuple[str, APIKey]:
        """Rotate an API key - creates new key and revokes old"""
        old_key = db.query(APIKey).filter(APIKey.id == api_key_id).first()

        if not old_key:
            raise ValueError("API key not found")

        # Create new key with same settings
        full_key, new_key = APIKeyProvider.create_api_key(
            db=db,
            user_id=old_key.user_id,
            integration_id=old_key.integration_id,
            name=f"{old_key.name or 'API Key'} (rotated)",
            scopes=old_key.scopes,
            rate_limit_per_minute=old_key.rate_limit_per_minute,
            rate_limit_per_hour=old_key.rate_limit_per_hour,
            rate_limit_per_day=old_key.rate_limit_per_day,
            expires_at=old_key.expires_at,
        )

        # Revoke old key
        old_key.is_active = False
        db.commit()

        return full_key, new_key

    @staticmethod
    def get_api_key_by_id(db: Session, api_key_id: str) -> Optional[APIKey]:
        """Get API key by ID"""
        return db.query(APIKey).filter(APIKey.id == api_key_id).first()

    @staticmethod
    def list_user_api_keys(db: Session, user_id: str) -> list:
        """List all API keys for a user"""
        return (
            db.query(APIKey)
            .filter(and_(APIKey.user_id == user_id, APIKey.is_active == True))
            .all()
        )

    @staticmethod
    def list_integration_api_keys(db: Session, integration_id: str) -> list:
        """List all API keys for an integration"""
        return (
            db.query(APIKey)
            .filter(
                and_(APIKey.integration_id == integration_id, APIKey.is_active == True)
            )
            .all()
        )
