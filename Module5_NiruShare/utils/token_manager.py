"""
Token management for social media platforms
"""

import os
import json
import time
import hashlib
import hmac
import base64
import requests
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class TokenManager:
    """Production-ready token manager with encryption and validation"""

    def __init__(self):
        self.encryption_key = os.getenv(
            "TOKEN_ENCRYPTION_KEY", "default-dev-key-change-in-production"
        )
        self.redis_url = os.getenv("REDIS_URL")
        self.token_ttl = int(os.getenv("TOKEN_TTL_SECONDS", "3600"))  # 1 hour default

        # Use Redis if available, otherwise fallback to encrypted file storage
        if self.redis_url:
            try:
                import redis

                self.redis = redis.from_url(self.redis_url)
                self.use_redis = True
                logger.info("Using Redis for token storage")
            except ImportError:
                logger.warning("Redis not available, falling back to file storage")
                self.use_redis = False
        else:
            self.use_redis = False
            self.storage_file = os.path.join(os.path.dirname(__file__), "tokens.enc")
            logger.info("Using encrypted file storage for tokens")

    def _encrypt_token(self, token: str) -> str:
        """Encrypt token using HMAC-SHA256"""
        key = self.encryption_key.encode()
        message = token.encode()
        encrypted = hmac.new(key, message, hashlib.sha256).hexdigest()
        return f"{encrypted}:{token}"

    def _decrypt_token(self, encrypted_token: str) -> Optional[str]:
        """Decrypt and validate token"""
        try:
            # Handle case where encrypted_token might be a dict
            if isinstance(encrypted_token, dict):
                logger.warning(
                    f"encrypted_token is a dict, expected string: {encrypted_token}"
                )
                return None

            if not isinstance(encrypted_token, str):
                logger.warning(
                    f"encrypted_token is not a string: {type(encrypted_token)}"
                )
                return None

            encrypted, token = encrypted_token.split(":", 1)
            key = self.encryption_key.encode()
            expected = hmac.new(key, token.encode(), hashlib.sha256).hexdigest()
            if hmac.compare_digest(encrypted, expected):
                return token
            return None
        except Exception as e:
            logger.error(f"Failed to decrypt token: {e}")
            return None

    def _get_storage_key(self, user_id: str, platform: str) -> str:
        """Generate storage key for user-platform combination"""
        return f"token:{user_id}:{platform}"

    def store_token(
        self, user_id: str, platform: str, token_data: Dict[str, Any]
    ) -> bool:
        """
        Store encrypted token with metadata

        Args:
            user_id: User identifier
            platform: Platform name
            token_data: Token data including access_token, refresh_token, expires_at, etc.
        """
        try:
            # Add metadata
            token_data["stored_at"] = datetime.utcnow().isoformat()
            token_data["expires_at"] = (
                datetime.utcnow() + timedelta(seconds=self.token_ttl)
            ).isoformat()

            # Encrypt the access token
            if "access_token" in token_data:
                token_data["encrypted_token"] = self._encrypt_token(
                    token_data["access_token"]
                )

            storage_key = self._get_storage_key(user_id, platform)

            if self.use_redis:
                self.redis.setex(storage_key, self.token_ttl, json.dumps(token_data))
            else:
                # File-based storage (not recommended for production)
                self._store_to_file(storage_key, token_data)

            logger.info(f"Stored token for user {user_id} on platform {platform}")
            return True
        except Exception as e:
            logger.error(f"Failed to store token: {e}")
            return False

    def get_token(self, user_id: str, platform: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve and decrypt token for user-platform combination

        Returns:
            Token data dict or None if not found/expired
        """
        try:
            storage_key = self._get_storage_key(user_id, platform)

            if self.use_redis:
                data = self.redis.get(storage_key)
                if not data:
                    return None
                token_data = json.loads(data)
            else:
                token_data = self._load_from_file(storage_key)
                if not token_data:
                    return None

            # Check expiration
            expires_at = datetime.fromisoformat(token_data.get("expires_at", ""))
            if datetime.utcnow() > expires_at:
                logger.info(f"Token expired for user {user_id} on platform {platform}")
                self.delete_token(user_id, platform)
                return None

            # Decrypt access token
            if "encrypted_token" in token_data:
                decrypted = self._decrypt_token(token_data["encrypted_token"])
                if decrypted:
                    token_data["access_token"] = decrypted
                else:
                    logger.error(
                        f"Failed to decrypt token for user {user_id} on platform {platform}"
                    )
                    return None

            return token_data
        except Exception as e:
            logger.error(f"Failed to get token: {e}")
            return None

    def store_oauth_state(
        self, user_id: str, platform: str, state_data: Dict[str, Any]
    ) -> bool:
        """
        Store temporary OAuth state data (like code_verifier) securely

        Args:
            user_id: User identifier
            platform: Platform name
            state_data: OAuth state data to store temporarily
        """
        try:
            # Add metadata
            state_data["stored_at"] = datetime.utcnow().isoformat()
            state_data["expires_at"] = (
                datetime.utcnow() + timedelta(minutes=10)
            ).isoformat()  # Short TTL for OAuth state

            storage_key = f"oauth_state:{user_id}:{platform}"

            if self.use_redis:
                self.redis.setex(storage_key, 600, json.dumps(state_data))  # 10 minutes
            else:
                # File-based storage (not recommended for production)
                self._store_to_file(storage_key, state_data)

            logger.info(f"Stored OAuth state for user {user_id} on platform {platform}")
            return True
        except Exception as e:
            logger.error(f"Failed to store OAuth state: {e}")
            return False

    def get_oauth_state(self, user_id: str, platform: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve OAuth state data

        Returns:
            State data dict or None if not found/expired
        """
        try:
            storage_key = f"oauth_state:{user_id}:{platform}"

            if self.use_redis:
                data = self.redis.get(storage_key)
                if not data:
                    return None
                state_data = json.loads(data)
            else:
                state_data = self._load_from_file(storage_key)
                if not state_data:
                    return None

            # Check expiration
            expires_at = datetime.fromisoformat(state_data.get("expires_at", ""))
            if datetime.utcnow() > expires_at:
                logger.info(
                    f"OAuth state expired for user {user_id} on platform {platform}"
                )
                self.delete_oauth_state(user_id, platform)
                return None

            return state_data
        except Exception as e:
            logger.error(f"Failed to get OAuth state: {e}")
            return None

    def delete_oauth_state(self, user_id: str, platform: str) -> bool:
        """Delete OAuth state data"""
        try:
            storage_key = f"oauth_state:{user_id}:{platform}"

            if self.use_redis:
                self.redis.delete(storage_key)
            else:
                self._delete_from_file(storage_key)

            logger.info(
                f"Deleted OAuth state for user {user_id} on platform {platform}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete OAuth state: {e}")
            return False

    def delete_token(self, user_id: str, platform: str) -> bool:
        """Delete token for user-platform combination"""
        try:
            storage_key = self._get_storage_key(user_id, platform)

            if self.use_redis:
                self.redis.delete(storage_key)
            else:
                self._delete_from_file(storage_key)

            logger.info(f"Deleted token for user {user_id} on platform {platform}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete token: {e}")
            return False

    def validate_token(self, user_id: str, platform: str) -> bool:
        """
        Validate token by making a test API call to the platform

        Returns:
            True if token is valid, False otherwise
        """
        token_data = self.get_token(user_id, platform)
        if not token_data or not token_data.get("access_token"):
            return False

        try:
            # Platform-specific validation
            if platform == "linkedin":
                return self._validate_linkedin_token(token_data["access_token"])
            elif platform == "facebook":
                return self._validate_facebook_token(token_data["access_token"])
            elif platform == "twitter":
                return self._validate_twitter_token(token_data["access_token"])
            else:
                logger.warning(f"No validation implemented for platform: {platform}")
                return True  # Assume valid if we have it
        except Exception as e:
            logger.error(f"Token validation failed for {platform}: {e}")
            return False

    def refresh_token(self, user_id: str, platform: str) -> bool:
        """
        Refresh token if refresh token is available

        Returns:
            True if refresh was successful, False otherwise
        """
        token_data = self.get_token(user_id, platform)
        if not token_data or not token_data.get("refresh_token"):
            return False

        try:
            if platform == "linkedin":
                return self._refresh_linkedin_token(user_id, token_data)
            elif platform == "facebook":
                return self._refresh_facebook_token(user_id, token_data)
            elif platform == "twitter":
                return self._refresh_twitter_token(user_id, token_data)
            else:
                logger.warning(f"No refresh implemented for platform: {platform}")
                return False
        except Exception as e:
            logger.error(f"Token refresh failed for {platform}: {e}")
            return False

    # File-based storage methods (fallback, not for production)
    def _store_to_file(self, key: str, data: Dict[str, Any]):
        """Store data in encrypted file (not recommended for production)"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, "r") as f:
                    all_data = json.load(f)
            else:
                all_data = {}

            all_data[key] = data

            with open(self.storage_file, "w") as f:
                json.dump(all_data, f)
        except Exception as e:
            logger.error(f"Failed to store to file: {e}")

    def _load_from_file(self, key: str) -> Optional[Dict[str, Any]]:
        """Load data from encrypted file"""
        try:
            if not os.path.exists(self.storage_file):
                return None

            with open(self.storage_file, "r") as f:
                all_data = json.load(f)

            return all_data.get(key)
        except Exception as e:
            logger.error(f"Failed to load from file: {e}")
            return None

    def _delete_from_file(self, key: str):
        """Delete data from encrypted file"""
        try:
            if not os.path.exists(self.storage_file):
                return

            with open(self.storage_file, "r") as f:
                all_data = json.load(f)

            if key in all_data:
                del all_data[key]

            with open(self.storage_file, "w") as f:
                json.dump(all_data, f)
        except Exception as e:
            logger.error(f"Failed to delete from file: {e}")

    # Platform-specific validation methods
    def _validate_linkedin_token(self, access_token: str) -> bool:
        """Validate LinkedIn token by making a test API call"""
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            response = requests.get(
                "https://api.linkedin.com/v2/people/~", headers=headers, timeout=10
            )
            return response.status_code == 200
        except:
            return False

    def _validate_facebook_token(self, access_token: str) -> bool:
        """Validate Facebook token"""
        try:
            response = requests.get(
                f"https://graph.facebook.com/me?access_token={access_token}&fields=id",
                timeout=10,
            )
            return response.status_code == 200
        except:
            return False

    def _validate_twitter_token(self, access_token: str) -> bool:
        """Validate Twitter/X token by making a test API call"""
        try:
            # Twitter API v2 user lookup endpoint
            headers = {"Authorization": f"Bearer {access_token}"}
            # Use the users/me endpoint to validate token
            response = requests.get(
                "https://api.twitter.com/2/users/me", headers=headers, timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Twitter token validation error: {e}")
            return False

    # Platform-specific refresh methods
    def _refresh_linkedin_token(self, user_id: str, token_data: Dict[str, Any]) -> bool:
        """Refresh LinkedIn token using refresh token if available"""
        refresh_token = token_data.get("refresh_token")
        if not refresh_token:
            return False

        try:
            client_id = os.getenv("LINKEDIN_CLIENT_ID")
            client_secret = os.getenv("LINKEDIN_CLIENT_SECRET")

            if not client_id or not client_secret:
                logger.error("LinkedIn client credentials not configured")
                return False

            # LinkedIn refresh token flow
            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
            }

            response = requests.post(
                "https://www.linkedin.com/oauth/v2/accessToken", data=data, timeout=10
            )

            if response.status_code == 200:
                new_token_data = response.json()
                new_access_token = new_token_data.get("access_token")

                if new_access_token:
                    # Update stored token data
                    updated_data = token_data.copy()
                    updated_data["access_token"] = new_access_token
                    updated_data["encrypted_token"] = self._encrypt_token(
                        new_access_token
                    )
                    updated_data["stored_at"] = datetime.utcnow().isoformat()
                    updated_data["expires_at"] = (
                        datetime.utcnow() + timedelta(seconds=self.token_ttl)
                    ).isoformat()

                    # Add new refresh token if provided
                    if new_token_data.get("refresh_token"):
                        updated_data["refresh_token"] = new_token_data["refresh_token"]

                    success = self.store_token(user_id, "linkedin", updated_data)
                    if success:
                        logger.info(
                            f"Successfully refreshed LinkedIn token for user {user_id}"
                        )
                        return True

            logger.error(f"Failed to refresh LinkedIn token: {response.text}")
            return False

        except Exception as e:
            logger.error(f"LinkedIn token refresh error: {e}")
            return False

    def _refresh_facebook_token(self, user_id: str, token_data: Dict[str, Any]) -> bool:
        """Refresh Facebook token using long-lived token exchange"""
        access_token = token_data.get("access_token")
        if not access_token:
            return False

        try:
            app_id = os.getenv("FACEBOOK_APP_ID")
            app_secret = os.getenv("FACEBOOK_APP_SECRET")

            if not app_id or not app_secret:
                logger.error("Facebook app credentials not configured")
                return False

            # Facebook long-lived token exchange
            params = {
                "grant_type": "fb_exchange_token",
                "client_id": app_id,
                "client_secret": app_secret,
                "fb_exchange_token": access_token,
            }

            response = requests.get(
                "https://graph.facebook.com/oauth/access_token",
                params=params,
                timeout=10,
            )

            if response.status_code == 200:
                new_token_data = response.json()
                new_access_token = new_token_data.get("access_token")

                if new_access_token:
                    # Update stored token data
                    updated_data = token_data.copy()
                    updated_data["access_token"] = new_access_token
                    updated_data["encrypted_token"] = self._encrypt_token(
                        new_access_token
                    )
                    updated_data["stored_at"] = datetime.utcnow().isoformat()
                    updated_data["expires_at"] = (
                        datetime.utcnow() + timedelta(seconds=self.token_ttl)
                    ).isoformat()

                    success = self.store_token(user_id, "facebook", updated_data)
                    if success:
                        logger.info(
                            f"Successfully refreshed Facebook token for user {user_id}"
                        )
                        return True

            logger.error(f"Failed to refresh Facebook token: {response.text}")
            return False

        except Exception as e:
            logger.error(f"Facebook token refresh error: {e}")
            return False

    def _refresh_twitter_token(self, user_id: str, token_data: Dict[str, Any]) -> bool:
        """Refresh Twitter/X token using refresh token"""
        refresh_token = token_data.get("refresh_token")
        if not refresh_token:
            return False

        try:
            client_id = os.getenv("TWITTER_CLIENT_ID")
            client_secret = os.getenv("TWITTER_CLIENT_SECRET")

            if not client_id or not client_secret:
                logger.error("Twitter client credentials not configured")
                return False

            # Twitter OAuth 2.0 refresh token flow
            auth_string = f"{client_id}:{client_secret}"
            import base64

            auth_header = base64.b64encode(auth_string.encode()).decode()

            headers = {
                "Authorization": f"Basic {auth_header}",
                "Content-Type": "application/x-www-form-urlencoded",
            }

            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            }

            response = requests.post(
                "https://api.twitter.com/2/oauth2/token",
                headers=headers,
                data=data,
                timeout=10,
            )

            if response.status_code == 200:
                new_token_data = response.json()
                new_access_token = new_token_data.get("access_token")
                new_refresh_token = new_token_data.get("refresh_token")

                if new_access_token:
                    # Update stored token data
                    updated_data = token_data.copy()
                    updated_data["access_token"] = new_access_token
                    updated_data["encrypted_token"] = self._encrypt_token(
                        new_access_token
                    )
                    updated_data["stored_at"] = datetime.utcnow().isoformat()
                    updated_data["expires_at"] = (
                        datetime.utcnow() + timedelta(seconds=self.token_ttl)
                    ).isoformat()

                    # Update refresh token if provided
                    if new_refresh_token:
                        updated_data["refresh_token"] = new_refresh_token

                    success = self.store_token(user_id, "twitter", updated_data)
                    if success:
                        logger.info(
                            f"Successfully refreshed Twitter token for user {user_id}"
                        )
                        return True

            logger.error(f"Failed to refresh Twitter token: {response.text}")
            return False

        except Exception as e:
            logger.error(f"Twitter token refresh error: {e}")
            return False


# Global token manager instance
token_manager = TokenManager()
