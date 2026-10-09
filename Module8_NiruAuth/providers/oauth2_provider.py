"""
OAuth 2.0 Provider
Handles OAuth 2.0 flows (client credentials, authorization code, refresh token)
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..models.auth_models import OAuthClient, OAuthToken, User
from ..config import config
from .jwt_provider import JWTProvider


class OAuth2Provider:
    """Handles OAuth 2.0 operations"""

    @staticmethod
    def hash_secret(secret: str) -> str:
        """Hash client secret for storage"""
        return hashlib.sha256(secret.encode("utf-8")).hexdigest()

    @staticmethod
    def generate_client_credentials() -> Tuple[str, str]:
        """Generate OAuth client ID and secret"""
        client_id = secrets.token_urlsafe(config.OAUTH_CLIENT_ID_LENGTH)
        client_secret = secrets.token_urlsafe(config.OAUTH_CLIENT_SECRET_LENGTH)
        return client_id, client_secret

    @staticmethod
    def create_client(
        db: Session,
        owner_user_id: str,
        name: str,
        description: Optional[str] = None,
        integration_id: Optional[str] = None,
        redirect_uris: Optional[List[str]] = None,
        grant_types: Optional[List[str]] = None,
        scopes: Optional[List[str]] = None,
    ) -> Tuple[str, str, OAuthClient]:
        """Create a new OAuth client"""
        client_id, client_secret = OAuth2Provider.generate_client_credentials()
        client_secret_hash = OAuth2Provider.hash_secret(client_secret)

        if grant_types is None:
            grant_types = ["authorization_code", "refresh_token"]

        if scopes is None:
            scopes = config.OAUTH_DEFAULT_SCOPES

        client = OAuthClient(
            client_id=client_id,
            client_secret_hash=client_secret_hash,
            name=name,
            description=description,
            owner_user_id=owner_user_id,
            integration_id=integration_id,
            redirect_uris=redirect_uris or [],
            grant_types=grant_types,
            scopes=scopes,
        )

        db.add(client)
        db.commit()
        db.refresh(client)

        return client_id, client_secret, client

    @staticmethod
    def validate_client(
        db: Session, client_id: str, client_secret: str
    ) -> Optional[OAuthClient]:
        """Validate OAuth client credentials"""
        client = (
            db.query(OAuthClient)
            .filter(
                and_(OAuthClient.client_id == client_id, OAuthClient.is_active == True)
            )
            .first()
        )

        if not client:
            return None

        # Verify secret
        secret_hash = OAuth2Provider.hash_secret(client_secret)
        if client.client_secret_hash != secret_hash:
            return None

        return client

    @staticmethod
    def generate_authorization_code(
        db: Session,
        client_id: str,
        user_id: str,
        redirect_uri: str,
        scopes: Optional[List[str]] = None,
    ) -> str:
        """Generate authorization code for authorization code flow"""
        # In a real implementation, you'd store this in a database
        # For simplicity, we'll use JWT to encode the authorization code
        code_data = {
            "client_id": client_id,
            "user_id": user_id,
            "redirect_uri": redirect_uri,
            "scopes": scopes or [],
            "exp": datetime.utcnow()
            + timedelta(minutes=config.OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES),
        }

        # Use a simple token for now (in production, use proper JWT)
        code = secrets.token_urlsafe(32)

        # Store in database or cache (simplified - in production use Redis)
        # For now, we'll encode it in the token itself via JWT
        return JWTProvider.encode_token(
            code_data, expires_minutes=config.OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES
        )

    @staticmethod
    def exchange_authorization_code(
        db: Session, code: str, client_id: str, client_secret: str, redirect_uri: str
    ) -> Optional[Dict]:
        """Exchange authorization code for access token"""
        # Validate client
        client = OAuth2Provider.validate_client(db, client_id, client_secret)
        if not client:
            return None

        # Decode authorization code
        try:
            code_data = JWTProvider.decode_token(code)
        except:
            return None

        # Verify code belongs to this client
        if code_data.get("client_id") != client_id:
            return None

        # Verify redirect URI
        if redirect_uri not in (client.redirect_uris or []):
            return None

        # Generate access token
        user_id = code_data.get("user_id")
        scopes = code_data.get("scopes", [])

        return OAuth2Provider.generate_access_token(db, client, user_id, scopes)

    @staticmethod
    def generate_access_token(
        db: Session,
        client: OAuthClient,
        user_id: Optional[str] = None,
        scopes: Optional[List[str]] = None,
    ) -> Dict:
        """Generate OAuth access token"""
        if scopes is None:
            scopes = client.scopes or []

        # Generate tokens
        access_token = JWTProvider.encode_token(
            {
                "client_id": client.id,
                "user_id": user_id,
                "scopes": scopes,
                "type": "access_token",
            },
            expires_minutes=config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
        )

        refresh_token = JWTProvider.encode_token(
            {
                "client_id": client.id,
                "user_id": user_id,
                "scopes": scopes,
                "type": "refresh_token",
            },
            expires_minutes=config.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60,
        )

        expires_at = datetime.utcnow() + timedelta(
            minutes=config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        )

        # Store token in database
        oauth_token = OAuthToken(
            access_token=access_token,
            refresh_token=refresh_token,
            client_id=client.id,
            user_id=user_id,
            scopes=scopes,
            expires_at=expires_at,
        )

        db.add(oauth_token)
        db.commit()

        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "refresh_token": refresh_token,
            "scope": " ".join(scopes) if scopes else None,
        }

    @staticmethod
    def validate_access_token(db: Session, access_token: str) -> Optional[OAuthToken]:
        """Validate OAuth access token"""
        # First check database
        token = (
            db.query(OAuthToken)
            .filter(
                and_(
                    OAuthToken.access_token == access_token,
                    OAuthToken.revoked == False,
                    OAuthToken.expires_at > datetime.utcnow(),
                )
            )
            .first()
        )

        if token:
            return token

        # Also validate as JWT (for stateless tokens)
        try:
            payload = JWTProvider.decode_token(access_token)
            if payload.get("type") == "access_token":
                # Create a virtual token object
                return type(
                    "OAuthToken",
                    (),
                    {
                        "client_id": payload.get("client_id"),
                        "user_id": payload.get("user_id"),
                        "scopes": payload.get("scopes", []),
                        "expires_at": datetime.fromtimestamp(payload.get("exp", 0)),
                    },
                )()
        except:
            pass

        return None

    @staticmethod
    def refresh_access_token(
        db: Session, refresh_token: str, client_id: str, client_secret: str
    ) -> Optional[Dict]:
        """Refresh access token using refresh token"""
        # Validate client
        client = OAuth2Provider.validate_client(db, client_id, client_secret)
        if not client:
            return None

        # Check database for refresh token
        token = (
            db.query(OAuthToken)
            .filter(
                and_(
                    OAuthToken.refresh_token == refresh_token,
                    OAuthToken.client_id == client.id,
                    OAuthToken.revoked == False,
                    OAuthToken.expires_at > datetime.utcnow(),
                )
            )
            .first()
        )

        if not token:
            # Try JWT validation
            try:
                payload = JWTProvider.decode_token(refresh_token)
                if (
                    payload.get("type") != "refresh_token"
                    or payload.get("client_id") != client.id
                ):
                    return None
                user_id = payload.get("user_id")
                scopes = payload.get("scopes", [])
            except:
                return None
        else:
            user_id = token.user_id
            scopes = token.scopes or []

        # Revoke old token
        if token:
            token.revoked = True
            token.revoked_at = datetime.utcnow()
            db.commit()

        # Generate new token
        return OAuth2Provider.generate_access_token(db, client, user_id, scopes)

    @staticmethod
    def revoke_token(db: Session, token: str) -> bool:
        """Revoke an OAuth token"""
        # Check access token
        oauth_token = (
            db.query(OAuthToken)
            .filter(and_(OAuthToken.access_token == token, OAuthToken.revoked == False))
            .first()
        )

        if not oauth_token:
            # Check refresh token
            oauth_token = (
                db.query(OAuthToken)
                .filter(
                    and_(OAuthToken.refresh_token == token, OAuthToken.revoked == False)
                )
                .first()
            )

        if oauth_token:
            oauth_token.revoked = True
            oauth_token.revoked_at = datetime.utcnow()
            db.commit()
            return True

        return False

    @staticmethod
    def get_client_by_id(db: Session, client_id: str) -> Optional[OAuthClient]:
        """Get OAuth client by client_id"""
        return db.query(OAuthClient).filter(OAuthClient.client_id == client_id).first()
