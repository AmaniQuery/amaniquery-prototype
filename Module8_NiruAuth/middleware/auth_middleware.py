"""
Authentication Middleware
Validates credentials and attaches authentication context to requests
"""

import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from typing import Optional, Callable
from sqlalchemy.orm import Session
from fastapi import status, HTTPException

logger = logging.getLogger(__name__)

from ..models.auth_models import User, Integration
from ..models.pydantic_models import AuthContext
from ..providers.api_key_provider import APIKeyProvider
from ..providers.oauth2_provider import OAuth2Provider
from ..providers.jwt_provider import JWTProvider
from ..providers.session_provider import SessionProvider
from ..config import config
from Module3_NiruDB.chat_models import create_database_engine, get_db_session


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for FastAPI"""

    # Public endpoints that don't require authentication
    PUBLIC_ENDPOINTS = [
        "/",
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/api/v1/auth/register",
        "/api/v1/auth/login",
        "/api/v1/auth/password/reset-request",
        "/api/v1/auth/password/reset",
        "/api/v1/auth/email/verify",
        "/api/v1/auth/email/verify-request",
    ]

    def __init__(self, app, database_url: Optional[str] = None):
        super().__init__(app)
        self.database_url = database_url or config.DATABASE_URL
        if self.database_url:
            self.engine = create_database_engine(self.database_url)
        else:
            self.engine = None

    def is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public"""
        # Check exact match
        if path in self.PUBLIC_ENDPOINTS:
            return True

        # Check prefix match (only for specific auth endpoints, not all /api/v1/auth/*)
        # Only allow exact matches or specific prefixes, not broad /api/v1/auth/ prefix
        public_prefixes = [
            "/api/v1/auth/register",
            "/api/v1/auth/login",
            "/api/v1/auth/password/reset-request",
            "/api/v1/auth/password/reset",
            "/api/v1/auth/email/verify",
            "/api/v1/auth/email/verify-request",
        ]
        for public_path in public_prefixes:
            if path == public_path or path.startswith(public_path + "/"):
                return True

        # Admin routes are explicitly NOT public, they are handled by AdminAuthMiddleware
        if path.startswith("/admin"):
            return False

        return False

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and attach auth context"""
        # Skip authentication for public endpoints
        if self.is_public_endpoint(request.url.path):
            return await call_next(request)

        # Skip if no database connection
        if not self.engine:
            logger.warning(
                "AuthMiddleware: No database engine available, skipping authentication"
            )
            return await call_next(request)

        auth_context = None

        try:
            # Use context manager for database session
            with get_db_session(self.engine) as db:
                # Ensure we have a fresh database session
                db.expire_all()

                # Try to authenticate
                auth_context = await self.authenticate_request(request, db)

                # Attach auth context to request state inside the session block
                # Note: The user object in auth_context will be detached when this block exits
                request.state.auth_context = auth_context

                if auth_context:
                    logger.debug(
                        f"Auth context attached for user {auth_context.user_id}"
                    )
                else:
                    # Log why it's None if we have a session token
                    session_token = request.cookies.get(
                        "session_token"
                    ) or request.headers.get("X-Session-Token")
                    if session_token:
                        logger.warning(
                            f"Auth context is None despite session token presence. Token: {session_token[:10]}..."
                        )

            # Continue with request (outside db context to avoid holding connection)
            response = await call_next(request)

            return response

        except HTTPException as e:
            return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
        except Exception as e:
            import traceback

            logger.error(f"AuthMiddleware error: {str(e)}\n{traceback.format_exc()}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Authentication error",
                    "detail": str(e),
                    "traceback": traceback.format_exc(),
                },
            )

    async def authenticate_request(
        self, request: Request, db: Session
    ) -> Optional[AuthContext]:
        """Authenticate request and return auth context"""
        # Try different authentication methods in order

        # 1. Try API Key (X-API-Key header)
        api_key = request.headers.get("X-API-Key")
        if api_key:
            api_key_record = APIKeyProvider.validate_api_key(db, api_key)
            if api_key_record:
                user = None
                integration = None

                if api_key_record.user_id:
                    user = (
                        db.query(User).filter(User.id == api_key_record.user_id).first()
                    )
                elif api_key_record.integration_id:
                    integration = (
                        db.query(Integration)
                        .filter(Integration.id == api_key_record.integration_id)
                        .first()
                    )

                if user or integration:
                    return AuthContext(
                        auth_method="api_key",
                        user_id=user.id if user else None,
                        integration_id=integration.id if integration else None,
                        api_key_id=api_key_record.id,
                        scopes=api_key_record.scopes or [],
                        user=user,  # Cache user object
                        integration=integration,  # Cache integration object
                    )

        # 2. Try Bearer token (JWT or OAuth)
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            token = authorization[7:]  # Remove "Bearer " prefix

            # Try OAuth token first
            oauth_token = OAuth2Provider.validate_access_token(db, token)
            if oauth_token:
                user = None
                if oauth_token.user_id:
                    user = db.query(User).filter(User.id == oauth_token.user_id).first()

                return AuthContext(
                    auth_method="oauth2",
                    user_id=user.id if user else None,
                    integration_id=None,
                    scopes=oauth_token.scopes or [],
                    user=user,  # Cache user object
                )

            # Try JWT token
            jwt_payload = JWTProvider.validate_token(token)
            if jwt_payload:
                user = None
                integration = None

                if jwt_payload.get("user_id"):
                    user = (
                        db.query(User).filter(User.id == jwt_payload["user_id"]).first()
                    )
                if jwt_payload.get("integration_id"):
                    integration = (
                        db.query(Integration)
                        .filter(Integration.id == jwt_payload["integration_id"])
                        .first()
                    )

                if user or integration:
                    return AuthContext(
                        auth_method="jwt",
                        user_id=user.id if user else None,
                        integration_id=integration.id if integration else None,
                        scopes=jwt_payload.get("scopes", []),
                        user=user,  # Cache user object
                        integration=integration,  # Cache integration object
                    )

        # 3. Try session token (Cookie or X-Session-Token header)
        session_token = request.cookies.get("session_token") or request.headers.get(
            "X-Session-Token"
        )
        if session_token:
            try:
                session = SessionProvider.validate_session(db, session_token)
                if session:
                    # Get user for this session
                    user = db.query(User).filter(User.id == session.user_id).first()
                    if user and user.status == "active":
                        return AuthContext(
                            auth_method="session",
                            user_id=user.id,
                            integration_id=None,
                            user=user,
                        )
            except Exception as e:
                logger.error(f"Session authentication error: {e}")

        if session_token and not session:
            logger.warning(
                f"AuthMiddleware: Session token present but validation failed (returned None). Token: {session_token[:10]}..."
            )

        # No authentication found - return None (endpoint may be optional auth)
        return None
