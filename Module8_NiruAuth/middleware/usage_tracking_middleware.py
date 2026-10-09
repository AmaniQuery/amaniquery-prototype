"""
Usage Tracking Middleware
Logs API usage for cost tracking and analytics
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from typing import Callable
from datetime import datetime
from sqlalchemy.orm import Session
import time

from ..models.auth_models import UsageLog
from ..config import config
from Module3_NiruDB.chat_models import create_database_engine, get_db_session


class UsageTrackingMiddleware(BaseHTTPMiddleware):
    """Tracks API usage for analytics and cost attribution"""

    def __init__(self, app, database_url: str = None):
        super().__init__(app)
        self.database_url = database_url or config.DATABASE_URL
        if self.database_url:
            self.engine = create_database_engine(self.database_url)
        else:
            self.engine = None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Track request usage"""
        if not self.engine:
            return await call_next(request)

        # Skip tracking for certain endpoints
        if request.url.path in ["/health", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Get auth context
        auth_context = getattr(request.state, "auth_context", None)

        # Track request start time
        start_time = time.time()
        # Get request size from Content-Length header (safer than reading body)
        # Reading body consumes it and can cause ClientDisconnect errors
        request_size = 0
        try:
            content_length = request.headers.get("content-length")
            if content_length:
                request_size = int(content_length)
        except (ValueError, TypeError):
            # Invalid content-length header - ignore
            request_size = 0

        # Process request
        response = await call_next(request)

        # Calculate metrics
        response_time_ms = (time.time() - start_time) * 1000
        # Get response size from Content-Length header if available
        # Don't try to read response.body as it may not be available or may cause issues
        response_size = 0
        try:
            if hasattr(response, "headers"):
                content_length = response.headers.get("content-length")
                if content_length:
                    response_size = int(content_length)
        except (ValueError, TypeError):
            response_size = 0

        # Extract tokens used from response (if available)
        tokens_used = 0
        cost = 0.0

        # Try to extract token usage from response headers or body
        if hasattr(response, "headers") and "X-Tokens-Used" in response.headers:
            tokens_used = int(response.headers.get("X-Tokens-Used", 0))

        # Try to get model from response headers
        model = response.headers.get("X-Model-Used", "unknown")
        if tokens_used > 0:
            cost = config.get_llm_cost(model, tokens_used)

        # Log usage asynchronously (don't block response)
        if auth_context:
            try:
                db = get_db_session(self.engine)
                try:
                    usage_log = UsageLog(
                        user_id=auth_context.user_id,
                        integration_id=auth_context.integration_id,
                        api_key_id=auth_context.api_key_id,
                        endpoint=request.url.path,
                        method=request.method,
                        status_code=response.status_code,
                        tokens_used=tokens_used,
                        cost=cost,
                        response_time_ms=response_time_ms,
                        request_size_bytes=request_size,
                        response_size_bytes=response_size,
                        ip_address=request.client.host if request.client else None,
                        user_agent=request.headers.get("user-agent"),
                    )
                    db.add(usage_log)
                    db.commit()
                finally:
                    db.close()
            except Exception as e:
                # Don't fail request if logging fails
                pass

        return response
