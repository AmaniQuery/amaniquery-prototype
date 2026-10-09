"""
Rate Limiting Middleware
Uses Redis-backed RateLimiter with SQLAlchemy fallback for rate limiting.
"""

import os
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from typing import Callable
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from fastapi import status

from ..models.auth_models import RateLimit
from ..config import config
from ..providers.rate_limiter import RateLimiter
from Module3_NiruDB.chat_models import create_database_engine, get_db_session

try:
    import redis.asyncio as aioredis

    _redis_client = aioredis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
except Exception:
    _redis_client = None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with Redis fast path + DB fallback"""

    def __init__(self, app, database_url: str = None):
        super().__init__(app)
        self.database_url = database_url or config.DATABASE_URL
        if self.database_url:
            self.engine = create_database_engine(self.database_url)
        else:
            self.engine = None
        self._limiter = RateLimiter(redis_client=_redis_client)

    async def dispatch(self, request: Request, call_next: Callable):
        """Process request with rate limiting"""
        if request.url.path in ["/health", "/docs", "/openapi.json"]:
            return await call_next(request)

        if not self.engine:
            return await call_next(request)

        auth_context = getattr(request.state, "auth_context", None)
        if not auth_context:
            return await call_next(request)

        identity = (
            auth_context.user_id
            or auth_context.integration_id
            or auth_context.api_key_id
            or "anonymous"
        )
        tier = getattr(auth_context, "tier", "basic")
        limits = config.get_rate_limit(tier, request.url.path)

        allowed, remaining = self._limiter.check(
            key=f"rl:{identity}",
            limit_per_minute=limits["per_minute"],
            limit_per_hour=limits["per_hour"],
            limit_per_day=limits["per_day"],
        )

        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": "Too many requests. Please try again later.",
                    "retry_after_minute": remaining.get("remaining_per_minute", 0),
                },
                headers={
                    "X-RateLimit-Limit": str(limits["per_minute"]),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": "60",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limits["per_minute"])
        response.headers["X-RateLimit-Remaining"] = str(
            remaining.get("remaining_per_minute", limits["per_minute"])
        )
        return response
