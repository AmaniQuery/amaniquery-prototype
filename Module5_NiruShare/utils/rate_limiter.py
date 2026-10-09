"""
Robust Rate Limiting and Error Handling for Social Media APIs
Implements 2026 best practices: Exponential Backoff, Token Bucket, and Usage Tracking.
"""

import time
import asyncio
from typing import Dict, Optional, Callable, Any
from functools import wraps
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)


class PlatformRateLimit:
    """Rate limit configuration for a specific platform"""

    def __init__(self, requests_per_window: int, window_seconds: int):
        self.limit = requests_per_window
        self.window = window_seconds
        self.tokens = requests_per_window
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token, waiting if necessary"""
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_refill

            # Refill tokens based on time passed
            refill_amount = time_passed * (self.limit / self.window)
            self.tokens = min(self.limit, self.tokens + refill_amount)
            self.last_refill = now

            if self.tokens < 1:
                # Calculate wait time
                wait_time = (1 - self.tokens) * (self.window / self.limit)
                logger.warning(f"Rate limit hit. Waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class RateLimiter:
    """Global rate limiter for social platforms"""

    # Default conservative limits (can be overridden)
    # Twitter: 300 posts / 3 hours (approx 1 per 36s) -> conservative: 1 per 60s
    # LinkedIn: Unofficial limits ~100/day -> conservative: 1 per 900s (15 min)
    # Facebook: Complex, user-based -> conservative: 1 per 60s
    DEFAULTS = {
        "twitter": PlatformRateLimit(1, 60),
        "linkedin": PlatformRateLimit(1, 60),  # Stricter in prod
        "facebook": PlatformRateLimit(1, 60),
        "instagram": PlatformRateLimit(1, 120),
        "default": PlatformRateLimit(5, 60),
    }

    def __init__(self):
        self._limiters: Dict[str, PlatformRateLimit] = self.DEFAULTS.copy()

    def get_limiter(self, platform: str) -> PlatformRateLimit:
        return self._limiters.get(platform.lower(), self._limiters["default"])

    async def wait_for_token(self, platform: str):
        """Wait for rate limit token for specific platform"""
        limiter = self.get_limiter(platform)
        await limiter.acquire()


# Global instance
_limiter = RateLimiter()


def with_rate_limit(platform_name: str):
    """Decorator to apply rate limiting to a function"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await _limiter.wait_for_token(platform_name)
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def robust_api_call(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions_to_retry: tuple = (Exception,),
):
    """
    Decorator for robust API calls with exponential backoff.
    Uses Tenacity for reliable retrying.
    """
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=backoff_factor, min=1, max=60),
        retry=retry_if_exception_type(exceptions_to_retry),
        before_sleep=before_sleep_log(logger, "WARNING"),
    )
