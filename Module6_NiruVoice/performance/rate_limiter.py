"""
Rate limiting for voice agent requests
"""

import time
from typing import Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
from loguru import logger
from threading import Lock

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""

    requests_per_minute: int = 60  # Max requests per minute
    requests_per_hour: int = 1000  # Max requests per hour
    burst_size: int = 10  # Allow burst of N requests

    # Use Redis for distributed rate limiting
    enable_redis: bool = False
    redis_url: Optional[str] = None

    def __post_init__(self):
        """Validate configuration"""
        if self.requests_per_minute < 1:
            raise ValueError("requests_per_minute must be at least 1")
        if self.requests_per_hour < self.requests_per_minute:
            raise ValueError("requests_per_hour must be >= requests_per_minute")
        if self.burst_size < 1:
            raise ValueError("burst_size must be at least 1")


class RateLimiter:
    """
    Rate limiter for voice agent requests
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        self._redis_client = None

        # Initialize Redis if enabled
        if self.config.enable_redis and REDIS_AVAILABLE:
            try:
                if self.config.redis_url:
                    self._redis_client = redis.from_url(self.config.redis_url)
                else:
                    self._redis_client = redis.Redis()

                self._redis_client.ping()
                logger.info("Redis rate limiter initialized")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize Redis rate limiter: {e}, using memory"
                )
                self._redis_client = None

        # In-memory tracking (fallback or when Redis not available)
        self._request_times: Dict[str, list] = defaultdict(list)
        self._lock = Lock()

        self._stats = {
            "allowed": 0,
            "blocked": 0,
        }

        logger.info(
            f"Rate limiter initialized: "
            f"{self.config.requests_per_minute} req/min, "
            f"{self.config.requests_per_hour} req/hour, "
            f"burst={self.config.burst_size}"
        )

    def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed

        Args:
            identifier: Unique identifier (e.g., session_id, user_id)

        Returns:
            True if allowed, False if rate limited
        """
        current_time = time.time()

        # Use Redis if available
        if self._redis_client:
            return self._is_allowed_redis(identifier, current_time)

        # Use memory-based rate limiting
        return self._is_allowed_memory(identifier, current_time)

    def _is_allowed_redis(self, identifier: str, current_time: float) -> bool:
        """Check rate limit using Redis"""
        try:
            # Sliding window rate limiting using sorted sets
            pipe = self._redis_client.pipeline()

            # Minute window
            minute_key = f"ratelimit:minute:{identifier}"
            minute_window_start = current_time - 60

            # Hour window
            hour_key = f"ratelimit:hour:{identifier}"
            hour_window_start = current_time - 3600

            # Clean old entries and count
            pipe.zremrangebyscore(minute_key, 0, minute_window_start)
            pipe.zcard(minute_key)
            pipe.zremrangebyscore(hour_key, 0, hour_window_start)
            pipe.zcard(hour_key)

            results = pipe.execute()
            minute_count = results[1]
            hour_count = results[3]

            # Check limits
            if minute_count >= self.config.requests_per_minute:
                self._stats["blocked"] += 1
                return False

            if hour_count >= self.config.requests_per_hour:
                self._stats["blocked"] += 1
                return False

            # Add current request
            pipe.zadd(minute_key, {str(current_time): current_time})
            pipe.zadd(hour_key, {str(current_time): current_time})
            pipe.expire(minute_key, 60)
            pipe.expire(hour_key, 3600)
            pipe.execute()

            self._stats["allowed"] += 1
            return True

        except Exception as e:
            logger.warning(f"Redis rate limit check error: {e}, falling back to memory")
            return self._is_allowed_memory(identifier, current_time)

    def _is_allowed_memory(self, identifier: str, current_time: float) -> bool:
        """Check rate limit using memory"""
        with self._lock:
            request_times = self._request_times[identifier]

            # Clean old entries (older than 1 hour)
            request_times[:] = [t for t in request_times if current_time - t < 3600]

            # Check hour limit
            if len(request_times) >= self.config.requests_per_hour:
                self._stats["blocked"] += 1
                return False

            # Check minute limit (last 60 seconds)
            recent_requests = [t for t in request_times if current_time - t < 60]
            if len(recent_requests) >= self.config.requests_per_minute:
                self._stats["blocked"] += 1
                return False

            # Allow request
            request_times.append(current_time)
            self._stats["allowed"] += 1
            return True

    def reset(self, identifier: Optional[str] = None):
        """
        Reset rate limit for identifier(s)

        Args:
            identifier: Specific identifier to reset (None = all)
        """
        if identifier:
            with self._lock:
                if identifier in self._request_times:
                    del self._request_times[identifier]

            if self._redis_client:
                try:
                    self._redis_client.delete(
                        f"ratelimit:minute:{identifier}",
                        f"ratelimit:hour:{identifier}",
                    )
                except Exception as e:
                    logger.warning(f"Error resetting Redis rate limit: {e}")
        else:
            with self._lock:
                self._request_times.clear()

            logger.info(f"Rate limit reset for {identifier or 'all identifiers'}")

    def get_stats(self) -> Dict:
        """Get rate limiter statistics"""
        total = self._stats["allowed"] + self._stats["blocked"]
        block_rate = self._stats["blocked"] / total if total > 0 else 0.0

        return {
            "allowed": self._stats["allowed"],
            "blocked": self._stats["blocked"],
            "block_rate": block_rate,
            "tracked_identifiers": len(self._request_times),
        }

    def reset_stats(self):
        """Reset statistics"""
        self._stats = {
            "allowed": 0,
            "blocked": 0,
        }
