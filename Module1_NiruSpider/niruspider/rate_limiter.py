"""
Rate Limiter for Per-Domain Request Throttling
Uses Redis for distributed rate limiting.
Non-blocking — does NOT use time.sleep(). Returns allow/deny decisions instead.
"""

import os
import time
from typing import Optional
from urllib.parse import urlparse
from loguru import logger

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, rate limiting will be disabled")


class RateLimiter:
    """
    Per-domain rate limiter using Redis.
    Sliding-window counter — non-blocking. Returns allow/deny.
    """

    def __init__(self, redis_url: Optional[str] = None, default_rate: float = 2.0):
        self.redis_client = None
        self.default_rate = default_rate
        self.domain_rates = {}

        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, rate limiting disabled")
            return

        if redis_url is None:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        try:
            if redis_url.startswith("redis://") or redis_url.startswith("rediss://"):
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
            else:
                self.redis_client = redis.Redis(
                    host="localhost", port=6379, db=0, decode_responses=True
                )
            self.redis_client.ping()
            logger.info("Rate limiter initialized with Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.warning("Rate limiting disabled, falling back to in-memory limiter")
            self.redis_client = None
            self._in_memory_limits = {}
            self._in_memory_tokens = {}

    def set_domain_rate(self, domain: str, rate: float):
        self.domain_rates[domain] = rate
        logger.debug(f"Set rate for {domain}: {rate} req/s")

    def allow_request(self, url: str) -> bool:
        """
        Non-blocking rate check.

        Args:
            url: Request URL

        Returns:
            True if request is allowed, False if rate-limited
        """
        domain = self._extract_domain(url)
        if not domain:
            return True

        rate = self.domain_rates.get(domain, self.default_rate)
        min_interval = 1.0 / rate

        if self.redis_client:
            return self._check_redis(domain, min_interval)
        else:
            return self._check_in_memory(domain, min_interval)

    def _check_redis(self, domain: str, min_interval: float) -> bool:
        try:
            key = f"rate_limit:{domain}"
            now = time.time()

            last_time_str = self.redis_client.get(key)

            if last_time_str:
                last_time = float(last_time_str)
                elapsed = now - last_time
                if elapsed < min_interval:
                    logger.debug(
                        f"Rate limiting {domain}: blocked (elapsed={elapsed:.2f}s < interval={min_interval:.2f}s)"
                    )
                    return False

            self.redis_client.set(key, str(time.time()), ex=3600)
            return True

        except Exception as e:
            logger.error(f"Redis rate limit check error: {e}")
            return self._check_in_memory(domain, min_interval)

    def _check_in_memory(self, domain: str, min_interval: float) -> bool:
        if not hasattr(self, "_in_memory_limits"):
            self._in_memory_limits = {}

        now = time.time()
        last_time = self._in_memory_limits.get(domain, 0)
        elapsed = now - last_time

        if elapsed < min_interval:
            logger.debug(f"Rate limiting {domain}: blocked (in-memory)")
            return False

        self._in_memory_limits[domain] = time.time()
        return True

    def _extract_domain(self, url: str) -> Optional[str]:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if ":" in domain:
                domain = domain.split(":")[0]
            return domain
        except Exception:
            return None

    def get_stats(self) -> dict:
        if not self.redis_client:
            return {"status": "disabled", "reason": "Redis not available"}

        try:
            keys = self.redis_client.keys("rate_limit:*")
            return {
                "status": "active",
                "domains_tracked": len(keys),
                "default_rate": self.default_rate,
                "custom_rates": self.domain_rates,
            }
        except Exception as e:
            logger.error(f"Error getting rate limit stats: {e}")
            return {"status": "error", "error": str(e)}
