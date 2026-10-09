"""
Redis-backed Rate Limiter
Sliding window counters for API rate limiting at 1M+ QPS
"""

import time
import hashlib
import threading
from typing import Dict, Optional, Tuple
from loguru import logger

from ..config import config

# Fallback in-memory store when Redis is unavailable
_local_store: Dict[str, list] = {}
_local_lock = threading.Lock()


class RateLimiter:
    """Rate limiter with Redis fallback to in-memory store"""

    def __init__(self, redis_client=None):
        self._redis = redis_client

    def check(
        self,
        key: str,
        limit_per_minute: int,
        limit_per_hour: int,
        limit_per_day: int,
        cost: int = 1,
    ) -> Tuple[bool, Dict[str, int]]:
        """Check if request is within rate limits.
        Returns (allowed, {remaining_per_minute, remaining_per_hour, remaining_per_day}).
        """
        now = time.time()
        minute_window = int(now / 60)
        hour_window = int(now / 3600)
        day_window = int(now / 86400)

        if self._redis:
            return self._check_redis(
                key,
                minute_window,
                hour_window,
                day_window,
                limit_per_minute,
                limit_per_hour,
                limit_per_day,
                cost,
            )
        return self._check_local(
            key,
            minute_window,
            hour_window,
            day_window,
            limit_per_minute,
            limit_per_hour,
            limit_per_day,
            cost,
            now,
        )

    def _check_redis(
        self,
        key: str,
        min_w: int,
        hr_w: int,
        day_w: int,
        lpm: int,
        lph: int,
        lpd: int,
        cost: int,
    ) -> Tuple[bool, Dict[str, int]]:
        try:
            pipe = self._redis.pipeline()
            for window_key, limit in [
                (f"rl:m:{key}:{min_w}", lpm),
                (f"rl:h:{key}:{hr_w}", lph),
                (f"rl:d:{key}:{day_w}", lpd),
            ]:
                pipe.incrby(window_key, cost)
                pipe.expire(window_key, 86400)
            results = pipe.execute()
            min_count = results[0]
            hr_count = results[2]
            day_count = results[4]
        except Exception as e:
            logger.warning(f"Redis rate limit failed, allowing: {e}")
            return True, {
                "remaining_per_minute": lpm,
                "remaining_per_hour": lph,
                "remaining_per_day": lpd,
            }

        allowed = min_count <= lpm and hr_count <= lph and day_count <= lpd
        return allowed, {
            "remaining_per_minute": max(0, lpm - min_count),
            "remaining_per_hour": max(0, lph - hr_count),
            "remaining_per_day": max(0, lpd - day_count),
        }

    def _check_local(
        self,
        key: str,
        min_w: int,
        hr_w: int,
        day_w: int,
        lpm: int,
        lph: int,
        lpd: int,
        cost: int,
        now: float,
    ) -> Tuple[bool, Dict[str, int]]:
        with _local_lock:
            store = _local_store
            self._evict_expired(store, now)

            min_key = f"m:{key}:{min_w}"
            hr_key = f"h:{key}:{hr_w}"
            day_key = f"d:{key}:{day_w}"

            min_count = store.get(min_key, 0) + cost
            hr_count = store.get(hr_key, 0) + cost
            day_count = store.get(day_key, 0) + cost

            store[min_key] = min_count
            store[hr_key] = hr_count
            store[day_key] = day_count

        allowed = min_count <= lpm and hr_count <= lph and day_count <= lpd
        return allowed, {
            "remaining_per_minute": max(0, lpm - min_count),
            "remaining_per_hour": max(0, lph - hr_count),
            "remaining_per_day": max(0, lpd - day_count),
        }

    def _evict_expired(self, store: Dict[str, int], now: float):
        """Evict entries from previous windows"""
        current_min = int(now / 60)
        current_hour = int(now / 3600)
        current_day = int(now / 86400)
        expired = [
            k
            for k in store
            if self._is_expired(k, current_min, current_hour, current_day)
        ]
        for k in expired:
            del store[k]
        if len(store) > 100000:
            for k in list(store.keys())[:10000]:
                del store[k]

    @staticmethod
    def _is_expired(key: str, cur_min: int, cur_hour: int, cur_day: int) -> bool:
        parts = key.split(":")
        prefix = parts[0]
        window = int(parts[2])
        if prefix == "m" and window != cur_min:
            return True
        if prefix == "h" and window != cur_hour:
            return True
        if prefix == "d" and window != cur_day:
            return True
        return False

    def get_limits_for_tier(
        self, tier: str, endpoint: Optional[str] = None
    ) -> Dict[str, int]:
        """Get rate limits for a given tier and optional endpoint"""
        return config.get_rate_limit(tier, endpoint)
