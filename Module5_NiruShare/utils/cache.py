"""
Simple in-memory cache for formatted posts and images
"""

from typing import Optional, Any
import hashlib
import json
from datetime import datetime, timedelta


class SimpleCache:
    """Simple in-memory cache with TTL"""

    def __init__(self, default_ttl: int = 3600):
        """
        Initialize cache

        Args:
            default_ttl: Default time-to-live in seconds (default: 1 hour)
        """
        self._cache: dict = {}
        self.default_ttl = default_ttl

    def _make_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        # Create a deterministic key from arguments
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items()) if kwargs else {},
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            return None

        entry = self._cache[key]
        expires_at = entry.get("expires_at")

        # Check if expired
        if expires_at and datetime.now() > expires_at:
            del self._cache[key]
            return None

        return entry.get("value")

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if not provided)
        """
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)

        self._cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": datetime.now(),
        }

    def delete(self, key: str) -> None:
        """Delete key from cache"""
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()

    def size(self) -> int:
        """Get number of cache entries"""
        return len(self._cache)
