"""
Response caching for voice agent queries
"""

import hashlib
import json
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
from threading import Lock

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.debug("Redis not available, using in-memory cache")


@dataclass
class CacheConfig:
    """Configuration for response cache"""

    ttl: float = 3600.0  # Time to live in seconds (1 hour default)
    max_size: int = 1000  # Maximum number of cached items
    enable_redis: bool = False  # Use Redis for distributed caching
    redis_url: Optional[str] = None  # Redis connection URL
    key_prefix: str = "voice_cache:"  # Prefix for cache keys

    def __post_init__(self):
        """Validate configuration"""
        if self.ttl <= 0:
            raise ValueError("TTL must be positive")
        if self.max_size < 1:
            raise ValueError("max_size must be at least 1")


class CacheEntry:
    """A single cache entry"""

    def __init__(self, key: str, value: Any, ttl: float):
        """
        Initialize cache entry

        Args:
            key: Cache key
            value: Cached value
            ttl: Time to live in seconds
        """
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl
        self.access_count = 0
        self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """Check if entry is expired"""
        return time.time() > self.expires_at

    def access(self):
        """Record access"""
        self.access_count += 1
        self.last_accessed = time.time()

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "value": self.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


class ResponseCache:
    """
    Caches responses to reduce redundant processing
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize response cache

        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._lock = Lock()
        self._redis_client = None

        # Initialize Redis if enabled
        if self.config.enable_redis and REDIS_AVAILABLE:
            try:
                if self.config.redis_url:
                    self._redis_client = redis.from_url(self.config.redis_url)
                else:
                    self._redis_client = redis.Redis()

                # Test connection
                self._redis_client.ping()
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize Redis cache: {e}, falling back to memory cache"
                )
                self._redis_client = None
        elif self.config.enable_redis and not REDIS_AVAILABLE:
            logger.warning("Redis requested but not available, using memory cache")

        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
        }

        logger.info(
            f"Response cache initialized: "
            f"backend={'redis' if self._redis_client else 'memory'}, "
            f"ttl={self.config.ttl}s, max_size={self.config.max_size}"
        )

    def _generate_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from arguments

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        # Create a deterministic key from arguments
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items()),
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()
        return f"{self.config.key_prefix}{key_hash}"

    def get(self, *args, **kwargs) -> Optional[Any]:
        """
        Get cached value

        Args:
            *args: Positional arguments used to generate key
            **kwargs: Keyword arguments used to generate key

        Returns:
            Cached value or None if not found/expired
        """
        key = self._generate_key(*args, **kwargs)

        # Try Redis first if available
        if self._redis_client:
            try:
                cached = self._redis_client.get(key)
                if cached:
                    entry_data = json.loads(cached)
                    # Check expiration
                    if time.time() < entry_data["expires_at"]:
                        self._stats["hits"] += 1
                        return entry_data["value"]
                    else:
                        # Expired, delete it
                        self._redis_client.delete(key)
                        self._stats["misses"] += 1
                        return None
                else:
                    self._stats["misses"] += 1
                    return None
            except Exception as e:
                logger.warning(f"Redis cache get error: {e}, falling back to memory")

        # Fall back to memory cache
        with self._lock:
            if key in self._memory_cache:
                entry = self._memory_cache[key]

                if entry.is_expired():
                    # Expired, remove it
                    del self._memory_cache[key]
                    self._stats["misses"] += 1
                    return None

                entry.access()
                self._stats["hits"] += 1
                return entry.value
            else:
                self._stats["misses"] += 1
                return None

    def set(self, value: Any, *args, **kwargs):
        """
        Set cached value

        Args:
            value: Value to cache
            *args: Positional arguments used to generate key
            **kwargs: Keyword arguments used to generate key
        """
        key = self._generate_key(*args, **kwargs)

        # Try Redis first if available
        if self._redis_client:
            try:
                entry = CacheEntry(key, value, self.config.ttl)
                entry_data = entry.to_dict()
                self._redis_client.setex(
                    key, int(self.config.ttl), json.dumps(entry_data, default=str)
                )
                self._stats["sets"] += 1
                return
            except Exception as e:
                logger.warning(f"Redis cache set error: {e}, falling back to memory")

        # Fall back to memory cache
        with self._lock:
            # Check if we need to evict
            if len(self._memory_cache) >= self.config.max_size:
                self._evict_oldest()

            entry = CacheEntry(key, value, self.config.ttl)
            self._memory_cache[key] = entry
            self._stats["sets"] += 1

    def _evict_oldest(self):
        """Evict oldest/least recently used entry"""
        if not self._memory_cache:
            return

        # Find least recently accessed entry
        oldest_key = min(
            self._memory_cache.keys(), key=lambda k: self._memory_cache[k].last_accessed
        )

        del self._memory_cache[oldest_key]
        self._stats["evictions"] += 1

    def clear(self):
        """Clear all cached entries"""
        with self._lock:
            self._memory_cache.clear()

        if self._redis_client:
            try:
                # Delete all keys with prefix
                keys = self._redis_client.keys(f"{self.config.key_prefix}*")
                if keys:
                    self._redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"Error clearing Redis cache: {e}")

        logger.info("Cache cleared")

    def cleanup_expired(self):
        """Remove expired entries from memory cache"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._memory_cache.items() if entry.is_expired()
            ]
            for key in expired_keys:
                del self._memory_cache[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0

        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "sets": self._stats["sets"],
            "evictions": self._stats["evictions"],
            "size": len(self._memory_cache),
            "max_size": self.config.max_size,
            "backend": "redis" if self._redis_client else "memory",
        }

    def reset_stats(self):
        """Reset cache statistics"""
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
        }
