"""
Redis Cache with Multi-Level Caching, Semantic Similarity, and AI-Powered TTL
Advanced caching strategies for real-time RAG performance
"""

import os
import json
import hashlib
import re
import time
import asyncio
from typing import Optional, Any, Callable, Dict, Union, List, Tuple
from functools import wraps, lru_cache
from collections import OrderedDict
from loguru import logger
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle

# Try importing standard redis first (preferred for local), then upstash
try:
    import redis

    STANDARD_REDIS_AVAILABLE = True
except ImportError:
    STANDARD_REDIS_AVAILABLE = False

try:
    from upstash_redis import Redis as UpstashRedis

    UPSTASH_AVAILABLE = True
except ImportError:
    UPSTASH_AVAILABLE = False


class CacheLevel(Enum):
    """Multi-level cache hierarchy"""

    L1_MEMORY = "l1_memory"  # In-memory (nanoseconds)
    L2_REDIS = "l2_redis"  # Redis (microseconds)
    L3_SEMANTIC = "l3_semantic"  # Semantic similarity (milliseconds)
    L4_VECTOR = "l4_vector"  # Vector similarity (tens of milliseconds)


@dataclass
class CacheConfig:
    """Advanced cache configuration"""

    l1_capacity: int = 1000
    l2_capacity: int = 10000
    semantic_threshold: float = 0.85
    vector_threshold: float = 0.80
    default_ttl: int = 3600 * 12  # 12 hours
    enable_compression: bool = True
    enable_predictive_cache: bool = True
    predictive_threshold: float = 0.8


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    data: Any
    timestamp: float
    ttl: int
    access_count: int = 1
    last_accessed: float = field(default_factory=time.time)
    semantic_embedding: Optional[np.ndarray] = None
    vector_embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)


# AI-Powered TTL Strategy based on query patterns and content type
AI_TTL_MAP = {
    "legal": 3600 * 24 * 90,  # 90 days - laws rarely change
    "news": 3600 * 2,  # 2 hours - news moves fast
    "trending": 300,  # 5 minutes - trending topics
    "parliamentary": 3600 * 24,  # 24 hours - parliamentary sessions
    "historical": 3600 * 24 * 365,  # 1 year - historical facts
    "calculator": 3600 * 24 * 365,  # 1 year - calculator results
    "default": 3600 * 6,  # 6 hours - optimized default
}


class BlazingFastCache:
    """Multi-level caching system with AI-powered optimization"""

    # Module-level Redis client (shared across instances for cross-pod consistency)
    _redis_cache: Optional["RedisCache"] = None
    _redis_lock = threading.Lock()

    @classmethod
    def _get_redis_backend(cls) -> Optional["RedisCache"]:
        """Get or create shared Redis backend for L2 cache"""
        if cls._redis_cache is None:
            with cls._redis_lock:
                if cls._redis_cache is None:
                    try:
                        cls._redis_cache = RedisCache()
                        logger.info(
                            "Global Redis backend initialized for BlazingFastCache L2"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to init Redis L2 backend: {e}")
                        cls._redis_cache = None
        return cls._redis_cache

    # Lock namespacing for distributed locks
    REDIS_LOCK_PREFIX = "bfcache:lock:"
    REDIS_LOCK_TTL = 10  # seconds before lock auto-expires

    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()

        # Multi-level cache storage
        self.l1_cache = OrderedDict()  # L1: Memory (fastest)
        self.l2_cache = {}  # L2: In-memory fallback when Redis is unavailable
        self.l3_cache = []  # L3: Semantic similarity
        self.l4_cache = []  # L4: Vector similarity

        # Wire up Redis backend
        self._redis = self._get_redis_backend()

        # Performance tracking
        self.stats = {
            "l1_hits": 0,
            "l1_misses": 0,
            "l2_hits": 0,
            "l2_misses": 0,
            "l3_hits": 0,
            "l3_misses": 0,
            "l4_hits": 0,
            "l4_misses": 0,
            "total_requests": 0,
            "avg_response_time": 0.0,
        }

        # Thread safety
        self.locks = {
            "l1": threading.RLock(),
            "l2": threading.RLock(),
            "l3": threading.RLock(),
            "l4": threading.RLock(),
            "stats": threading.RLock(),
        }

        # Background executor for async operations
        self.executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="CacheWorker"
        )

        # Predictive cache for anticipated queries
        self.predictive_cache = OrderedDict()

        logger.info(
            f"Cache initialized with {self.config.l1_capacity} L1, L2={'Redis' if self._redis else 'dict fallback'}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self.locks["stats"]:
            total_hits = (
                self.stats["l1_hits"]
                + self.stats["l2_hits"]
                + self.stats["l3_hits"]
                + self.stats["l4_hits"]
            )
            total_misses = (
                self.stats["l1_misses"]
                + self.stats["l2_misses"]
                + self.stats["l3_misses"]
                + self.stats["l4_misses"]
            )

            return {
                **self.stats,
                "total_hits": total_hits,
                "total_misses": total_misses,
                "hit_rate": total_hits / max(1, total_hits + total_misses),
                "l1_hit_rate": self.stats["l1_hits"]
                / max(1, self.stats["l1_hits"] + self.stats["l1_misses"]),
                "l2_hit_rate": self.stats["l2_hits"]
                / max(1, self.stats["l2_hits"] + self.stats["l2_misses"]),
                "l3_hit_rate": self.stats["l3_hits"]
                / max(1, self.stats["l3_hits"] + self.stats["l3_misses"]),
                "l4_hit_rate": self.stats["l4_hits"]
                / max(1, self.stats["l4_hits"] + self.stats["l4_misses"]),
                "cache_sizes": {
                    "l1": len(self.l1_cache),
                    "l2": len(self.l2_cache),
                    "l3": len(self.l3_cache),
                    "l4": len(self.l4_cache),
                    "predictive": len(self.predictive_cache),
                },
            }

    def get(
        self,
        key: str,
        query_embedding: Optional[np.ndarray] = None,
        vector_embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[Any]:
        """Multi-level cache get with semantic and vector similarity"""
        start_time = time.time()

        with self.locks["stats"]:
            self.stats["total_requests"] += 1

        # 1. L1 Memory Cache (nanoseconds)
        with self.locks["l1"]:
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                if self._is_entry_valid(entry):
                    entry.access_count += 1
                    entry.last_accessed = time.time()
                    self.l1_cache.move_to_end(key)  # LRU update

                    with self.locks["stats"]:
                        self.stats["l1_hits"] += 1

                    self._update_avg_response_time(time.time() - start_time)
                    logger.debug(f"[INFO] L1 cache hit for key: {key[:50]}...")
                    return entry.data
                else:
                    # Expired entry
                    del self.l1_cache[key]

        with self.locks["stats"]:
            self.stats["l1_misses"] += 1

        # 2. L2 Redis Cache (microseconds)
        hit = False
        if self._redis:
            try:
                ser = self._redis.get(key)
                if ser is not None:
                    entry = CacheEntry(
                        data=ser,
                        timestamp=time.time(),
                        ttl=AI_TTL_MAP["default"],
                    )
                    hit = True
                    # Promote to L1
                    self._promote_to_l1(key, entry)
                    with self.locks["stats"]:
                        self.stats["l2_hits"] += 1
                    self._update_avg_response_time(time.time() - start_time)
                    logger.debug(f"[INFO] L2 Redis hit for key: {key[:50]}...")
                    return entry.data
            except Exception as e:
                logger.warning(f"L2 Redis get failed, falling back: {e}")

        if not hit:
            # Fallback to in-memory L2 dict
            with self.locks["l2"]:
                if key in self.l2_cache:
                    entry = self.l2_cache[key]
                    if self._is_entry_valid(entry):
                        entry.access_count += 1
                        entry.last_accessed = time.time()

                        # Promote to L1
                        self._promote_to_l1(key, entry)

                        with self.locks["stats"]:
                            self.stats["l2_hits"] += 1

                        self._update_avg_response_time(time.time() - start_time)
                        logger.debug(f"[INFO] L2 dict hit for key: {key[:50]}...")
                        return entry.data
                    else:
                        del self.l2_cache[key]
                else:
                    with self.locks["stats"]:
                        self.stats["l2_misses"] += 1

        # 3. L3 Semantic Cache (milliseconds)
        if query_embedding is not None:
            with self.locks["l3"]:
                semantic_result = self._find_semantic_match(query_embedding)
                if semantic_result:
                    with self.locks["stats"]:
                        self.stats["l3_hits"] += 1

                    self._update_avg_response_time(time.time() - start_time)
                    logger.debug(
                        f"[INFO] L3 semantic cache hit with similarity: {semantic_result['similarity']:.3f}"
                    )
                    return semantic_result["data"]

        with self.locks["stats"]:
            self.stats["l3_misses"] += 1

        # 4. L4 Vector Cache (tens of milliseconds)
        if vector_embedding is not None:
            with self.locks["l4"]:
                vector_result = self._find_vector_match(vector_embedding)
                if vector_result:
                    with self.locks["stats"]:
                        self.stats["l4_hits"] += 1

                    self._update_avg_response_time(time.time() - start_time)
                    logger.debug(
                        f"[INFO] L4 vector cache hit with similarity: {vector_result['similarity']:.3f}"
                    )
                    return vector_result["data"]

        with self.locks["stats"]:
            self.stats["l4_misses"] += 1

        self._update_avg_response_time(time.time() - start_time)
        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        query_embedding: Optional[np.ndarray] = None,
        vector_embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Multi-level cache set with intelligent TTL"""

        # Determine TTL based on content and metadata
        if ttl is None:
            ttl = self._calculate_smart_ttl(value, metadata)

        entry = CacheEntry(
            data=value,
            timestamp=time.time(),
            ttl=ttl,
            semantic_embedding=query_embedding,
            vector_embedding=vector_embedding,
            metadata=metadata or {},
        )

        # 1. Store in L1 Memory Cache
        with self.locks["l1"]:
            self.l1_cache[key] = entry
            self.l1_cache.move_to_end(key)

            # Evict old entries if over capacity
            while len(self.l1_cache) > self.config.l1_capacity:
                self.l1_cache.popitem(last=False)

        # 2. Store in L2 Redis Cache
        if self._redis:
            try:
                self._redis.set(key, entry.data, ttl)
            except Exception as e:
                logger.warning(f"L2 Redis set failed, falling back to dict: {e}")
                self._fallback_l2_set(key, entry)
        else:
            self._fallback_l2_set(key, entry)

        # 3. Store in L3 Semantic Cache
        if query_embedding is not None:
            with self.locks["l3"]:
                self._add_semantic_entry(key, entry)

        # 4. Store in L4 Vector Cache
        if vector_embedding is not None:
            with self.locks["l4"]:
                self._add_vector_entry(key, entry)

        logger.debug(f"[INFO] Cached key: {key[:50]}... with TTL: {ttl}s")

    def _fallback_l2_set(self, key: str, entry: CacheEntry):
        """Fallback L2 store to in-memory dict when Redis is unavailable"""
        with self.locks["l2"]:
            self.l2_cache[key] = entry
            while len(self.l2_cache) > self.config.l2_capacity:
                lru_key = min(
                    self.l2_cache.keys(), key=lambda k: self.l2_cache[k].last_accessed
                )
                del self.l2_cache[lru_key]

    def _is_entry_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid"""
        return (time.time() - entry.timestamp) < entry.ttl

    def _promote_to_l1(self, key: str, entry: CacheEntry) -> None:
        """Promote entry from L2 to L1 cache"""
        with self.locks["l1"]:
            self.l1_cache[key] = entry
            self.l1_cache.move_to_end(key)

            # Evict if necessary
            while len(self.l1_cache) > self.config.l1_capacity:
                self.l1_cache.popitem(last=False)

    def _find_semantic_match(self, query_embedding: np.ndarray) -> Optional[Dict]:
        """Find semantic match in L3 cache"""
        if not self.l3_cache:
            return None

        best_similarity = -1
        best_match = None

        for entry in self.l3_cache:
            if entry["embedding"] is not None:
                similarity = self._cosine_similarity(
                    query_embedding, entry["embedding"]
                )
                if (
                    similarity > best_similarity
                    and similarity >= self.config.semantic_threshold
                ):
                    best_similarity = similarity
                    best_match = entry

        if best_match:
            return {"data": best_match["entry"].data, "similarity": best_similarity}

        return None

    def _find_vector_match(self, vector_embedding: np.ndarray) -> Optional[Dict]:
        """Find vector match in L4 cache"""
        if not self.l4_cache:
            return None

        best_similarity = -1
        best_match = None

        for entry in self.l4_cache:
            if entry["embedding"] is not None:
                similarity = self._cosine_similarity(
                    vector_embedding, entry["embedding"]
                )
                if (
                    similarity > best_similarity
                    and similarity >= self.config.vector_threshold
                ):
                    best_similarity = similarity
                    best_match = entry

        if best_match:
            return {"data": best_match["entry"].data, "similarity": best_similarity}

        return None

    def _add_semantic_entry(self, key: str, entry: CacheEntry) -> None:
        """Add entry to semantic cache"""
        if entry.semantic_embedding is not None:
            self.l3_cache.append(
                {"key": key, "entry": entry, "embedding": entry.semantic_embedding}
            )

            # Limit cache size
            if len(self.l3_cache) > 1000:
                self.l3_cache.pop(0)

    def _add_vector_entry(self, key: str, entry: CacheEntry) -> None:
        """Add entry to vector cache"""
        if entry.vector_embedding is not None:
            self.l4_cache.append(
                {"key": key, "entry": entry, "embedding": entry.vector_embedding}
            )

            # Limit cache size
            if len(self.l4_cache) > 500:
                self.l4_cache.pop(0)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def _calculate_smart_ttl(self, value: Any, metadata: Optional[Dict]) -> int:
        """AI-powered TTL calculation based on content analysis"""

        # Extract text content for analysis
        text_content = ""
        if isinstance(value, str):
            text_content = value.lower()
        elif isinstance(value, dict):
            text_content = str(value).lower()

        # Analyze content type
        if any(
            keyword in text_content
            for keyword in ["law", "act", "bill", "constitution", "legal"]
        ):
            return AI_TTL_MAP["legal"]
        elif any(
            keyword in text_content
            for keyword in ["news", "current", "today", "breaking", "latest"]
        ):
            return AI_TTL_MAP["news"]
        elif any(
            keyword in text_content for keyword in ["trending", "viral", "popular"]
        ):
            return AI_TTL_MAP["trending"]
        elif any(
            keyword in text_content
            for keyword in ["parliament", "debate", "session", "mp"]
        ):
            return AI_TTL_MAP["parliamentary"]
        elif any(
            keyword in text_content
            for keyword in ["calculator", "compute", "calculate"]
        ):
            return AI_TTL_MAP["calculator"]
        elif any(
            keyword in text_content for keyword in ["history", "historical", "past"]
        ):
            return AI_TTL_MAP["historical"]

        # Check metadata for explicit TTL
        if metadata and "ttl_type" in metadata:
            return AI_TTL_MAP.get(metadata["ttl_type"], AI_TTL_MAP["default"])

        return AI_TTL_MAP["default"]

    def _update_avg_response_time(self, response_time: float):
        """Update average response time statistics"""
        with self.locks["stats"]:
            total_requests = self.stats["total_requests"]
            if total_requests == 1:
                self.stats["avg_response_time"] = response_time
            else:
                self.stats["avg_response_time"] = (
                    self.stats["avg_response_time"] * (total_requests - 1)
                    + response_time
                ) / total_requests

    def clear_expired_entries(self):
        """Clean up expired cache entries"""
        current_time = time.time()

        # Clean L1 cache
        with self.locks["l1"]:
            expired_keys = [
                key
                for key, entry in self.l1_cache.items()
                if not self._is_entry_valid(entry)
            ]
            for key in expired_keys:
                del self.l1_cache[key]

        # Clean L2 cache (Redis handles its own TTL, clean dict fallback)
        with self.locks["l2"]:
            expired_keys = [
                key
                for key, entry in self.l2_cache.items()
                if not self._is_entry_valid(entry)
            ]
            for key in expired_keys:
                del self.l2_cache[key]

        logger.info(f"[INFO] Cleaned {len(expired_keys)} expired entries")

    def preload_cache(
        self, key_value_pairs: List[Tuple[str, Any]], ttl: Optional[int] = None
    ):
        """Preload cache with key-value pairs for warm start"""
        logger.info(f"[INFO] Preloading {len(key_value_pairs)} items into cache")

        for key, value in key_value_pairs:
            self.set(key, value, ttl)

        logger.info("[DONE] Cache preloading completed")


class RedisCache:
    """High-performance Redis cache with connection pooling and compression"""

    def __init__(self, redis_url: Optional[str] = None, max_connections: int = 20):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.max_connections = max_connections

        # Connection pool for thread safety
        self.connection_pool = None
        self._init_redis_connection()

        # Compression settings
        self.compression_threshold = 1024  # Compress values larger than 1KB

        logger.info(
            f"[INFO] Redis Cache initialized with {max_connections} connections"
        )

    def _init_redis_connection(self):
        """Initialize Redis connection pool"""
        try:
            if STANDARD_REDIS_AVAILABLE:
                import redis

                self.connection_pool = redis.ConnectionPool.from_url(
                    self.redis_url,
                    max_connections=self.max_connections,
                    socket_keepalive=True,
                    socket_keepalive_options={},
                    health_check_interval=30,
                )
                self.client = redis.Redis(connection_pool=self.connection_pool)

                # Test connection
                self.client.ping()
                logger.info("[DONE] Standard Redis connection established")

            elif UPSTASH_AVAILABLE:
                self.client = UpstashRedis(url=self.redis_url)
                logger.info("[DONE] Upstash Redis connection established")

            else:
                logger.error("[FAILED] No Redis client available")
                self.client = None

        except Exception as e:
            logger.error(f"[FAILED] Failed to initialize Redis: {e}")
            self.client = None

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis with automatic decompression"""
        if not self.client:
            return None

        try:
            value = self.client.get(key)
            if value is None:
                return None

            # Handle compressed values
            if isinstance(value, bytes) and value.startswith(b"\x00COMPRESSED"):
                import gzip

                compressed_data = value[12:]  # Remove compression marker
                decompressed = gzip.decompress(compressed_data)
                return pickle.loads(decompressed)

            # Handle regular JSON values
            if isinstance(value, bytes):
                value = value.decode("utf-8")

            return json.loads(value)

        except Exception as e:
            logger.error(f"Redis get failed for key {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis with intelligent compression"""
        if not self.client:
            return False

        try:
            # Serialize value
            serialized = json.dumps(value)

            # Compress if value is large
            if len(serialized) > self.compression_threshold:
                import gzip

                compressed = gzip.compress(serialized.encode("utf-8"))
                final_value = b"\x00COMPRESSED" + compressed
            else:
                final_value = serialized

            # Store with optional TTL
            if ttl:
                self.client.setex(key, ttl, final_value)
            else:
                self.client.set(key, final_value)

            return True

        except Exception as e:
            logger.error(f"Redis set failed for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        if not self.client:
            return False

        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Redis delete failed for key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        if not self.client:
            return False

        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Redis exists check failed for key {key}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get Redis connection statistics"""
        if not self.client:
            return {"status": "disconnected"}

        try:
            info = self.client.info()
            return {
                "status": "connected",
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0)
                / max(1, info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0)),
            }
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {"status": "error", "error": str(e)}


# Advanced cache decorator with intelligent TTL and multi-level caching
def intelligent_cache(
    ttl: Optional[int] = None,
    key_prefix: str = "",
    enable_semantic: bool = True,
    enable_vector: bool = True,
    compress_large_values: bool = True,
):
    """
    Intelligent caching decorator with AI-powered TTL and multi-level caching

    Args:
        ttl: Time to live in seconds (None for smart TTL)
        key_prefix: Prefix for cache keys
        enable_semantic: Enable semantic similarity caching
        enable_vector: Enable vector similarity caching
        compress_large_values: Compress large cached values
    """

    def decorator(func):
        # Initialize cache instance
        cache = BlazingFastCache()

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = (
                f"{key_prefix}{func.__name__}:{_generate_cache_key(args, kwargs)}"
            )

            # Try to get from cache
            # For semantic/vector caching, we'd need embeddings from the function context
            cached_result = cache.get(cache_key)

            if cached_result is not None:
                logger.debug(f"[INFO] Cache hit for {func.__name__}")
                return cached_result

            # Execute function
            result = func(*args, **kwargs)

            # Cache result with smart TTL
            cache.set(cache_key, result, ttl)

            return result

        # Add cache control methods to wrapper
        wrapper.cache_clear = lambda: cache.clear_expired_entries()
        wrapper.cache_stats = lambda: cache.get_stats()
        wrapper.cache_instance = cache

        return wrapper

    return decorator


def _generate_cache_key(args: tuple, kwargs: dict) -> str:
    """Generate deterministic cache key from function arguments"""
    # Create a stable representation of arguments
    key_parts = []

    # Add positional arguments
    for arg in args:
        if hasattr(arg, "__dict__"):
            # Handle objects with __dict__
            key_parts.append(str(sorted(arg.__dict__.items())))
        elif isinstance(arg, (dict, list, set)):
            # Handle collections
            key_parts.append(json.dumps(arg, sort_keys=True))
        else:
            key_parts.append(str(arg))

    # Add keyword arguments
    for key, value in sorted(kwargs.items()):
        if isinstance(value, (dict, list, set)):
            key_parts.append(f"{key}={json.dumps(value, sort_keys=True)}")
        else:
            key_parts.append(f"{key}={value}")

    # Generate hash
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


# Specialized RAG cache for query-result caching
class RAGCache:
    """Specialized cache for RAG query-result pairs with semantic similarity"""

    def __init__(self, capacity: int = 1000, semantic_threshold: float = 0.85):
        self.cache = BlazingFastCache(
            CacheConfig(l1_capacity=capacity, semantic_threshold=semantic_threshold)
        )
        self.query_embeddings = {}  # Store query embeddings for similarity
        self.capacity = capacity

        logger.info(f" RAG Cache initialized with capacity: {capacity}")

    def get_similar_query(
        self, query: str, query_embedding: np.ndarray
    ) -> Optional[Dict]:
        """Get cached result for similar query"""
        cache_key = f"rag_query:{hashlib.md5(query.encode()).hexdigest()}"

        # Store embedding for future similarity searches
        self.query_embeddings[cache_key] = query_embedding

        # Try to get exact match first
        result = self.cache.get(cache_key, query_embedding=query_embedding)
        if result:
            logger.info(f"[INFO] RAG cache hit for query: {query[:50]}...")
            return result

        return None

    def cache_query_result(
        self,
        query: str,
        query_embedding: np.ndarray,
        result: Dict,
        ttl: Optional[int] = None,
    ) -> None:
        """Cache query-result pair"""
        cache_key = f"rag_query:{hashlib.md5(query.encode()).hexdigest()}"

        # Store embedding
        self.query_embeddings[cache_key] = query_embedding

        # Cache result
        self.cache.set(cache_key, result, ttl=ttl, query_embedding=query_embedding)

        # Cleanup old embeddings if over capacity
        if len(self.query_embeddings) > self.capacity * 2:
            # Remove oldest embeddings
            oldest_keys = list(self.query_embeddings.keys())[: self.capacity]
            for key in oldest_keys:
                if key in self.query_embeddings:
                    del self.query_embeddings[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG cache statistics"""
        stats = self.cache.get_stats()
        stats["query_embeddings_stored"] = len(self.query_embeddings)
        return stats

    def clear(self):
        """Clear RAG cache"""
        self.cache.l1_cache.clear()
        self.cache.l2_cache.clear()
        self.query_embeddings.clear()
        logger.info("[INFO] RAG cache cleared")


# Global cache instances for easy access
_blazing_cache = BlazingFastCache()
_rag_cache = RAGCache()


def get_blazing_cache() -> BlazingFastCache:
    """Get global blazing fast cache instance"""
    return _blazing_cache


def get_rag_cache() -> RAGCache:
    """Get global RAG cache instance"""
    return _rag_cache


# Utility functions for cache management
def cache_clear_all():
    """Clear all global caches"""
    _blazing_cache.clear_expired_entries()
    _rag_cache.clear()
    logger.info("[INFO] All global caches cleared")


def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics"""
    return {
        "blazing_cache": _blazing_cache.get_stats(),
        "rag_cache": _rag_cache.get_stats(),
    }


class CacheManager:
    """Compatibility wrapper with distributed lock and request coalescing"""

    def __init__(self, config_manager=None):
        self.blazing_cache = _blazing_cache
        self.redis_cache = RedisCache()
        self.redis_client = self.redis_cache.client

        # Request coalescing: tracks in-flight computations so duplicate requests
        # wait for the first one to complete rather than all hitting the LLM.
        self._in_flight: Dict[str, asyncio.Future] = {}
        self._in_flight_lock = threading.Lock()

    async def get_or_compute(
        self, key: str, compute_func: Callable, ttl_type: str = "default"
    ) -> Any:
        """Get from cache or compute with distributed lock + request coalescing"""
        cache_key = f"rag_query:{hashlib.md5(key.encode()).hexdigest()}"

        # 1. Try cache first
        cached = self.blazing_cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit: {key[:50]}...")
            return cached

        # 2. Request coalescing: check if same key is being computed by another request
        with self._in_flight_lock:
            if cache_key in self._in_flight:
                future = self._in_flight[cache_key]
                logger.info(f"Coalescing duplicate request: {key[:50]}...")
            else:
                future = None

        if future is not None:
            return await future

        # 3. Distributed lock via Redis to prevent thundering herd across pods
        lock_key = f"{BlazingFastCache.REDIS_LOCK_PREFIX}{cache_key}"
        lock_acquired = False
        lock_ttl = BlazingFastCache.REDIS_LOCK_TTL

        if self.redis_client:
            try:
                # SET NX with TTL — atomic lock acquire
                import secrets

                lock_token = secrets.token_hex(16)
                acquired = self.redis_cache.client.set(
                    lock_key, lock_token, nx=True, ex=lock_ttl
                )
                if acquired:
                    lock_acquired = True
            except Exception as e:
                logger.warning(f"Distributed lock failed, proceeding without: {e}")

        if not lock_acquired:
            # Lock held by another pod — wait briefly and retry cache
            await asyncio.sleep(0.05)
            cached = self.blazing_cache.get(cache_key)
            if cached is not None:
                return cached

        # 4. Create future for this computation (coalescing)
        future = asyncio.get_event_loop().create_future()
        with self._in_flight_lock:
            self._in_flight[cache_key] = future

        try:
            result = await compute_func()

            ttl = AI_TTL_MAP.get(ttl_type, AI_TTL_MAP["default"])
            self.blazing_cache.set(cache_key, result, ttl=ttl)

            # Resolve coalesced futures
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            with self._in_flight_lock:
                self._in_flight.pop(cache_key, None)
            # Release distributed lock
            if lock_acquired and self.redis_client:
                try:
                    # Only delete if our token matches (safe release)
                    current_val = self.redis_cache.client.get(lock_key)
                    if current_val and current_val.decode() == lock_token:
                        self.redis_cache.client.delete(lock_key)
                except Exception:
                    pass

        return result

    def delete_pattern(self, pattern: str):
        if self.redis_client:
            try:
                keys = self.redis_client.keys(pattern)
                for k in keys:
                    self.redis_client.delete(k)
            except Exception as e:
                logger.error(f"Failed to delete pattern {pattern}: {e}")


# Keep track of global instance
_cache_manager_instance = None


def get_cache_manager(config_manager=None) -> CacheManager:
    """Get the global CacheManager compatibility instance"""
    global _cache_manager_instance
    if _cache_manager_instance is None:
        _cache_manager_instance = CacheManager(config_manager)
    return _cache_manager_instance
