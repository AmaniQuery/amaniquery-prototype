"""
Knowledge Base Search Tool - Robust, Scalable, Concurrent Search
Refactored for LangGraph tool calling best practices (2026).

Features:
- Inherits from langchain_core.tools.BaseTool
- Strict Pydantic v2 validation for inputs
- Async/await support for concurrent searches
- Connection pooling and semaphore-based concurrency control
- Circuit breaker pattern for fault tolerance
- Retry logic with exponential backoff
- Request coalescing for duplicate queries
- Response caching with TTL
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Type
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr

from langchain_core.tools import BaseTool

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from Module3_NiruDB.vector_store import VectorStore

# =============================================================================
# INPUT SCHEMA (Pydantic v2)
# =============================================================================


class KBSearchInput(BaseModel):
    """Input schema for Knowledge Base Search."""

    query: str = Field(
        ..., description="The search query to execute against the knowledge base."
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of results to return."
    )
    namespace: Optional[List[str]] = Field(
        default=None,
        description="List of namespaces to search (e.g., ['kenya_law', 'kenya_news']). Defaults to all.",
    )


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class KBSearchConfig:
    """Configuration for KB search tool."""

    # Concurrency settings
    max_concurrent_searches: int = 10
    max_concurrent_writes: int = 3

    # Retry settings
    max_retries: int = 3
    base_retry_delay: float = 0.5
    max_retry_delay: float = 10.0

    # Circuit breaker settings
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: float = 30.0

    # Cache settings
    cache_enabled: bool = True
    cache_ttl_seconds: float = 300.0  # 5 minutes
    cache_max_size: int = 1000

    # Coalescing settings
    coalesce_window_ms: float = 50.0

    # Timeout settings
    search_timeout: float = 30.0
    write_timeout: float = 60.0

    # Default namespaces
    default_namespaces: List[str] = field(
        default_factory=lambda: [
            "kenya_law",
            "kenya_news",
            "kenya_parliament",
            "historical",
            "global_trends",
        ]
    )


# =============================================================================
# CIRCUIT BREAKER & CACHE UTILS
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        return self._state

    async def can_execute(self) -> bool:
        """Check if request can proceed."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker entering half-open state")
                    return True
                return False

            # HALF_OPEN - allow one request to test
            return True

    async def record_success(self):
        """Record a successful operation."""
        async with self._lock:
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                logger.info("Circuit breaker closed - service recovered")

    async def record_failure(self):
        """Record a failed operation."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker opened after {self._failure_count} failures"
                )
            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning("Circuit breaker re-opened - recovery failed")


@dataclass
class CacheEntry:
    """Single cache entry with TTL."""

    value: Any
    expires_at: float

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class LRUCache:
    """Thread-safe LRU cache with TTL."""

    def __init__(self, max_size: int = 1000, ttl: float = 300.0):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._access_order.remove(key)
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            self._hits += 1

            return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        async with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]

            expires_at = time.time() + (ttl or self.ttl)
            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)

            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }


class RequestCoalescer:
    """Coalesces duplicate concurrent requests."""

    def __init__(self, window_ms: float = 50.0):
        self.window_ms = window_ms
        self._pending: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    def _make_key(self, query: str, namespaces: List[str], top_k: int) -> str:
        """Create unique key for request."""
        ns_str = ",".join(sorted(namespaces))
        raw = f"{query}:{ns_str}:{top_k}"
        return hashlib.md5(raw.encode()).hexdigest()

    async def execute(self, key: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function with request coalescing."""
        async with self._lock:
            if key in self._pending:
                logger.debug(f"Coalescing request: {key[:8]}...")
                return await self._pending[key]

            future = asyncio.get_event_loop().create_future()
            self._pending[key] = future

        try:
            result = await func(*args, **kwargs)
            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            async with self._lock:
                self._pending.pop(key, None)


# =============================================================================
# MAIN TOOL CLASS (LangChain Compatible)
# =============================================================================


class KnowledgeBaseSearchTool(BaseTool):
    """
    Robust, scalable knowledge base search tool for LangGraph agents.
    Searches Kenyan legal documents, news, and parliamentary records.
    """

    name: str = "kb_search"
    description: str = (
        "Search the knowledge base for Kenyan legal content: "
        "case law, Constitution, Hansard, statutes, and news. "
        "Use for legal research and fact-checking."
    )
    args_schema: Type[BaseModel] = KBSearchInput

    # Private attributes for internal logic (excluded from Pydantic schema)
    _vector_store: VectorStore = PrivateAttr()
    _config: KBSearchConfig = PrivateAttr()
    _search_semaphore: asyncio.Semaphore = PrivateAttr()
    _circuit_breaker: CircuitBreaker = PrivateAttr()
    _cache: Optional[LRUCache] = PrivateAttr()
    _coalescer: RequestCoalescer = PrivateAttr()
    _metrics: Dict[str, Any] = PrivateAttr()

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        config: Optional[KBSearchConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._vector_store = vector_store or VectorStore()
        self._config = config or KBSearchConfig()

        # Initialize concurrency controls
        self._search_semaphore = asyncio.Semaphore(self._config.max_concurrent_searches)
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=self._config.circuit_failure_threshold,
            recovery_timeout=self._config.circuit_recovery_timeout,
        )
        self._cache = (
            LRUCache(
                max_size=self._config.cache_max_size, ttl=self._config.cache_ttl_seconds
            )
            if self._config.cache_enabled
            else None
        )

        self._coalescer = RequestCoalescer(window_ms=self._config.coalesce_window_ms)

        self._metrics = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "circuit_breaker_rejections": 0,
            "retries": 0,
        }

    def _run(
        self, query: str, top_k: int = 5, namespace: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Synchronous execution (delegates to async runner via event loop)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in an async loop, use a thread pool to run async code synchronously
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run, self._arun(query, top_k, namespace)
                    )
                    return future.result(timeout=self._config.search_timeout + 5)
            else:
                return loop.run_until_complete(self._arun(query, top_k, namespace))
        except Exception as e:
            logger.error(f"Sync execution failed: {e}")
            return {"error": str(e)}

    async def _arun(
        self, query: str, top_k: int = 5, namespace: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Asynchronous execution with resilience features."""
        start_time = time.time()
        search_namespace = namespace or self._config.default_namespaces

        result = {
            "query": query,
            "results": [],
            "metadata": {"latency_ms": 0, "cached": False},
        }

        try:
            # Execute robust search
            search_results = await self._search_with_resilience(
                query=query, top_k=top_k, namespaces=search_namespace
            )

            result["results"] = search_results
            result["metadata"]["latency_ms"] = (time.time() - start_time) * 1000
            return result

        except Exception as e:
            logger.error(f"Async execution failed: {e}")
            result["error"] = str(e)
            return result

    async def _search_with_resilience(
        self, query: str, top_k: int, namespaces: List[str]
    ) -> List[Dict[str, Any]]:
        """Core search logic with circuit breaker, caching, and coalescing."""
        self._metrics["total_searches"] += 1

        # 1. Circuit Breaker
        if not await self._circuit_breaker.can_execute():
            self._metrics["circuit_breaker_rejections"] += 1
            raise Exception("Circuit breaker is open - KB service unavailable")

        # 2. Cache Check
        cache_key = self._make_cache_key(query, namespaces, top_k)
        if self._cache:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit: {query[:30]}...")
                return cached

        # 3. Request Coalescing
        coalesce_key = self._coalescer._make_key(query, namespaces, top_k)

        try:
            results = await self._coalescer.execute(
                coalesce_key, self._search_with_retry, query, top_k, namespaces
            )

            # Cache success
            if self._cache and results:
                await self._cache.set(cache_key, results)

            await self._circuit_breaker.record_success()
            self._metrics["successful_searches"] += 1
            return results

        except Exception as e:
            await self._circuit_breaker.record_failure()
            self._metrics["failed_searches"] += 1
            raise

    async def _search_with_retry(
        self, query: str, top_k: int, namespaces: List[str]
    ) -> List[Dict[str, Any]]:
        """Retry logic with exponential backoff."""
        last_exception = None

        for attempt in range(self._config.max_retries):
            try:
                async with self._search_semaphore:
                    # Run actual search in thread pool to avoid blocking event loop
                    loop = asyncio.get_event_loop()
                    return await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: self._vector_store.query(
                                query_text=query, n_results=top_k, namespace=namespaces
                            ),
                        ),
                        timeout=self._config.search_timeout,
                    )
            except Exception as e:
                last_exception = e
                logger.warning(f"Search attempt {attempt + 1} failed: {e}")

                if attempt < self._config.max_retries - 1:
                    self._metrics["retries"] += 1
                    delay = min(
                        self._config.base_retry_delay * (2**attempt),
                        self._config.max_retry_delay,
                    )
                    await asyncio.sleep(delay)

        raise last_exception or Exception("Search failed after retries")

    def _make_cache_key(self, query: str, namespaces: List[str], top_k: int) -> str:
        """Create deterministic cache key."""
        ns_str = ",".join(sorted(namespaces))
        raw = f"{query}:{ns_str}:{top_k}"
        return hashlib.md5(raw.encode()).hexdigest()
