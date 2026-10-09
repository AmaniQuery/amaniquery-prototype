"""
Web Search Tool - Robust, Production-Ready Web Search
Refactored for LangGraph tool calling best practices (2026).

Features:
- Inherits from langchain_core.tools.BaseTool
- Strict Pydantic v2 validation for inputs
- Async/await support
- Multiple fallback providers (DuckDuckGo, Tavily, SearXNG)
- Retry logic with exponential backoff
- Rate limiting
- Result caching
"""

import os
import asyncio
import time
import hashlib
from typing import Dict, Any, List, Optional, Type
from datetime import datetime
from dataclasses import dataclass, field
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr

from langchain_core.tools import BaseTool

# Provider imports
try:
    from duckduckgo_search import DDGS

    DDGS_AVAILABLE = True
except ImportError:
    try:
        from ddgs import DDGS

        DDGS_AVAILABLE = True
    except ImportError:
        DDGS = None
        DDGS_AVAILABLE = False
        logger.warning("ddgs package not installed. Install with: pip install ddgs")

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# =============================================================================
# INPUT SCHEMA (Pydantic v2)
# =============================================================================


class WebSearchInput(BaseModel):
    """Input schema for Web Search."""

    query: str = Field(..., description="The search query to execute.")
    max_results: int = Field(
        default=10, ge=1, le=50, description="Maximum number of results to return."
    )
    region: str = Field(
        default="us-en", description="Search region (e.g., 'us-en', 'ke-en')."
    )
    search_type: str = Field(
        default="text", description="Type of search: 'text' or 'news'."
    )
    time_range: Optional[str] = Field(
        default=None,
        description="Time range: 'd' (day), 'w' (week), 'm' (month), 'y' (year).",
    )


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class WebSearchConfig:
    """Configuration for web search tool."""

    max_retries: int = 3
    base_retry_delay: float = 1.0
    max_retry_delay: float = 10.0
    rate_limit_delay: float = 1.0
    timeout: float = 30.0
    cache_enabled: bool = True
    cache_ttl: float = 300.0  # 5 minutes
    default_max_results: int = 10
    default_region: str = "us-en"
    # Fallback providers
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    searxng_url: str = field(default_factory=lambda: os.getenv("SEARXNG_URL", ""))


# =============================================================================
# MAIN TOOL CLASS (LangChain Compatible)
# =============================================================================


class WebSearchTool(BaseTool):
    """
    Robust web search tool for LangGraph agents.
    Search the web for current information, news, and general knowledge.
    """

    name: str = "web_search"
    description: str = (
        "Search the web for current information, news, and general knowledge. "
        "Best for: recent events, general questions, fact-checking, research."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    # Private attributes
    _config: WebSearchConfig = PrivateAttr()
    _ddgs: Optional[Any] = PrivateAttr()
    _last_request_time: float = PrivateAttr()
    _cache: Dict[str, Dict] = PrivateAttr()
    _cache_times: Dict[str, float] = PrivateAttr()
    _metrics: Dict[str, Any] = PrivateAttr()

    def __init__(self, config: Optional[WebSearchConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self._config = config or WebSearchConfig()

        # Initialize providers
        self._ddgs = DDGS() if DDGS_AVAILABLE else None

        # Rate limiting
        self._last_request_time = 0.0

        # Simple cache
        self._cache = {}
        self._cache_times = {}

        # Metrics
        self._metrics = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "cache_hits": 0,
            "retries": 0,
            "provider_fallbacks": 0,
        }

        logger.info(f"WebSearchTool initialized (ddgs: {DDGS_AVAILABLE})")

    def _run(
        self,
        query: str,
        max_results: int = 10,
        region: str = "us-en",
        search_type: str = "text",
        time_range: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous execution (delegates to async runner via event loop)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self._arun(query, max_results, region, search_type, time_range),
                    )
                    return future.result(timeout=self._config.timeout + 5)
            else:
                return loop.run_until_complete(
                    self._arun(query, max_results, region, search_type, time_range)
                )
        except Exception as e:
            logger.error(f"Sync execution failed: {e}")
            return self._error_response(query, str(e))

    async def _arun(
        self,
        query: str,
        max_results: int = 10,
        region: str = "us-en",
        search_type: str = "text",
        time_range: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async execution with fallback providers and retry logic."""
        if not query or not query.strip():
            return self._error_response(query, "Empty query provided")

        # Validate max_results
        max_results = max(1, min(max_results, 50))

        self._metrics["total_searches"] += 1
        start_time = time.time()

        # Check cache
        cache_key = self._make_cache_key(query, region, search_type, max_results)
        if self._config.cache_enabled:
            cached = self._get_from_cache(cache_key)
            if cached:
                self._metrics["cache_hits"] += 1
                cached["metadata"]["cached"] = True
                return cached

        # Try providers in order
        providers = [
            ("duckduckgo", self._search_ddg),
            ("tavily", self._search_tavily),
            ("searxng", self._search_searxng),
        ]

        last_error = None
        for provider_name, provider_func in providers:
            try:
                results = await self._search_with_retry(
                    provider_func,
                    query=query,
                    max_results=max_results,
                    region=region,
                    search_type=search_type,
                    time_range=time_range,
                )

                if results and results.get("results"):
                    results["metadata"] = {
                        "provider": provider_name,
                        "latency_ms": (time.time() - start_time) * 1000,
                        "cached": False,
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    # Cache successful results
                    if self._config.cache_enabled:
                        self._set_cache(cache_key, results)

                    self._metrics["successful_searches"] += 1
                    return results

            except Exception as e:
                last_error = e
                self._metrics["provider_fallbacks"] += 1
                logger.warning(f"Provider {provider_name} failed: {e}")
                continue

        # All providers failed
        self._metrics["failed_searches"] += 1
        return self._error_response(
            query, str(last_error) if last_error else "All search providers failed"
        )

    async def _search_with_retry(self, search_func, **kwargs) -> Dict[str, Any]:
        """Execute search with retry logic."""
        last_error = None

        for attempt in range(self._config.max_retries):
            try:
                # Rate limiting
                await self._rate_limit()

                return await asyncio.wait_for(
                    search_func(**kwargs), timeout=self._config.timeout
                )
            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"Search timed out after {self._config.timeout}s"
                )
            except Exception as e:
                last_error = e

            if attempt < self._config.max_retries - 1:
                self._metrics["retries"] += 1
                delay = min(
                    self._config.base_retry_delay * (2**attempt),
                    self._config.max_retry_delay,
                )
                await asyncio.sleep(delay)

        raise last_error or Exception("Search failed after retries")

    async def _search_ddg(
        self,
        query: str,
        max_results: int,
        region: str,
        search_type: str,
        time_range: Optional[str],
    ) -> Dict[str, Any]:
        """Search using DuckDuckGo."""
        if not self._ddgs:
            raise RuntimeError("DuckDuckGo not available")

        loop = asyncio.get_event_loop()

        def _do_search():
            if search_type == "news":
                results = list(
                    self._ddgs.news(
                        query,
                        max_results=max_results,
                        region=region,
                        timelimit=time_range,
                    )
                )
            else:
                results = list(
                    self._ddgs.text(
                        query,
                        max_results=max_results,
                        region=region,
                        timelimit=time_range,
                    )
                )
            return results

        raw_results = await loop.run_in_executor(None, _do_search)
        return self._format_results(query, raw_results, "duckduckgo")

    async def _search_tavily(
        self,
        query: str,
        max_results: int,
        region: str,
        search_type: str,
        time_range: Optional[str],
    ) -> Dict[str, Any]:
        """Search using Tavily API."""
        if not self._config.tavily_api_key or not HTTPX_AVAILABLE:
            raise RuntimeError("Tavily not available")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self._config.tavily_api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
                timeout=self._config.timeout,
            )
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "body": item.get("content", ""),
                    "href": item.get("url", ""),
                }
            )
        return self._format_results(query, results, "tavily")

    async def _search_searxng(
        self,
        query: str,
        max_results: int,
        region: str,
        search_type: str,
        time_range: Optional[str],
    ) -> Dict[str, Any]:
        """Search using SearXNG."""
        if not self._config.searxng_url or not HTTPX_AVAILABLE:
            raise RuntimeError("SearXNG not available")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self._config.searxng_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "pageno": 1,
                },
                timeout=self._config.timeout,
            )
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("results", [])[:max_results]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "body": item.get("content", ""),
                    "href": item.get("url", ""),
                }
            )
        return self._format_results(query, results, "searxng")

    def _format_results(
        self, query: str, raw_results: List[Dict], provider: str
    ) -> Dict[str, Any]:
        """Format raw results into standard format."""
        formatted = []
        sources = []

        for item in raw_results:
            formatted.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("body", item.get("snippet", "")),
                    "url": item.get("href", item.get("url", "")),
                }
            )
            sources.append(
                {
                    "type": "web",
                    "title": item.get("title", ""),
                    "url": item.get("href", item.get("url", "")),
                    "snippet": (item.get("body", item.get("snippet", "")))[:200],
                }
            )

        return {
            "query": query,
            "results": formatted,
            "sources": sources,
            "count": len(formatted),
        }

    def _error_response(self, query: str, error: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            "query": query,
            "results": [],
            "sources": [],
            "count": 0,
            "error": error,
        }

    async def _rate_limit(self):
        """Apply rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self._config.rate_limit_delay:
            await asyncio.sleep(self._config.rate_limit_delay - time_since_last)

        self._last_request_time = time.time()

    def _make_cache_key(
        self, query: str, region: str, search_type: str, max_results: int
    ) -> str:
        """Create cache key."""
        raw = f"{query}:{region}:{search_type}:{max_results}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """Get from cache if not expired."""
        if key not in self._cache:
            return None

        if time.time() - self._cache_times.get(key, 0) > self._config.cache_ttl:
            del self._cache[key]
            del self._cache_times[key]
            return None
        return self._cache[key].copy()

    def _set_cache(self, key: str, value: Dict):
        """Set cache entry."""
        self._cache[key] = value.copy()
        self._cache_times[key] = time.time()

        # Simple cache eviction (keep last 100)
        if len(self._cache) > 100:
            oldest_key = min(self._cache_times, key=self._cache_times.get)
            del self._cache[oldest_key]
            del self._cache_times[oldest_key]

    @property
    def metrics(self) -> Dict[str, Any]:
        """Get search metrics."""
        return self._metrics.copy()
