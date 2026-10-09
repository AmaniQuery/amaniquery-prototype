"""
News Search Tool - Multi-Provider News Search
Refactored for LangGraph tool calling best practices (2026).

Features:
- Inherits from langchain_core.tools.BaseTool
- Strict Pydantic v2 validation for inputs
- Multiple providers (SerpAPI, NewsAPI, DuckDuckGo News)
- Async/await support
- Retry logic with exponential backoff
- Kenya-focused news sources
- Result caching
"""

import os
import asyncio
import time
import hashlib
from typing import Dict, Any, List, Optional, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr

from langchain_core.tools import BaseTool

# Provider imports
try:
    from serpapi import GoogleSearch

    SERPAPI_AVAILABLE = True
except ImportError:
    GoogleSearch = None
    SERPAPI_AVAILABLE = False

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

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# =============================================================================
# INPUT SCHEMA (Pydantic v2)
# =============================================================================


class NewsSearchInput(BaseModel):
    """Input schema for News Search."""

    query: str = Field(..., description="News search query.")
    max_results: int = Field(
        default=10, ge=1, le=50, description="Maximum number of results to return."
    )
    location: str = Field(
        default="Kenya",
        description="Location for news relevance (e.g., 'Kenya', 'Global').",
    )
    time_range: Optional[str] = Field(
        default=None, description="Time range: 'd' (day), 'w' (week), 'm' (month)."
    )
    sources: Optional[List[str]] = Field(
        default=None, description="Specific sources to search (e.g., ['nation.co.ke'])."
    )


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class NewsSearchConfig:
    """Configuration for news search tool."""

    max_retries: int = 3
    base_retry_delay: float = 1.0
    timeout: float = 30.0
    cache_enabled: bool = True
    cache_ttl: float = 600.0  # 10 minutes for news
    default_max_results: int = 10
    default_location: str = "Kenya"
    # API keys
    serpapi_key: str = field(default_factory=lambda: os.getenv("SERPAPI_API_KEY", ""))
    newsapi_key: str = field(default_factory=lambda: os.getenv("NEWSAPI_KEY", ""))


# =============================================================================
# MAIN TOOL CLASS (LangChain Compatible)
# =============================================================================


class NewsSearchTool(BaseTool):
    """
    Multi-provider news search tool for LangGraph agents.
    Features SerpAPI, NewsAPI, and DuckDuckGo fallback.
    """

    name: str = "news_search"
    description: str = (
        "Search for current news and updates. "
        "Supports Kenya-focused news from sources like Nation, Standard, Star. "
        "Best for: breaking news, current events, policy updates."
    )
    args_schema: Type[BaseModel] = NewsSearchInput

    # Private attributes
    _config: NewsSearchConfig = PrivateAttr()
    _ddgs: Optional[Any] = PrivateAttr()
    _cache: Dict[str, Dict] = PrivateAttr()
    _cache_times: Dict[str, float] = PrivateAttr()
    _metrics: Dict[str, Any] = PrivateAttr()

    def __init__(self, config: Optional[NewsSearchConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self._config = config or NewsSearchConfig()

        # Initialize providers
        self._ddgs = DDGS() if DDGS_AVAILABLE else None

        # Cache
        self._cache = {}
        self._cache_times = {}

        # Metrics
        self._metrics = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "cache_hits": 0,
            "provider_fallbacks": 0,
        }

        providers = []
        if self._config.serpapi_key:
            providers.append("serpapi")
        if self._config.newsapi_key:
            providers.append("newsapi")
        if DDGS_AVAILABLE:
            providers.append("duckduckgo")

        logger.info(f"NewsSearchTool initialized (providers: {providers})")

    def _run(
        self,
        query: str,
        max_results: int = 10,
        location: str = "Kenya",
        time_range: Optional[str] = None,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Synchronous execution (delegates to async runner via event loop)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self._arun(query, max_results, location, time_range, sources),
                    )
                    return future.result(timeout=self._config.timeout + 5)
            else:
                return loop.run_until_complete(
                    self._arun(query, max_results, location, time_range, sources)
                )
        except Exception as e:
            logger.error(f"Sync execution failed: {e}")
            return self._error_response(query, str(e))

    async def _arun(
        self,
        query: str,
        max_results: int = 10,
        location: str = "Kenya",
        time_range: Optional[str] = None,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Async execution with fallback providers."""
        if not query or not query.strip():
            return self._error_response(query, "Empty query provided")

        max_results = max(1, min(max_results, 50))
        self._metrics["total_searches"] += 1
        start_time = time.time()

        # Check cache
        cache_key = self._make_cache_key(query, location, time_range, max_results)
        if self._config.cache_enabled:
            cached = self._get_from_cache(cache_key)
            if cached:
                self._metrics["cache_hits"] += 1
                cached["metadata"]["cached"] = True
                return cached

        # Try providers
        providers = [
            ("serpapi", self._search_serpapi),
            ("newsapi", self._search_newsapi),
            ("duckduckgo", self._search_ddg_news),
        ]

        last_error = None
        for provider_name, provider_func in providers:
            try:
                results = await self._search_with_retry(
                    provider_func,
                    query=query,
                    max_results=max_results,
                    location=location,
                    time_range=time_range,
                    sources=sources,
                )

                if results and results.get("results"):
                    results["metadata"] = {
                        "provider": provider_name,
                        "latency_ms": (time.time() - start_time) * 1000,
                        "cached": False,
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    if self._config.cache_enabled:
                        self._set_cache(cache_key, results)

                    self._metrics["successful_searches"] += 1
                    return results

            except Exception as e:
                last_error = e
                self._metrics["provider_fallbacks"] += 1
                logger.warning(f"News provider {provider_name} failed: {e}")
                continue

        self._metrics["failed_searches"] += 1
        return self._error_response(
            query, str(last_error) if last_error else "All news providers failed"
        )

    async def _search_with_retry(self, search_func, **kwargs) -> Dict[str, Any]:
        """Execute search with retry logic."""
        last_error = None

        for attempt in range(self._config.max_retries):
            try:
                return await asyncio.wait_for(
                    search_func(**kwargs), timeout=self._config.timeout
                )
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"News search timed out")
            except Exception as e:
                last_error = e

            if attempt < self._config.max_retries - 1:
                delay = self._config.base_retry_delay * (2**attempt)
                await asyncio.sleep(delay)

        raise last_error or Exception("News search failed")

    async def _search_serpapi(
        self,
        query: str,
        max_results: int,
        location: str,
        time_range: Optional[str],
        sources: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Search using SerpAPI Google News."""
        if not self._config.serpapi_key or not SERPAPI_AVAILABLE:
            raise RuntimeError("SerpAPI not available")

        loop = asyncio.get_event_loop()

        def _do_search():
            params = {
                "q": query,
                "tbm": "nws",
                "api_key": self._config.serpapi_key,
                "num": max_results,
                "location": location,
            }

            if time_range:
                time_map = {"d": "d", "w": "w", "m": "m", "y": "y"}
                params["tbs"] = f"qdr:{time_map.get(time_range, 'd')}"

            search = GoogleSearch(params)
            return search.get_dict()

        results = await loop.run_in_executor(None, _do_search)
        return self._format_serpapi_results(query, results.get("news_results", []))

    async def _search_newsapi(
        self,
        query: str,
        max_results: int,
        location: str,
        time_range: Optional[str],
        sources: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Search using NewsAPI."""
        if not self._config.newsapi_key or not HTTPX_AVAILABLE:
            raise RuntimeError("NewsAPI not available")

        # Calculate date range
        from_date = datetime.utcnow() - timedelta(days=30)
        if time_range == "d":
            from_date = datetime.utcnow() - timedelta(days=1)
        elif time_range == "w":
            from_date = datetime.utcnow() - timedelta(days=7)
        elif time_range == "m":
            from_date = datetime.utcnow() - timedelta(days=30)

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "apiKey": self._config.newsapi_key,
                    "pageSize": max_results,
                    "from": from_date.strftime("%Y-%m-%d"),
                    "sortBy": "publishedAt",
                    "language": "en",
                },
                timeout=self._config.timeout,
            )
            response.raise_for_status()
            data = response.json()

        return self._format_newsapi_results(query, data.get("articles", []))

    async def _search_ddg_news(
        self,
        query: str,
        max_results: int,
        location: str,
        time_range: Optional[str],
        sources: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Search using DuckDuckGo News."""
        if not self._ddgs:
            raise RuntimeError("DuckDuckGo not available")

        loop = asyncio.get_event_loop()
        search_query = query
        if location.lower() == "kenya":
            search_query = f"{query} Kenya"

        def _do_search():
            return list(
                self._ddgs.news(
                    search_query,
                    max_results=max_results,
                    timelimit=time_range,
                )
            )

        results = await loop.run_in_executor(None, _do_search)
        return self._format_ddg_results(query, results)

    def _format_serpapi_results(self, query: str, items: List[Dict]) -> Dict[str, Any]:
        """Format SerpAPI results."""
        formatted = []
        sources = []

        for item in items:
            formatted.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "source": item.get("source", ""),
                    "date": item.get("date", ""),
                    "link": item.get("link", ""),
                }
            )
            sources.append(
                {
                    "type": "news",
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "source": item.get("source", ""),
                    "snippet": (item.get("snippet", ""))[:200],
                    "date": item.get("date", ""),
                }
            )

        return {
            "query": query,
            "results": formatted,
            "sources": sources,
            "count": len(formatted),
        }

    def _format_newsapi_results(self, query: str, items: List[Dict]) -> Dict[str, Any]:
        """Format NewsAPI results."""
        formatted = []
        sources = []

        for item in items:
            formatted.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("description", ""),
                    "source": item.get("source", {}).get("name", ""),
                    "date": item.get("publishedAt", ""),
                    "link": item.get("url", ""),
                    "author": item.get("author", ""),
                }
            )
            sources.append(
                {
                    "type": "news",
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "source": item.get("source", {}).get("name", ""),
                    "snippet": (item.get("description", "") or "")[:200],
                    "date": item.get("publishedAt", ""),
                }
            )

        return {
            "query": query,
            "results": formatted,
            "sources": sources,
            "count": len(formatted),
        }

    def _format_ddg_results(self, query: str, items: List[Dict]) -> Dict[str, Any]:
        """Format DuckDuckGo news results."""
        formatted = []
        sources = []

        for item in items:
            formatted.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("body", ""),
                    "source": item.get("source", ""),
                    "date": item.get("date", ""),
                    "link": item.get("url", item.get("link", "")),
                }
            )
            sources.append(
                {
                    "type": "news",
                    "title": item.get("title", ""),
                    "url": item.get("url", item.get("link", "")),
                    "source": item.get("source", ""),
                    "snippet": (item.get("body", ""))[:200],
                    "date": item.get("date", ""),
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

    def _make_cache_key(
        self, query: str, location: str, time_range: Optional[str], max_results: int
    ) -> str:
        """Create cache key."""
        raw = f"news:{query}:{location}:{time_range}:{max_results}"
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

        # Simple eviction
        if len(self._cache) > 50:
            oldest_key = min(self._cache_times, key=self._cache_times.get)
            del self._cache[oldest_key]
            del self._cache_times[oldest_key]

    @property
    def metrics(self) -> Dict[str, Any]:
        """Get search metrics."""
        return self._metrics.copy()
