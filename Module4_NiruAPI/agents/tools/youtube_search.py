"""
YouTube Search Tool - Enhanced Video Search with SerpAPI

Features:
- SerpAPI YouTube Search integration
- Direct YouTube Data API fallback
- Async/await support
- Video transcript extraction (optional)
- Channel and playlist search
- Retry logic with exponential backoff
- Result caching
- LLM-ready tool schema
"""

import os
import re
import asyncio
import time
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from loguru import logger

# SerpAPI import
try:
    from serpapi import GoogleSearch

    SERPAPI_AVAILABLE = True
except ImportError:
    GoogleSearch = None
    SERPAPI_AVAILABLE = False
    logger.warning(
        "google-search-results not installed. Install with: pip install google-search-results"
    )

# HTTP client for direct API calls
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class YouTubeSearchConfig:
    """Configuration for YouTube search tool."""

    max_retries: int = 3
    base_retry_delay: float = 1.0
    timeout: float = 30.0
    cache_enabled: bool = True
    cache_ttl: float = 3600.0  # 1 hour for video results
    default_max_results: int = 10
    # API keys
    serpapi_key: str = field(default_factory=lambda: os.getenv("SERPAPI_API_KEY", ""))
    youtube_api_key: str = field(
        default_factory=lambda: os.getenv("YOUTUBE_API_KEY", "")
    )


class YouTubeSearchTool:
    """
    Enhanced YouTube search tool with SerpAPI and direct API support.

    Features:
    - SerpAPI YouTube Search (primary)
    - YouTube Data API v3 (fallback)
    - Video, Channel, and Playlist search
    - Duration and view count filtering
    - Transcript extraction (upcoming)
    """

    name = "youtube_search"
    description = (
        "Search YouTube for videos, channels, and playlists. "
        "Returns video details including title, description, duration, views, channel. "
        "Best for: tutorial lookups, educational content, news videos, entertainment."
    )

    # Duration filters
    DURATION_FILTERS = {
        "short": "PT4M",  # Under 4 minutes
        "medium": "PT20M",  # 4-20 minutes
        "long": "PT20M+",  # Over 20 minutes
    }

    # Sort options
    SORT_OPTIONS = {
        "relevance": "relevance",
        "date": "date",
        "views": "viewCount",
        "rating": "rating",
    }

    def __init__(self, config: Optional[YouTubeSearchConfig] = None):
        """Initialize YouTube search tool."""
        self.config = config or YouTubeSearchConfig()

        # Rate limiting
        self._last_request_time = 0.0

        # Cache
        self._cache: Dict[str, Dict] = {}
        self._cache_times: Dict[str, float] = {}

        # Metrics
        self._metrics = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "cache_hits": 0,
            "serpapi_calls": 0,
            "youtube_api_calls": 0,
            "retries": 0,
        }

        providers = []
        if self.config.serpapi_key and SERPAPI_AVAILABLE:
            providers.append("serpapi")
        if self.config.youtube_api_key and HTTPX_AVAILABLE:
            providers.append("youtube_api")

        logger.info(f"YouTubeSearchTool initialized (providers: {providers})")

    def execute(
        self,
        query: str,
        max_results: int = 10,
        search_type: str = "video",
        duration: Optional[str] = None,
        sort_by: str = "relevance",
        published_after: Optional[str] = None,
        channel_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Synchronous YouTube search (backwards compatible).

        Args:
            query: Search query
            max_results: Maximum number of results (1-50)
            search_type: 'video', 'channel', or 'playlist'
            duration: 'short', 'medium', or 'long'
            sort_by: 'relevance', 'date', 'views', 'rating'
            published_after: ISO date string (e.g., '2024-01-01')
            channel_id: Filter by specific channel

        Returns:
            YouTube search results with video details and sources
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.aexecute(
                            query,
                            max_results,
                            search_type,
                            duration,
                            sort_by,
                            published_after,
                            channel_id,
                        ),
                    )
                    return future.result(timeout=self.config.timeout + 5)
            else:
                return loop.run_until_complete(
                    self.aexecute(
                        query,
                        max_results,
                        search_type,
                        duration,
                        sort_by,
                        published_after,
                        channel_id,
                    )
                )
        except Exception as e:
            logger.error(f"Sync execute error: {e}")
            return self._error_response(query, str(e))

    async def aexecute(
        self,
        query: str,
        max_results: int = 10,
        search_type: str = "video",
        duration: Optional[str] = None,
        sort_by: str = "relevance",
        published_after: Optional[str] = None,
        channel_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async YouTube search with fallback providers."""
        if not query or not query.strip():
            return self._error_response(query, "Empty query provided")

        max_results = max(1, min(max_results, 50))
        self._metrics["total_searches"] += 1
        start_time = time.time()

        # Check cache
        cache_key = self._make_cache_key(
            query, search_type, max_results, duration, sort_by
        )
        if self.config.cache_enabled:
            cached = self._get_from_cache(cache_key)
            if cached:
                self._metrics["cache_hits"] += 1
                cached["metadata"]["cached"] = True
                return cached

        # Try providers in order
        providers = [
            ("serpapi", self._search_serpapi),
            ("youtube_api", self._search_youtube_api),
        ]

        last_error = None
        for provider_name, provider_func in providers:
            try:
                results = await self._search_with_retry(
                    provider_func,
                    query=query,
                    max_results=max_results,
                    search_type=search_type,
                    duration=duration,
                    sort_by=sort_by,
                    published_after=published_after,
                    channel_id=channel_id,
                )

                if results and results.get("results"):
                    results["metadata"] = {
                        "provider": provider_name,
                        "search_type": search_type,
                        "latency_ms": (time.time() - start_time) * 1000,
                        "cached": False,
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    if self.config.cache_enabled:
                        self._set_cache(cache_key, results)

                    self._metrics["successful_searches"] += 1
                    return results

            except Exception as e:
                last_error = e
                logger.warning(f"YouTube provider {provider_name} failed: {e}")
                continue

        self._metrics["failed_searches"] += 1
        return self._error_response(
            query,
            str(last_error) if last_error else "All YouTube search providers failed",
        )

    async def _search_with_retry(self, search_func, **kwargs) -> Dict[str, Any]:
        """Execute search with retry logic."""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                # Rate limiting
                await self._rate_limit()

                return await asyncio.wait_for(
                    search_func(**kwargs), timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"YouTube search timed out")
            except Exception as e:
                last_error = e

            if attempt < self.config.max_retries - 1:
                self._metrics["retries"] += 1
                delay = self.config.base_retry_delay * (2**attempt)
                await asyncio.sleep(delay)

        raise last_error or Exception("YouTube search failed")

    async def _search_serpapi(
        self,
        query: str,
        max_results: int,
        search_type: str,
        duration: Optional[str],
        sort_by: str,
        published_after: Optional[str],
        channel_id: Optional[str],
    ) -> Dict[str, Any]:
        """Search using SerpAPI YouTube Search."""
        if not self.config.serpapi_key or not SERPAPI_AVAILABLE:
            raise RuntimeError("SerpAPI not available")

        loop = asyncio.get_event_loop()
        self._metrics["serpapi_calls"] += 1

        def _do_search():
            # SerpAPI YouTube search uses "youtube" engine
            params = {
                "engine": "youtube",
                "search_query": query,
                "api_key": self.config.serpapi_key,
            }

            # Add filters
            if duration and duration in self.DURATION_FILTERS:
                params["sp"] = self._get_duration_filter(duration)

            search = GoogleSearch(params)
            return search.get_dict()

        raw_results = await loop.run_in_executor(None, _do_search)

        return self._format_serpapi_results(
            query, raw_results, search_type, max_results
        )

    async def _search_youtube_api(
        self,
        query: str,
        max_results: int,
        search_type: str,
        duration: Optional[str],
        sort_by: str,
        published_after: Optional[str],
        channel_id: Optional[str],
    ) -> Dict[str, Any]:
        """Search using YouTube Data API v3 (fallback)."""
        if not self.config.youtube_api_key or not HTTPX_AVAILABLE:
            raise RuntimeError("YouTube API not available")

        self._metrics["youtube_api_calls"] += 1

        params = {
            "part": "snippet",
            "q": query,
            "key": self.config.youtube_api_key,
            "maxResults": max_results,
            "type": search_type,
            "order": self.SORT_OPTIONS.get(sort_by, "relevance"),
        }

        if duration and duration in self.DURATION_FILTERS:
            params["videoDuration"] = duration

        if published_after:
            params["publishedAfter"] = f"{published_after}T00:00:00Z"

        if channel_id:
            params["channelId"] = channel_id

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.googleapis.com/youtube/v3/search",
                params=params,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()

        return self._format_youtube_api_results(query, data)

    def _format_serpapi_results(
        self, query: str, raw_results: Dict, search_type: str, max_results: int
    ) -> Dict[str, Any]:
        """Format SerpAPI YouTube results."""
        formatted = []
        sources = []

        # Get video results from SerpAPI response
        video_results = raw_results.get("video_results", [])

        for item in video_results[:max_results]:
            video_data = {
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "link": item.get("link", ""),
                "video_id": self._extract_video_id(item.get("link", "")),
                "thumbnail": item.get("thumbnail", {}).get("static", ""),
                "duration": item.get("length", {}).get("text", ""),
                "views": item.get("views", 0),
                "views_text": self._format_views(item.get("views", 0)),
                "published": item.get("published_date", ""),
                "channel": {
                    "name": item.get("channel", {}).get("name", ""),
                    "link": item.get("channel", {}).get("link", ""),
                    "verified": item.get("channel", {}).get("verified", False),
                },
            }
            formatted.append(video_data)

            sources.append(
                {
                    "type": "youtube",
                    "title": video_data["title"],
                    "url": video_data["link"],
                    "channel": video_data["channel"]["name"],
                    "snippet": (
                        (video_data["description"])[:200]
                        if video_data["description"]
                        else ""
                    ),
                    "duration": video_data["duration"],
                    "views": video_data["views_text"],
                }
            )

        return {
            "query": query,
            "results": formatted,
            "sources": sources,
            "count": len(formatted),
        }

    def _format_youtube_api_results(self, query: str, data: Dict) -> Dict[str, Any]:
        """Format YouTube Data API results."""
        formatted = []
        sources = []

        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            video_id = item.get("id", {}).get("videoId", "")

            video_data = {
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "link": (
                    f"https://www.youtube.com/watch?v={video_id}" if video_id else ""
                ),
                "video_id": video_id,
                "thumbnail": snippet.get("thumbnails", {})
                .get("high", {})
                .get("url", ""),
                "published": snippet.get("publishedAt", ""),
                "channel": {
                    "name": snippet.get("channelTitle", ""),
                    "id": snippet.get("channelId", ""),
                },
            }
            formatted.append(video_data)

            sources.append(
                {
                    "type": "youtube",
                    "title": video_data["title"],
                    "url": video_data["link"],
                    "channel": video_data["channel"]["name"],
                    "snippet": (
                        (video_data["description"])[:200]
                        if video_data["description"]
                        else ""
                    ),
                }
            )

        return {
            "query": query,
            "results": formatted,
            "sources": sources,
            "count": len(formatted),
        }

    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        patterns = [
            r"(?:v=|/)([0-9A-Za-z_-]{11}).*",
            r"(?:embed/)([0-9A-Za-z_-]{11})",
            r"(?:youtu\.be/)([0-9A-Za-z_-]{11})",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return ""

    def _format_views(self, views: Any) -> str:
        """Format view count for display."""
        try:
            views = int(views)
            if views >= 1_000_000_000:
                return f"{views / 1_000_000_000:.1f}B views"
            elif views >= 1_000_000:
                return f"{views / 1_000_000:.1f}M views"
            elif views >= 1_000:
                return f"{views / 1_000:.1f}K views"
            else:
                return f"{views} views"
        except (ValueError, TypeError):
            return str(views) if views else ""

    def _get_duration_filter(self, duration: str) -> str:
        """Get SerpAPI duration filter parameter."""
        # SerpAPI uses 'sp' parameter for filters
        # These are base64 encoded filter values
        filters = {
            "short": "EgIYAQ%3D%3D",  # Under 4 minutes
            "medium": "EgIYAw%3D%3D",  # 4-20 minutes
            "long": "EgIYAg%3D%3D",  # Over 20 minutes
        }
        return filters.get(duration, "")

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

        min_delay = 0.5  # Minimum 500ms between requests
        if time_since_last < min_delay:
            await asyncio.sleep(min_delay - time_since_last)

        self._last_request_time = time.time()

    def _make_cache_key(
        self,
        query: str,
        search_type: str,
        max_results: int,
        duration: Optional[str],
        sort_by: str,
    ) -> str:
        """Create cache key."""
        raw = f"youtube:{query}:{search_type}:{max_results}:{duration}:{sort_by}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """Get from cache if not expired."""
        if key not in self._cache:
            return None

        if time.time() - self._cache_times.get(key, 0) > self.config.cache_ttl:
            del self._cache[key]
            del self._cache_times[key]
            return None

        return self._cache[key].copy()

    def _set_cache(self, key: str, value: Dict):
        """Set cache entry."""
        self._cache[key] = value.copy()
        self._cache_times[key] = time.time()

        # Simple eviction
        if len(self._cache) > 100:
            oldest_key = min(self._cache_times, key=self._cache_times.get)
            del self._cache[oldest_key]
            del self._cache_times[oldest_key]

    def get_tool_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM function calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "YouTube search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results (1-50)",
                        "default": 10,
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["video", "channel", "playlist"],
                        "description": "Type of content to search",
                        "default": "video",
                    },
                    "duration": {
                        "type": "string",
                        "enum": ["short", "medium", "long"],
                        "description": "Video duration filter",
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["relevance", "date", "views", "rating"],
                        "description": "Sort order",
                        "default": "relevance",
                    },
                },
                "required": ["query"],
            },
        }

    async def get_video_details(self, video_id: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific video.

        Args:
            video_id: YouTube video ID

        Returns:
            Video details including statistics
        """
        if not self.config.youtube_api_key or not HTTPX_AVAILABLE:
            return {"error": "YouTube API not available"}

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={
                    "part": "snippet,statistics,contentDetails",
                    "id": video_id,
                    "key": self.config.youtube_api_key,
                },
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()

        if not data.get("items"):
            return {"error": "Video not found"}

        item = data["items"][0]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        content = item.get("contentDetails", {})

        return {
            "video_id": video_id,
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "channel": snippet.get("channelTitle", ""),
            "channel_id": snippet.get("channelId", ""),
            "published": snippet.get("publishedAt", ""),
            "duration": content.get("duration", ""),
            "views": int(stats.get("viewCount", 0)),
            "likes": int(stats.get("likeCount", 0)),
            "comments": int(stats.get("commentCount", 0)),
            "thumbnail": snippet.get("thumbnails", {}).get("maxres", {}).get("url", ""),
            "tags": snippet.get("tags", []),
            "category_id": snippet.get("categoryId", ""),
        }

    @property
    def metrics(self) -> Dict[str, Any]:
        """Get search metrics."""
        return self._metrics.copy()
