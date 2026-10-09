"""
AmaniQ v2 Tool Executor Node - Fault-Tolerant Parallel Execution
================================================================

This module implements the LangGraph tool_executor node with:
- Parallel tool execution via asyncio.gather
- 5-second hard timeouts per tool
- 3 retries with exponential backoff
- Uses existing kb_search tool with Qdrant/ChromaDB vector store
- Graceful degradation with cached fallbacks
- OpenTelemetry instrumentation for latency tracking

All tools query the CLOUD Qdrant vector store - NO external API calls.

Author: Eng. Onyango Benard
Version: 2.0
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar, Union
from functools import wraps
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

from pydantic import BaseModel, Field
from loguru import logger

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Type variables for generic decorators
T = TypeVar("T")


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================


class ToolConfig:
    """Central configuration for tool execution"""

    # Timeout settings
    DEFAULT_TIMEOUT_SECONDS: float = 5.0
    KB_SEARCH_TIMEOUT_SECONDS: float = 5.0
    CALCULATOR_TIMEOUT_SECONDS: float = 1.0

    # Retry settings
    MAX_RETRIES: int = 3
    INITIAL_BACKOFF_SECONDS: float = 0.5
    MAX_BACKOFF_SECONDS: float = 4.0
    BACKOFF_MULTIPLIER: float = 2.0

    # Cache settings
    CACHE_TTL_HOURS: int = 6
    DEFAULT_CACHE_TTL_SECONDS: int = 3600  # 1 hour

    # Redis key prefixes
    CACHE_KEY_PREFIX: str = "amq:v2:cache:tool"
    LATENCY_KEY_PREFIX: str = "amq:v2:metrics:latency"

    # Vector store namespaces for different tool types
    # These map to the existing Qdrant collections
    NAMESPACES = {
        "search_kenya_law_reports": ["kenya_law"],
        "search_constitution": ["kenya_law"],  # Constitution is in kenya_law namespace
        "search_hansard": ["kenya_parliament"],
        "search_recent_news": ["kenya_news", "global_trends"],
        "lookup_court_calendar": ["kenya_law"],  # Court calendar data in kenya_law
        "kb_search": [
            "kenya_law",
            "kenya_news",
            "kenya_parliament",
            "historical",
            "global_trends",
        ],
    }


class ToolStatus(str, Enum):
    """Execution status for each tool"""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    CACHED = "cached"
    DEGRADED = "degraded"
    RETRYING = "retrying"


# =============================================================================
# PYDANTIC MODELS FOR TOOL RESULTS
# =============================================================================


class ToolExecutionResult(BaseModel):
    """Result from a single tool execution"""

    tool_id: str = Field(..., description="Unique ID for this tool call")
    tool_name: str = Field(..., description="Name of the tool executed")
    query: str = Field(..., description="Query passed to the tool")
    status: ToolStatus = Field(..., description="Execution status")

    # Result data
    result: Optional[Dict[str, Any]] = Field(
        default=None, description="Tool output if successful"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )

    # Metadata
    latency_ms: int = Field(..., description="Execution time in milliseconds")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    from_cache: bool = Field(
        default=False, description="Whether result came from cache"
    )
    cache_age_seconds: Optional[int] = Field(
        default=None, description="Age of cached result in seconds"
    )

    # Degradation info
    degraded: bool = Field(
        default=False, description="Whether this is a degraded/fallback result"
    )
    degradation_reason: Optional[str] = Field(
        default=None, description="Why degraded result was returned"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When this result was produced"
    )


class ToolExecutorOutput(BaseModel):
    """Complete output from tool_executor node"""

    results: Dict[str, ToolExecutionResult] = Field(
        default_factory=dict, description="Map of tool_id to execution result"
    )

    # Aggregate metrics
    total_tools: int = Field(default=0)
    successful_tools: int = Field(default=0)
    failed_tools: int = Field(default=0)
    cached_tools: int = Field(default=0)
    degraded_tools: int = Field(default=0)

    total_latency_ms: int = Field(
        default=0, description="Wall-clock time for all tools"
    )

    # Banners for UI
    banners: List[str] = Field(
        default_factory=list,
        description="Warning banners to show user (e.g., 'Using cached Kenya Law data')",
    )

    # For LangGraph state update
    should_retry: bool = Field(
        default=False, description="Whether supervisor should retry failed tools"
    )
    failed_tool_names: List[str] = Field(
        default_factory=list, description="Names of tools that failed"
    )


# =============================================================================
# CONNECTION POOL MANAGERS (SINGLETONS)
# =============================================================================


@dataclass
class ConnectionPools:
    """
    Singleton manager for connection pools.
    Initialized once, reused across all tool executions.

    NOTE: Since we use kb_search for vector store operations, this class
    is primarily used for Redis caching. The vector store connections are
    managed by the VectorStore class in Module3_NiruDB.
    """

    _instance: Optional["ConnectionPools"] = None
    _redis_client: Any = None
    _initialized: bool = False

    @classmethod
    def get_instance(cls) -> "ConnectionPools":
        """Get or create singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self) -> None:
        """
        Initialize connection pools.
        Must be called once at application startup.

        Creates:
        - Redis async client for caching

        NOTE: Vector store connections are managed by kb_search tool.
        """
        if self._initialized:
            return

        try:
            # Initialize Redis for caching
            import redis.asyncio as aioredis
            import os

            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self._redis_client = aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=50,
            )
            logger.info("Redis connection pool initialized")
        except ImportError:
            logger.warning("redis.asyncio not available, caching disabled")
            self._redis_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize Redis pool: {e}")
            self._redis_client = None

        self._initialized = True

    async def close(self) -> None:
        """Close all connection pools gracefully"""
        if self._redis_client:
            await self._redis_client.close()
        self._initialized = False
        logger.info("Connection pools closed")

    @property
    def redis(self):
        return self._redis_client


# =============================================================================
# OPENTELEMETRY INSTRUMENTATION
# =============================================================================


class TelemetryRecorder:
    """
    Records tool execution metrics to OpenTelemetry.
    Falls back to loguru if OTEL not configured.
    """

    _tracer = None
    _meter = None
    _tool_latency_histogram = None
    _tool_error_counter = None

    @classmethod
    def initialize(cls) -> None:
        """Initialize OpenTelemetry instrumentation"""
        try:
            from opentelemetry import trace, metrics
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.metrics import MeterProvider

            # Get or create tracer
            cls._tracer = trace.get_tracer("amaniq.tool_executor")

            # Get or create meter
            cls._meter = metrics.get_meter("amaniq.tool_executor")

            # Create histogram for latencies
            cls._tool_latency_histogram = cls._meter.create_histogram(
                name="tool_execution_latency_ms",
                description="Tool execution latency in milliseconds",
                unit="ms",
            )

            # Create counter for errors
            cls._tool_error_counter = cls._meter.create_counter(
                name="tool_execution_errors",
                description="Number of tool execution errors",
            )

            logger.info("OpenTelemetry instrumentation initialized")
        except ImportError:
            logger.warning("OpenTelemetry not available, using fallback logging")
        except Exception as e:
            logger.warning(f"Failed to initialize OTEL: {e}")

    @classmethod
    def record_latency(
        cls,
        tool_name: str,
        latency_ms: int,
        status: ToolStatus,
        attributes: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record tool execution latency to OpenTelemetry.

        Args:
            tool_name: Name of the tool
            latency_ms: Execution time in milliseconds
            status: Execution status
            attributes: Additional attributes to record
        """
        attrs = {
            "tool.name": tool_name,
            "tool.status": status.value,
            **(attributes or {}),
        }

        if cls._tool_latency_histogram:
            cls._tool_latency_histogram.record(latency_ms, attrs)

        # Always log for debugging
        logger.debug(
            f"Tool latency: {tool_name} = {latency_ms}ms (status={status.value})"
        )

    @classmethod
    def record_error(
        cls,
        tool_name: str,
        error_type: str,
        attributes: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record tool execution error"""
        attrs = {
            "tool.name": tool_name,
            "error.type": error_type,
            **(attributes or {}),
        }

        if cls._tool_error_counter:
            cls._tool_error_counter.add(1, attrs)

        logger.warning(f"Tool error: {tool_name} - {error_type}")

    @classmethod
    def start_span(cls, name: str, attributes: Optional[Dict[str, str]] = None):
        """Start a tracing span"""
        if cls._tracer:
            return cls._tracer.start_as_current_span(name, attributes=attributes)
        return None


# =============================================================================
# CACHE MANAGER
# =============================================================================


class CacheManager:
    """
    Manages tool result caching with Redis.
    Falls back to in-memory cache if Redis unavailable.
    """

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self._memory_cache: Dict[str, tuple[Dict, datetime]] = {}

    def _make_cache_key(self, tool_name: str, query: str) -> str:
        """Generate cache key from tool name and query"""
        query_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]
        return f"{ToolConfig.CACHE_KEY_PREFIX}:{tool_name}:{query_hash}"

    async def get_cached_result(
        self,
        tool_name: str,
        query: str,
    ) -> Optional[tuple[Dict[str, Any], int]]:
        """
        Get cached tool result if available.

        Args:
            tool_name: Name of the tool
            query: Query string

        Returns:
            Tuple of (cached_result, age_seconds) or None
        """
        cache_key = self._make_cache_key(tool_name, query)

        # Try Redis first
        if self.redis:
            try:
                import json

                cached = await self.redis.get(cache_key)
                if cached:
                    data = json.loads(cached)
                    cached_at = datetime.fromisoformat(data.get("cached_at", ""))
                    age_seconds = int((datetime.utcnow() - cached_at).total_seconds())
                    return data.get("result"), age_seconds
            except Exception as e:
                logger.debug(f"Redis cache read error: {e}")

        # Fallback to memory cache
        if cache_key in self._memory_cache:
            result, cached_at = self._memory_cache[cache_key]
            age_seconds = int((datetime.utcnow() - cached_at).total_seconds())
            max_age = ToolConfig.CACHE_TTL_HOURS * 3600
            if age_seconds <= max_age:
                return result, age_seconds
            else:
                del self._memory_cache[cache_key]

        return None

    async def set_cached_result(
        self,
        tool_name: str,
        query: str,
        result: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Cache a tool result.

        Args:
            tool_name: Name of the tool
            query: Query string
            result: Result to cache
            ttl_seconds: Cache TTL (uses tool-specific default if not provided)
        """
        cache_key = self._make_cache_key(tool_name, query)
        ttl = ttl_seconds or ToolConfig.DEFAULT_CACHE_TTL_SECONDS

        # Try Redis
        if self.redis:
            try:
                import json

                cache_data = {
                    "result": result,
                    "cached_at": datetime.utcnow().isoformat(),
                    "tool_name": tool_name,
                }
                await self.redis.setex(cache_key, ttl, json.dumps(cache_data))
                logger.debug(f"Cached {tool_name} result in Redis for {ttl}s")
                return
            except Exception as e:
                logger.debug(f"Redis cache write error: {e}")

        # Fallback to memory cache
        self._memory_cache[cache_key] = (result, datetime.utcnow())
        logger.debug(f"Cached {tool_name} result in memory")

    async def get_fallback_result(
        self,
        tool_name: str,
        query: str,
    ) -> Optional[tuple[Dict[str, Any], int, str]]:
        """
        Get cached result for fallback (up to configured hours old).

        Returns:
            Tuple of (result, age_seconds, banner_message) or None
        """
        result = await self.get_cached_result(tool_name, query)

        if result:
            cached_data, age_seconds = result
            max_age = ToolConfig.CACHE_TTL_HOURS * 3600

            if age_seconds <= max_age:
                hours_old = age_seconds // 3600
                minutes_old = (age_seconds % 3600) // 60

                if hours_old > 0:
                    age_str = f"{hours_old}h {minutes_old}m"
                else:
                    age_str = f"{minutes_old}m"

                tool_display = tool_name.replace("_", " ").title()
                banner = (
                    f"⚠️ {tool_display} search encountered an issue. "
                    f"Showing cached results from {age_str} ago."
                )
                return cached_data, age_seconds, banner

        return None


# =============================================================================
# LEGAL TOOL IMPLEMENTATIONS (Using kb_search with Qdrant namespaces)
# =============================================================================


class LegalToolsImplementation:
    """
    Implementation of all legal research tools using the kb_search tool
    and Qdrant vector store with namespace filtering.

    All tools query the CLOUD vector store - NO external API calls.

    Architecture:
    ─────────────
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    DATA SOURCE: QDRANT VECTOR STORE                     │
    │                                                                          │
    │   Namespaces:                                                            │
    │   ├── kenya_law        → Case law, Constitution, Court calendar         │
    │   ├── kenya_parliament → Hansard records, Parliamentary debates         │
    │   ├── kenya_news       → News articles                                  │
    │   ├── historical       → Historical legal documents                     │
    │   └── global_trends    → International context                          │
    │                                                                          │
    │   NO EXTERNAL API CALLS - All data from cloud vector store              │
    └─────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, vector_store=None):
        """
        Initialize legal tools with shared vector store.

        Args:
            vector_store: Shared VectorStore instance (imports and creates if None)
        """
        # Lazy import to avoid circular dependencies
        try:
            from Module3_NiruDB.vector_store import VectorStore
            from Module4_NiruAPI.agents.tools.kb_search import KnowledgeBaseSearchTool

            self.vector_store = vector_store or VectorStore()
            self.kb_search = KnowledgeBaseSearchTool(vector_store=self.vector_store)
            logger.info(
                f"LegalToolsImplementation initialized with backend: {getattr(self.vector_store, 'backend', 'unknown')}"
            )
        except ImportError as e:
            logger.warning(f"Could not import vector store modules: {e}")
            self.vector_store = None
            self.kb_search = None

        # Thread pool for running sync operations
        self._executor = ThreadPoolExecutor(max_workers=10)

    def _run_sync_in_executor(self, func: Callable, *args, **kwargs):
        """Run a synchronous function in thread pool executor"""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))

    async def search_kenya_law_reports(
        self, query: str, top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Search Kenya Law Reports for case law, judgments, and legal precedents.

        Searches the 'kenya_law' namespace in Qdrant vector store.

        Args:
            query: Legal search query (e.g., "land rights", "Njoya v AG")
            top_k: Number of results to return

        Returns:
            Dict with cases, citations, and metadata
        """
        if not self.kb_search:
            raise ConnectionError("KnowledgeBaseSearchTool not initialized")

        # Add legal context to query for better embedding match
        enhanced_query = f"Kenya law case judgment: {query}"

        result = await self._run_sync_in_executor(
            self.kb_search.execute,
            query=enhanced_query,
            top_k=top_k,
            namespace=ToolConfig.NAMESPACES["search_kenya_law_reports"],
        )

        # Transform results to legal format
        cases = []
        for item in result.get("search_results", []):
            metadata = item.get("metadata", {})
            cases.append(
                {
                    "title": metadata.get("title", "Untitled Case"),
                    "citation": metadata.get(
                        "citation", metadata.get("source_name", "")
                    ),
                    "court": metadata.get(
                        "court", metadata.get("category", "Unknown Court")
                    ),
                    "year": metadata.get("year", metadata.get("publication_date", "")),
                    "snippet": item.get("content", "")[:500],
                    "relevance_score": item.get("score", 0.0),
                    "source_url": metadata.get("source_url", ""),
                }
            )

        return {
            "query": query,
            "cases": cases,
            "count": len(cases),
            "namespace": "kenya_law",
            "source": "qdrant_vector_store",
        }

    async def search_constitution(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Search the Constitution of Kenya 2010.

        Filters for constitutional provisions within the kenya_law namespace.

        Args:
            query: Constitutional query (e.g., "Article 27", "Bill of Rights")
            top_k: Number of results to return

        Returns:
            Dict with constitutional articles and provisions
        """
        if not self.kb_search:
            raise ConnectionError("KnowledgeBaseSearchTool not initialized")

        # Enhance query with constitutional context
        enhanced_query = f"Constitution of Kenya 2010: {query}"

        result = await self._run_sync_in_executor(
            self.kb_search.execute,
            query=enhanced_query,
            top_k=top_k,
            namespace=ToolConfig.NAMESPACES["search_constitution"],
        )

        # Transform to constitutional format
        articles = []
        import re

        for item in result.get("search_results", []):
            metadata = item.get("metadata", {})
            content = item.get("content", "")

            # Try to extract article number from content or metadata
            article_num = metadata.get("article", "")
            if not article_num and "Article" in content:
                # Simple extraction
                match = re.search(r"Article\s+(\d+)", content)
                if match:
                    article_num = f"Article {match.group(1)}"

            articles.append(
                {
                    "article": article_num
                    or metadata.get("title", "Constitutional Provision"),
                    "chapter": metadata.get("chapter", metadata.get("category", "")),
                    "content": content[:800],
                    "relevance_score": item.get("score", 0.0),
                }
            )

        return {
            "query": query,
            "articles": articles,
            "count": len(articles),
            "namespace": "kenya_law",
            "document": "Constitution of Kenya 2010",
        }

    async def search_hansard(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Search Kenya Parliamentary Hansard records.

        Searches the 'kenya_parliament' namespace for legislative debates.

        Args:
            query: Parliamentary search query
            top_k: Number of results to return

        Returns:
            Dict with debates, statements, and metadata
        """
        if not self.kb_search:
            raise ConnectionError("KnowledgeBaseSearchTool not initialized")

        enhanced_query = f"Kenya Parliament debate Hansard: {query}"

        result = await self._run_sync_in_executor(
            self.kb_search.execute,
            query=enhanced_query,
            top_k=top_k,
            namespace=ToolConfig.NAMESPACES["search_hansard"],
        )

        debates = []
        for item in result.get("search_results", []):
            metadata = item.get("metadata", {})
            debates.append(
                {
                    "date": metadata.get("date", metadata.get("publication_date", "")),
                    "speaker": metadata.get(
                        "speaker", metadata.get("author", "Unknown")
                    ),
                    "house": metadata.get("house", "National Assembly"),
                    "topic": metadata.get("title", metadata.get("topic", "")),
                    "content": item.get("content", "")[:600],
                    "relevance_score": item.get("score", 0.0),
                }
            )

        return {
            "query": query,
            "debates": debates,
            "count": len(debates),
            "namespace": "kenya_parliament",
            "source": "hansard_records",
        }

    async def search_recent_news(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Search recent Kenyan news articles from vector store.

        Searches 'kenya_news' and 'global_trends' namespaces.

        Args:
            query: News search query
            top_k: Number of results to return

        Returns:
            Dict with news articles
        """
        if not self.kb_search:
            raise ConnectionError("KnowledgeBaseSearchTool not initialized")

        enhanced_query = f"Kenya news article: {query}"

        result = await self._run_sync_in_executor(
            self.kb_search.execute,
            query=enhanced_query,
            top_k=top_k,
            namespace=ToolConfig.NAMESPACES["search_recent_news"],
        )

        articles = []
        for item in result.get("search_results", []):
            metadata = item.get("metadata", {})
            articles.append(
                {
                    "title": metadata.get("title", "News Article"),
                    "source": metadata.get(
                        "source_name", metadata.get("source", "Unknown")
                    ),
                    "date": metadata.get("publication_date", metadata.get("date", "")),
                    "snippet": item.get("content", "")[:400],
                    "url": metadata.get("source_url", ""),
                    "category": metadata.get("category", "News"),
                    "relevance_score": item.get("score", 0.0),
                }
            )

        return {
            "query": query,
            "articles": articles,
            "count": len(articles),
            "namespace": ["kenya_news", "global_trends"],
            "source": "news_archive",
        }

    async def lookup_court_calendar(
        self, query: str, top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Look up court schedules and hearing dates.

        Searches for court calendar and case scheduling information.

        Args:
            query: Court calendar query (case number, date, court name)
            top_k: Number of results to return

        Returns:
            Dict with hearing schedules
        """
        if not self.kb_search:
            raise ConnectionError("KnowledgeBaseSearchTool not initialized")

        enhanced_query = f"Kenya court calendar hearing schedule: {query}"

        result = await self._run_sync_in_executor(
            self.kb_search.execute,
            query=enhanced_query,
            top_k=top_k,
            namespace=ToolConfig.NAMESPACES["lookup_court_calendar"],
        )

        hearings = []
        for item in result.get("search_results", []):
            metadata = item.get("metadata", {})
            hearings.append(
                {
                    "case_number": metadata.get(
                        "case_number", metadata.get("title", "")
                    ),
                    "court": metadata.get(
                        "court", metadata.get("category", "High Court")
                    ),
                    "date": metadata.get(
                        "hearing_date", metadata.get("publication_date", "")
                    ),
                    "judge": metadata.get("judge", ""),
                    "parties": metadata.get("parties", ""),
                    "status": metadata.get("status", "Scheduled"),
                    "content": item.get("content", "")[:300],
                    "relevance_score": item.get("score", 0.0),
                }
            )

        return {
            "query": query,
            "hearings": hearings,
            "count": len(hearings),
            "namespace": "kenya_law",
            "source": "court_calendar",
        }

    async def calculate(self, expression: str) -> Dict[str, Any]:
        """
        Perform mathematical calculations (for legal calculations).

        Args:
            expression: Mathematical expression

        Returns:
            Calculation result
        """
        try:
            from Module4_NiruAPI.agents.tools.calculator import CalculatorTool

            calculator = CalculatorTool()
            return await self._run_sync_in_executor(
                calculator.execute, expression=expression
            )
        except ImportError:
            # Fallback to simple eval with safety restrictions
            import ast
            import operator

            # Safe operators
            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
            }

            def safe_eval(node):
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return ops[type(node.op)](
                        safe_eval(node.left), safe_eval(node.right)
                    )
                elif isinstance(node, ast.UnaryOp):
                    return ops[type(node.op)](safe_eval(node.operand))
                else:
                    raise TypeError(f"Unsupported operation: {type(node)}")

            try:
                result = safe_eval(ast.parse(expression, mode="eval").body)
                return {"expression": expression, "result": result}
            except Exception as e:
                return {"expression": expression, "error": str(e)}

    async def general_kb_search(
        self, query: str, top_k: int = 10, namespace: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        General knowledge base search across all namespaces.

        Args:
            query: Search query
            top_k: Number of results
            namespace: Optional specific namespaces to search

        Returns:
            Search results
        """
        if not self.kb_search:
            raise ConnectionError("KnowledgeBaseSearchTool not initialized")

        return await self._run_sync_in_executor(
            self.kb_search.execute,
            query=query,
            top_k=top_k,
            namespace=namespace or ToolConfig.NAMESPACES["kb_search"],
        )

    def get_tool_function(self, tool_name: str) -> Optional[Callable]:
        """Get the async function for a tool name"""
        tool_map = {
            "search_kenya_law_reports": self.search_kenya_law_reports,
            "search_constitution": self.search_constitution,
            "search_hansard": self.search_hansard,
            "search_recent_news": self.search_recent_news,
            "lookup_court_calendar": self.lookup_court_calendar,
            "calculator": self.calculate,
            "kb_search": self.general_kb_search,
        }
        return tool_map.get(tool_name)

    def list_available_tools(self) -> List[str]:
        """List all available tool names"""
        return [
            "search_kenya_law_reports",
            "search_constitution",
            "search_hansard",
            "search_recent_news",
            "lookup_court_calendar",
            "calculator",
            "kb_search",
        ]


# Global tools instance (singleton pattern)
_tools_instance: Optional[LegalToolsImplementation] = None


def get_tools_instance(vector_store=None) -> LegalToolsImplementation:
    """Get or create the global tools instance"""
    global _tools_instance
    if _tools_instance is None:
        _tools_instance = LegalToolsImplementation(vector_store=vector_store)
    return _tools_instance


# =============================================================================
# MAIN TOOL EXECUTOR NODE
# =============================================================================


async def tool_executor_node(
    state: Dict[str, Any],
    *,
    pools: Optional[ConnectionPools] = None,
) -> Dict[str, Any]:
    """
    LangGraph tool_executor node - Executes tools in parallel with fault tolerance.

    This node is the execution engine for the AmaniQ v2 agent. It takes tool calls
    from the Supervisor decision and executes them in parallel with comprehensive
    fault tolerance.

    Architecture:
    ─────────────

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         TOOL EXECUTOR NODE                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Input: state["supervisor_decision"]["tool_plan"]                        │
    │         List[ToolCall] from Supervisor                                   │
    │                                                                          │
    │  ┌────────────────────────────────────────────────────────────────────┐ │
    │  │                    PARALLEL EXECUTION LAYER                        │ │
    │  │                                                                    │ │
    │  │   asyncio.gather(                                                  │ │
    │  │       execute_single_tool(tool_1),  ─┐                             │ │
    │  │       execute_single_tool(tool_2),   │  All run simultaneously    │ │
    │  │       execute_single_tool(tool_3),   │  with individual timeouts   │ │
    │  │       execute_single_tool(tool_4),  ─┘                             │ │
    │  │       return_exceptions=True  # Don't fail fast                    │ │
    │  │   )                                                                │ │
    │  └────────────────────────────────────────────────────────────────────┘ │
    │                              │                                          │
    │                              ▼                                          │
    │  ┌────────────────────────────────────────────────────────────────────┐ │
    │  │                    PER-TOOL ERROR HANDLING                         │ │
    │  │                                                                    │ │
    │  │   For each tool:                                                   │ │
    │  │   1. Check cache first (skip execution if hit)                     │ │
    │  │   2. Apply 5-second hard timeout via asyncio.wait_for              │ │
    │  │   3. On failure: retry up to 3x with exponential backoff           │ │
    │  │      - Backoff: 0.5s → 1s → 2s (capped at 4s)                      │ │
    │  │   4. On final failure: return partial result with error message    │ │
    │  │   5. Kenya Law special case: return 6-hour cache + banner          │ │
    │  └────────────────────────────────────────────────────────────────────┘ │
    │                              │                                          │
    │                              ▼                                          │
    │  ┌────────────────────────────────────────────────────────────────────┐ │
    │  │                    TELEMETRY & CACHING                             │ │
    │  │                                                                    │ │
    │  │   - Record latency to OpenTelemetry histogram                      │ │
    │  │   - Cache successful results to Redis                              │ │
    │  │   - Log all operations via loguru                                  │ │
    │  └────────────────────────────────────────────────────────────────────┘ │
    │                                                                         │
    │  Output: state["tool_results"] = {                                      │
    │      "tool_id_1": ToolExecutionResult,                                  │
    │      "tool_id_2": ToolExecutionResult,                                  │
    │      ...                                                                │
    │  }                                                                      │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    Fault Tolerance Strategy:
    ─────────────────────────

    1. TIMEOUTS (5 seconds per tool)
       - Hard timeout via asyncio.wait_for
       - Prevents any single tool from blocking the pipeline
       - Timeout counts as failure, triggers retry

    2. RETRIES (3 attempts with exponential backoff)
       - Initial backoff: 0.5 seconds
       - Multiplier: 2x per retry
       - Max backoff: 4 seconds
       - Retryable: TimeoutError, ConnectionError, IOError
       - Non-retryable: ValueError, ValidationError (fail immediately)

    3. PARTIAL RESULTS
       - Never fail the entire node due to one tool failure
       - Return what we have + error messages for failures
       - Supervisor can decide whether to retry or proceed

    4. KENYA LAW FALLBACK (Special Case)
       - If Kenya Law Reports times out after all retries:
         a. Check Redis cache for results from last 6 hours
         b. If found: return cached results + banner message
         c. If not found: return error with suggestion to try later
       - Banner format: "⚠️ Kenya Law Reports is currently unavailable.
         Showing cached results from 2h 15m ago."

    5. CONNECTION POOLING
       - HTTPX: 100 max connections, 20 keepalive
       - Qdrant: 10 pooled connections
       - Redis: 50 pooled connections
       - All pools are singleton, initialized once at startup

    6. TELEMETRY
       - Every tool execution logged to OpenTelemetry:
         * tool_execution_latency_ms (histogram)
         * tool_execution_errors (counter)
       - Attributes: tool.name, tool.status, error.type
       - Falls back to loguru if OTEL not configured

    Args:
        state: LangGraph state dictionary containing:
            - supervisor_decision: SupervisorDecision with tool_plan
            - thread_id: Conversation thread ID
            - request_id: Current request ID for tracing

        pools: Optional ConnectionPools instance (uses singleton if not provided)

    Returns:
        Dictionary with state updates:
            - tool_results: Dict[str, ToolExecutionResult]
            - failed_tools: List[str] of tool names that failed
            - banners: List[str] of warning messages for UI
            - total_latency_ms: Wall-clock execution time

    Raises:
        Never raises - all errors are captured in ToolExecutionResult

    Example:
        >>> state = {
        ...     "supervisor_decision": {
        ...         "tool_plan": [
        ...             {"tool_name": "search_kenya_law_reports", "query": "land rights", "priority": 1},
        ...             {"tool_name": "search_constitution", "query": "Article 40 property", "priority": 2},
        ...         ]
        ...     },
        ...     "thread_id": "abc-123",
        ...     "request_id": "req-456",
        ... }
        >>> result = await tool_executor_node(state)
        >>> print(result["tool_results"].keys())
        dict_keys(['tool_0_search_kenya_law_reports', 'tool_1_search_constitution'])

    State Updates:
        - tool_results: Added (Dict[str, ToolExecutionResult])
        - failed_tools: Added (List[str])
        - banners: Added (List[str])
        - current_phase: Updated to "synthesis" or stays "tool_execution"
    """

    # Initialize telemetry
    TelemetryRecorder.initialize()

    # Get tools instance (uses kb_search with Qdrant)
    tools = get_tools_instance()

    # Initialize cache (with optional Redis)
    redis_client = None
    try:
        pools = ConnectionPools.get_instance()
        if pools._initialized:
            redis_client = pools.redis
    except Exception:
        pass
    cache = CacheManager(redis_client)

    # Extract tool calls from state
    supervisor_decision = state.get("supervisor_decision", {})
    tool_plan = supervisor_decision.get("tool_plan", [])

    if not tool_plan:
        logger.warning("tool_executor called with empty tool_plan")
        return {
            "tool_results": {},
            "failed_tools": [],
            "banners": [],
            "total_latency_ms": 0,
            "current_phase": "synthesis",
        }

    logger.info(f"Executing {len(tool_plan)} tools in parallel via kb_search")

    # Track overall timing
    start_time = time.perf_counter()

    # Build execution tasks
    async def execute_single_tool(
        tool_call: Dict[str, Any],
        tool_id: str,
    ) -> ToolExecutionResult:
        """Execute a single tool with full error handling"""

        tool_name = tool_call.get("tool_name", "unknown")
        query = tool_call.get("query", "")
        tool_start = time.perf_counter()

        # Check cache first
        cached = await cache.get_cached_result(tool_name, query)
        if cached:
            cached_result, age_seconds = cached
            latency_ms = int((time.perf_counter() - tool_start) * 1000)

            TelemetryRecorder.record_latency(tool_name, latency_ms, ToolStatus.CACHED)

            return ToolExecutionResult(
                tool_id=tool_id,
                tool_name=tool_name,
                query=query,
                status=ToolStatus.CACHED,
                result=cached_result,
                latency_ms=latency_ms,
                from_cache=True,
                cache_age_seconds=age_seconds,
            )

        # Get the tool function from our LegalToolsImplementation
        tool_func = tools.get_tool_function(tool_name)
        if tool_func is None:
            latency_ms = int((time.perf_counter() - tool_start) * 1000)
            return ToolExecutionResult(
                tool_id=tool_id,
                tool_name=tool_name,
                query=query,
                status=ToolStatus.ERROR,
                error_message=f"Unknown tool: {tool_name}. Available: {tools.list_available_tools()}",
                latency_ms=latency_ms,
            )

        # Execute with timeout and retry
        try:
            # Wrap in timeout
            async def run_with_timeout_and_retry():
                """Run tool with timeout, retry on failure"""
                last_exception = None
                backoff = ToolConfig.INITIAL_BACKOFF_SECONDS

                for attempt in range(ToolConfig.MAX_RETRIES + 1):
                    try:
                        return (
                            await asyncio.wait_for(
                                tool_func(query),
                                timeout=ToolConfig.DEFAULT_TIMEOUT_SECONDS,
                            ),
                            attempt,
                        )
                    except asyncio.TimeoutError as e:
                        last_exception = e
                        if attempt < ToolConfig.MAX_RETRIES:
                            logger.warning(
                                f"Retry {attempt + 1}/{ToolConfig.MAX_RETRIES} for {tool_name}: timeout"
                            )
                            await asyncio.sleep(backoff)
                            backoff = min(
                                backoff * ToolConfig.BACKOFF_MULTIPLIER,
                                ToolConfig.MAX_BACKOFF_SECONDS,
                            )
                    except (ConnectionError, IOError) as e:
                        last_exception = e
                        if attempt < ToolConfig.MAX_RETRIES:
                            logger.warning(
                                f"Retry {attempt + 1}/{ToolConfig.MAX_RETRIES} for {tool_name}: {e}"
                            )
                            await asyncio.sleep(backoff)
                            backoff = min(
                                backoff * ToolConfig.BACKOFF_MULTIPLIER,
                                ToolConfig.MAX_BACKOFF_SECONDS,
                            )

                raise last_exception or asyncio.TimeoutError("Max retries exceeded")

            result, retry_count = await run_with_timeout_and_retry()
            latency_ms = int((time.perf_counter() - tool_start) * 1000)

            # Cache successful result
            await cache.set_cached_result(tool_name, query, result)

            TelemetryRecorder.record_latency(tool_name, latency_ms, ToolStatus.SUCCESS)

            return ToolExecutionResult(
                tool_id=tool_id,
                tool_name=tool_name,
                query=query,
                status=ToolStatus.SUCCESS,
                result=result,
                latency_ms=latency_ms,
                retry_count=retry_count,
            )

        except asyncio.TimeoutError as e:
            latency_ms = int((time.perf_counter() - tool_start) * 1000)

            # Try fallback cache for any tool
            fallback = await cache.get_fallback_result(tool_name, query)
            if fallback:
                cached_result, age_seconds, banner = fallback

                TelemetryRecorder.record_latency(
                    tool_name, latency_ms, ToolStatus.DEGRADED
                )

                return ToolExecutionResult(
                    tool_id=tool_id,
                    tool_name=tool_name,
                    query=query,
                    status=ToolStatus.DEGRADED,
                    result=cached_result,
                    latency_ms=latency_ms,
                    from_cache=True,
                    cache_age_seconds=age_seconds,
                    degraded=True,
                    degradation_reason=banner,
                )

            TelemetryRecorder.record_latency(tool_name, latency_ms, ToolStatus.TIMEOUT)
            TelemetryRecorder.record_error(tool_name, "timeout")

            return ToolExecutionResult(
                tool_id=tool_id,
                tool_name=tool_name,
                query=query,
                status=ToolStatus.TIMEOUT,
                error_message=f"Tool timed out after {ToolConfig.DEFAULT_TIMEOUT_SECONDS}s",
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - tool_start) * 1000)

            TelemetryRecorder.record_latency(tool_name, latency_ms, ToolStatus.ERROR)
            TelemetryRecorder.record_error(tool_name, type(e).__name__)

            return ToolExecutionResult(
                tool_id=tool_id,
                tool_name=tool_name,
                query=query,
                status=ToolStatus.ERROR,
                error_message=str(e),
                latency_ms=latency_ms,
            )

    # Execute all tools in parallel
    tasks = [
        execute_single_tool(
            tool_call, f"tool_{i}_{tool_call.get('tool_name', 'unknown')}"
        )
        for i, tool_call in enumerate(tool_plan)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    tool_results: Dict[str, ToolExecutionResult] = {}
    failed_tools: List[str] = []
    banners: List[str] = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Unexpected exception in gather
            tool_call = tool_plan[i]
            tool_id = f"tool_{i}_{tool_call.get('tool_name', 'unknown')}"
            tool_results[tool_id] = ToolExecutionResult(
                tool_id=tool_id,
                tool_name=tool_call.get("tool_name", "unknown"),
                query=tool_call.get("query", ""),
                status=ToolStatus.ERROR,
                error_message=f"Unexpected error: {result}",
                latency_ms=0,
            )
            failed_tools.append(tool_call.get("tool_name", "unknown"))
        elif isinstance(result, ToolExecutionResult):
            tool_results[result.tool_id] = result

            if result.status in (ToolStatus.ERROR, ToolStatus.TIMEOUT):
                failed_tools.append(result.tool_name)

            if result.degraded and result.degradation_reason:
                banners.append(result.degradation_reason)

    # Calculate total latency
    total_latency_ms = int((time.perf_counter() - start_time) * 1000)

    # Log summary
    success_count = sum(
        1 for r in tool_results.values() if r.status == ToolStatus.SUCCESS
    )
    cached_count = sum(
        1 for r in tool_results.values() if r.status == ToolStatus.CACHED
    )

    logger.info(
        f"Tool executor completed: {success_count} success, {cached_count} cached, "
        f"{len(failed_tools)} failed, {total_latency_ms}ms total"
    )

    # Return state updates
    return {
        "tool_results": {k: v.model_dump() for k, v in tool_results.items()},
        "failed_tools": failed_tools,
        "banners": banners,
        "total_latency_ms": total_latency_ms,
        "current_phase": "synthesis" if not failed_tools else "tool_execution",
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "ToolConfig",
    "ToolStatus",
    # Models
    "ToolExecutionResult",
    "ToolExecutorOutput",
    # Connection management (kept for Redis caching)
    "ConnectionPools",
    # Telemetry
    "TelemetryRecorder",
    # Cache
    "CacheManager",
    # Tool implementations (kb_search based)
    "LegalToolsImplementation",
    "get_tools_instance",
    # Main node
    "tool_executor_node",
]
