"""
AmaniQ v2 Speculative Pre-Fetch & OpenTelemetry Tracing
=======================================================

Performance optimization through:
1. Speculative pre-fetch for legal queries (race-condition safe)
2. OpenTelemetry instrumentation for latency debugging
3. Token usage tracking per turn
4. Cache hit rate monitoring

Trigger Words for Pre-Fetch:
- "section", "article", "act", "cap", "constitution"
- "penal code", "finance act", "court of appeal", "high court", "judgment"

Author: Eng. Onyango Benard
Version: 2.0
"""

import asyncio
import time
import re
from typing import Dict, Any, List, Optional, Set, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from contextlib import asynccontextmanager
from functools import wraps
from loguru import logger

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


# =============================================================================
# CONFIGURATION
# =============================================================================

# Trigger words/phrases for speculative pre-fetch (case-insensitive)
LEGAL_TRIGGER_PATTERNS = [
    r"\bsection\s+\d+",  # "section 35"
    r"\barticle\s+\d+",  # "article 27"
    r"\bact\b",  # "act"
    r"\bcap\s*\d+",  # "cap 63" or "cap63"
    r"\bconstitution\b",  # "constitution"
    r"\bpenal\s+code\b",  # "penal code"
    r"\bfinance\s+act\b",  # "finance act"
    r"\bcourt\s+of\s+appeal\b",  # "court of appeal"
    r"\bhigh\s+court\b",  # "high court"
    r"\bjudgment\b",  # "judgment"
    r"\bjudgement\b",  # British spelling
    r"\bstatute\b",  # "statute"
    r"\blaw\s+reports?\b",  # "law report(s)"
    r"\beklr\b",  # "[2022] eKLR"
    r"\bklr\b",  # "KLR"
    r"\blegal\b",  # "legal"
    r"\bcourt\b",  # "court"
    r"\bcriminal\b",  # "criminal"
    r"\bcivil\b",  # "civil"
]

# Pre-compiled regex for efficiency
TRIGGER_REGEX = re.compile("|".join(LEGAL_TRIGGER_PATTERNS), re.IGNORECASE)


@dataclass
class PrefetchConfig:
    """Configuration for speculative pre-fetch"""

    enabled: bool = True
    timeout_seconds: float = 3.0  # Max time to wait for prefetch
    cancel_grace_period: float = 0.1  # Grace period before cancellation
    max_concurrent_prefetches: int = 3
    namespaces: List[str] = field(
        default_factory=lambda: ["kenya_law", "kenya_parliament"]
    )


# =============================================================================
# OPENTELEMETRY SETUP
# =============================================================================


class TelemetryMetrics:
    """OpenTelemetry metrics and tracing"""

    _tracer = None
    _meter = None
    _initialized = False

    # Histograms
    _node_latency = None
    _tool_latency = None
    _token_usage = None

    # Counters
    _cache_hits = None
    _cache_misses = None
    _prefetch_hits = None
    _prefetch_cancels = None
    _tool_timeouts = None

    @classmethod
    def initialize(cls) -> bool:
        """Initialize OpenTelemetry instrumentation"""
        if cls._initialized:
            return True

        try:
            from opentelemetry import trace, metrics
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
            import os

            # Check if OTEL endpoint is configured
            otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")

            if otlp_endpoint:
                # Production: Export to OTLP collector
                trace_provider = TracerProvider()
                trace_provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
                )
                trace.set_tracer_provider(trace_provider)

                metric_reader = PeriodicExportingMetricReader(
                    OTLPMetricExporter(endpoint=otlp_endpoint),
                    export_interval_millis=60000,
                )
                meter_provider = MeterProvider(metric_readers=[metric_reader])
                metrics.set_meter_provider(meter_provider)

            # Get tracer and meter
            cls._tracer = trace.get_tracer("amaniq.agents", "2.0.0")
            cls._meter = metrics.get_meter("amaniq.agents", "2.0.0")

            # Create histograms
            cls._node_latency = cls._meter.create_histogram(
                name="amaniq.node.latency",
                description="Latency of each LangGraph node in milliseconds",
                unit="ms",
            )

            cls._tool_latency = cls._meter.create_histogram(
                name="amaniq.tool.latency",
                description="Latency of each tool execution in milliseconds",
                unit="ms",
            )

            cls._token_usage = cls._meter.create_histogram(
                name="amaniq.tokens.usage",
                description="Token usage per turn",
                unit="tokens",
            )

            # Create counters
            cls._cache_hits = cls._meter.create_counter(
                name="amaniq.cache.hits",
                description="Number of cache hits",
            )

            cls._cache_misses = cls._meter.create_counter(
                name="amaniq.cache.misses",
                description="Number of cache misses",
            )

            cls._prefetch_hits = cls._meter.create_counter(
                name="amaniq.prefetch.hits",
                description="Number of prefetch results used",
            )

            cls._prefetch_cancels = cls._meter.create_counter(
                name="amaniq.prefetch.cancels",
                description="Number of prefetches cancelled",
            )

            cls._tool_timeouts = cls._meter.create_counter(
                name="amaniq.tool.timeouts",
                description="Number of tool timeouts",
            )

            cls._initialized = True
            logger.info("OpenTelemetry instrumentation initialized")
            return True

        except ImportError as e:
            logger.warning(f"OpenTelemetry not available: {e}")
            cls._initialized = True  # Mark as initialized to avoid retrying
            return False
        except Exception as e:
            logger.warning(f"OpenTelemetry setup failed: {e}")
            cls._initialized = True
            return False

    @classmethod
    @asynccontextmanager
    async def trace_node(
        cls, node_name: str, attributes: Optional[Dict[str, str]] = None
    ):
        """
        Context manager to trace a LangGraph node.

        Usage:
            async with TelemetryMetrics.trace_node("supervisor") as span:
                result = await process()
                span.set_attribute("tokens", 150)
        """
        cls.initialize()
        start = time.time()
        span = None

        try:
            if cls._tracer:
                from opentelemetry import trace

                span = cls._tracer.start_span(
                    f"node.{node_name}",
                    attributes={"node.name": node_name, **(attributes or {})},
                )
                span.__enter__()

            yield span

        except Exception as e:
            if span:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
            raise

        finally:
            latency_ms = (time.time() - start) * 1000

            # Record metrics
            if cls._node_latency:
                cls._node_latency.record(latency_ms, {"node": node_name})

            if span:
                span.set_attribute("latency_ms", latency_ms)
                span.__exit__(None, None, None)

            logger.debug(f"Node {node_name}: {latency_ms:.1f}ms")

    @classmethod
    @asynccontextmanager
    async def trace_tool(cls, tool_name: str, query: str):
        """Context manager to trace a tool execution"""
        cls.initialize()
        start = time.time()
        span = None

        try:
            if cls._tracer:
                from opentelemetry import trace

                span = cls._tracer.start_span(
                    f"tool.{tool_name}",
                    attributes={
                        "tool.name": tool_name,
                        "tool.query": query[:100],
                    },
                )
                span.__enter__()

            yield span

        except asyncio.TimeoutError:
            if cls._tool_timeouts:
                cls._tool_timeouts.add(1, {"tool": tool_name})
            if span:
                from opentelemetry import trace

                span.set_status(trace.Status(trace.StatusCode.ERROR, "timeout"))
            raise

        except Exception as e:
            if span:
                from opentelemetry import trace

                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
            raise

        finally:
            latency_ms = (time.time() - start) * 1000

            if cls._tool_latency:
                cls._tool_latency.record(latency_ms, {"tool": tool_name})

            if span:
                span.set_attribute("latency_ms", latency_ms)
                span.__exit__(None, None, None)

    @classmethod
    def record_tokens(
        cls, prompt_tokens: int, completion_tokens: int, node: str = "unknown"
    ):
        """Record token usage"""
        cls.initialize()

        total = prompt_tokens + completion_tokens

        if cls._token_usage:
            cls._token_usage.record(prompt_tokens, {"type": "prompt", "node": node})
            cls._token_usage.record(
                completion_tokens, {"type": "completion", "node": node}
            )

        logger.debug(
            f"Tokens ({node}): {prompt_tokens} prompt + {completion_tokens} completion = {total}"
        )

    @classmethod
    def record_cache_hit(cls, cache_type: str = "answer"):
        """Record cache hit"""
        cls.initialize()
        if cls._cache_hits:
            cls._cache_hits.add(1, {"type": cache_type})

    @classmethod
    def record_cache_miss(cls, cache_type: str = "answer"):
        """Record cache miss"""
        cls.initialize()
        if cls._cache_misses:
            cls._cache_misses.add(1, {"type": cache_type})

    @classmethod
    def record_prefetch_hit(cls):
        """Record prefetch result was used"""
        cls.initialize()
        if cls._prefetch_hits:
            cls._prefetch_hits.add(1)

    @classmethod
    def record_prefetch_cancel(cls):
        """Record prefetch was cancelled"""
        cls.initialize()
        if cls._prefetch_cancels:
            cls._prefetch_cancels.add(1)


# =============================================================================
# SPECULATIVE PRE-FETCH MANAGER
# =============================================================================


class PrefetchResult:
    """Container for prefetch results with cancellation support"""

    def __init__(self, query: str, namespaces: List[str]):
        self.query = query
        self.namespaces = namespaces
        self.results: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.task: Optional[asyncio.Task] = None
        self.completed = asyncio.Event()
        self.cancelled = False
        self.start_time = time.time()

    @property
    def latency_ms(self) -> float:
        return (time.time() - self.start_time) * 1000

    def cancel(self):
        """Cancel the prefetch"""
        self.cancelled = True
        if self.task and not self.task.done():
            self.task.cancel()
        TelemetryMetrics.record_prefetch_cancel()


class SpeculativePrefetcher:
    """
    Manages speculative pre-fetch for legal queries.

    Race-Condition Safety:
    ─────────────────────
    1. Prefetch runs in separate asyncio.Task
    2. Results stored in thread-safe dict with unique request_id
    3. Supervisor decision checked BEFORE consuming results
    4. Automatic cancellation if Supervisor doesn't need the results
    5. Grace period allows in-flight requests to complete

    Flow:
    ─────
    User Message
         │
         ├──► [Trigger Detection] ──► Start Prefetch Task ──► Store in pending_prefetches
         │                                    │
         ▼                                    │
    Supervisor Node                           │
         │                                    │
         ├── Decision: LEGAL_RESEARCH? ◄──────┘
         │         │
         │    YES  │  NO
         │    ▼    ▼
         │  Use    Cancel
         │  Results Prefetch
         │
         ▼
    Tool Executor (uses prefetch OR fresh search)
    """

    def __init__(self, config: Optional[PrefetchConfig] = None):
        self.config = config or PrefetchConfig()
        self._pending: Dict[str, PrefetchResult] = {}
        self._search_fn: Optional[Callable[[str, List[str]], Awaitable[Dict]]] = None
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_prefetches)

    def set_search_function(self, fn: Callable[[str, List[str]], Awaitable[Dict]]):
        """Set the search function to use for prefetch"""
        self._search_fn = fn

    def should_prefetch(self, message: str) -> bool:
        """Check if message contains legal trigger words"""
        if not self.config.enabled:
            return False
        return bool(TRIGGER_REGEX.search(message))

    def extract_search_query(self, message: str) -> str:
        """Extract the best search query from user message"""
        # Remove common filler words but keep legal terms
        query = message.strip()

        # If message is short, use as-is
        if len(query.split()) <= 10:
            return query

        # For longer messages, try to extract the legal part
        # Look for patterns like "What does Article 27 say about..."
        patterns = [
            r"(article\s+\d+[^.?!]*)",
            r"(section\s+\d+[^.?!]*)",
            r"((?:penal|finance|employment|land)\s+(?:code|act)[^.?!]*)",
            r"(constitution[^.?!]*(?:rights?|article|section)[^.?!]*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: use first 15 words
        words = query.split()[:15]
        return " ".join(words)

    async def start_prefetch(
        self, request_id: str, message: str
    ) -> Optional[PrefetchResult]:
        """
        Start speculative prefetch for a message.

        Args:
            request_id: Unique ID for this request (for race-condition safety)
            message: User's message

        Returns:
            PrefetchResult handle or None if prefetch not started
        """
        if not self.should_prefetch(message):
            return None

        if not self._search_fn:
            logger.warning("Prefetch search function not set")
            return None

        query = self.extract_search_query(message)

        async with self._lock:
            # Check if we already have a prefetch for this request
            if request_id in self._pending:
                return self._pending[request_id]

            # Create prefetch result container
            prefetch = PrefetchResult(query, self.config.namespaces)
            self._pending[request_id] = prefetch

        # Start prefetch task
        prefetch.task = asyncio.create_task(
            self._execute_prefetch(request_id, prefetch)
        )

        logger.debug(f"Started prefetch for: {query[:50]}... (request_id={request_id})")
        return prefetch

    async def _execute_prefetch(self, request_id: str, prefetch: PrefetchResult):
        """Execute the actual prefetch search"""
        async with self._semaphore:
            if prefetch.cancelled:
                return

            try:
                async with TelemetryMetrics.trace_tool(
                    "prefetch_kb_search", prefetch.query
                ):
                    result = await asyncio.wait_for(
                        self._search_fn(prefetch.query, prefetch.namespaces),
                        timeout=self.config.timeout_seconds,
                    )

                    if not prefetch.cancelled:
                        prefetch.results = result
                        logger.debug(
                            f"Prefetch complete: {len(result.get('search_results', []))} results "
                            f"in {prefetch.latency_ms:.1f}ms"
                        )

            except asyncio.TimeoutError:
                prefetch.error = "timeout"
                logger.warning(f"Prefetch timeout for: {prefetch.query[:30]}...")

            except asyncio.CancelledError:
                prefetch.error = "cancelled"

            except Exception as e:
                prefetch.error = str(e)
                logger.warning(f"Prefetch error: {e}")

            finally:
                prefetch.completed.set()

    async def get_results(
        self, request_id: str, supervisor_needs_search: bool, timeout: float = 0.5
    ) -> Optional[Dict[str, Any]]:
        """
        Get prefetch results if available and needed.

        Args:
            request_id: Request ID to look up
            supervisor_needs_search: Whether Supervisor decided to use search
            timeout: Max time to wait for pending prefetch

        Returns:
            Prefetch results or None
        """
        async with self._lock:
            prefetch = self._pending.get(request_id)

        if not prefetch:
            return None

        # If Supervisor doesn't need search, cancel prefetch
        if not supervisor_needs_search:
            prefetch.cancel()
            await self._cleanup(request_id)
            return None

        # Wait for prefetch to complete (with short timeout)
        try:
            await asyncio.wait_for(prefetch.completed.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # Prefetch taking too long, don't use it
            logger.debug(f"Prefetch not ready in time for {request_id}")
            return None

        if prefetch.cancelled or prefetch.error:
            await self._cleanup(request_id)
            return None

        # Use prefetch results!
        TelemetryMetrics.record_prefetch_hit()
        logger.info(f"Using prefetch results (saved ~{prefetch.latency_ms:.0f}ms)")

        results = prefetch.results
        await self._cleanup(request_id)

        return results

    async def cancel_prefetch(self, request_id: str):
        """Cancel a pending prefetch"""
        async with self._lock:
            prefetch = self._pending.get(request_id)

        if prefetch:
            prefetch.cancel()
            await asyncio.sleep(self.config.cancel_grace_period)
            await self._cleanup(request_id)

    async def _cleanup(self, request_id: str):
        """Remove prefetch from pending dict"""
        async with self._lock:
            self._pending.pop(request_id, None)


# =============================================================================
# INSTRUMENTED LANGGRAPH NODES
# =============================================================================


def traced_node(node_name: str):
    """
    Decorator to add tracing to a LangGraph node.

    Usage:
        @traced_node("supervisor")
        async def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
            ...
    """

    def decorator(fn):
        @wraps(fn)
        async def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            async with TelemetryMetrics.trace_node(node_name) as span:
                result = await fn(state)

                # Record token usage if available
                if "token_usage" in result:
                    usage = result["token_usage"]
                    TelemetryMetrics.record_tokens(
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                        node_name,
                    )
                    if span:
                        span.set_attribute(
                            "tokens.prompt", usage.get("prompt_tokens", 0)
                        )
                        span.set_attribute(
                            "tokens.completion", usage.get("completion_tokens", 0)
                        )

                return result

        return wrapper

    return decorator


# =============================================================================
# PREFETCH-ENABLED GRAPH MIDDLEWARE
# =============================================================================


class PrefetchMiddleware:
    """
    Middleware to integrate speculative prefetch into LangGraph.

    Usage:
        middleware = PrefetchMiddleware()

        # At start of conversation turn
        request_id = str(uuid4())
        await middleware.on_message_received(request_id, user_message)

        # After supervisor decision
        prefetch_results = await middleware.get_prefetch_if_needed(
            request_id,
            supervisor_decision
        )

        # Use prefetch_results in tool_executor if available
    """

    def __init__(self):
        self._prefetcher = SpeculativePrefetcher()
        self._initialized = False

    async def initialize(self, search_fn: Callable[[str, List[str]], Awaitable[Dict]]):
        """Initialize with search function"""
        self._prefetcher.set_search_function(search_fn)
        self._initialized = True

    async def on_message_received(self, request_id: str, message: str) -> bool:
        """
        Called when user message is received.
        Starts prefetch if message contains legal triggers.

        Args:
            request_id: Unique request ID
            message: User's message

        Returns:
            Whether prefetch was started
        """
        if not self._initialized:
            return False

        prefetch = await self._prefetcher.start_prefetch(request_id, message)
        return prefetch is not None

    async def get_prefetch_if_needed(
        self, request_id: str, supervisor_decision: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get prefetch results if Supervisor needs them.

        Args:
            request_id: Request ID
            supervisor_decision: Supervisor's routing decision

        Returns:
            Prefetch results or None
        """
        intent = supervisor_decision.get("intent", "")
        needs_search = intent in ("LEGAL_RESEARCH", "NEWS_SUMMARY")

        # Check if any tool in the plan uses kb_search
        tool_plan = supervisor_decision.get("tool_plan", [])
        uses_kb_search = any(
            t.get("tool_name") == "kb_search" or t.get("tool") == "kb_search"
            for t in tool_plan
        )

        needs_search = needs_search or uses_kb_search

        return await self._prefetcher.get_results(request_id, needs_search)

    async def cancel(self, request_id: str):
        """Cancel prefetch for a request"""
        await self._prefetcher.cancel_prefetch(request_id)


# =============================================================================
# METRICS DASHBOARD DATA
# =============================================================================


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics for monitoring"""

    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Latencies (p50, p95, p99)
    node_latencies: Dict[str, Dict[str, float]] = field(default_factory=dict)
    tool_latencies: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Rates
    cache_hit_rate: float = 0.0
    prefetch_hit_rate: float = 0.0
    tool_timeout_rate: float = 0.0

    # Token usage
    avg_tokens_per_turn: float = 0.0
    total_tokens_last_hour: int = 0


class MetricsCollector:
    """
    Collects metrics for dashboard display.

    Usage:
        collector = MetricsCollector()
        snapshot = await collector.get_snapshot()
    """

    def __init__(self):
        # In-memory metrics for when OTEL isn't available
        self._node_latencies: Dict[str, List[float]] = {}
        self._tool_latencies: Dict[str, List[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._prefetch_hits = 0
        self._prefetch_starts = 0
        self._tool_timeouts = 0
        self._tool_calls = 0
        self._tokens: List[int] = []

    def record_node_latency(self, node: str, latency_ms: float):
        """Record node latency"""
        if node not in self._node_latencies:
            self._node_latencies[node] = []
        self._node_latencies[node].append(latency_ms)
        # Keep last 1000 samples
        if len(self._node_latencies[node]) > 1000:
            self._node_latencies[node] = self._node_latencies[node][-1000:]

    def record_tool_latency(self, tool: str, latency_ms: float):
        """Record tool latency"""
        if tool not in self._tool_latencies:
            self._tool_latencies[tool] = []
        self._tool_latencies[tool].append(latency_ms)
        if len(self._tool_latencies[tool]) > 1000:
            self._tool_latencies[tool] = self._tool_latencies[tool][-1000:]

    def record_cache_result(self, hit: bool):
        """Record cache hit/miss"""
        if hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

    def record_prefetch(self, started: bool, used: bool = False):
        """Record prefetch stats"""
        if started:
            self._prefetch_starts += 1
        if used:
            self._prefetch_hits += 1

    def record_tool_timeout(self):
        """Record tool timeout"""
        self._tool_timeouts += 1
        self._tool_calls += 1

    def record_tool_success(self):
        """Record successful tool call"""
        self._tool_calls += 1

    def record_tokens(self, count: int):
        """Record token usage"""
        self._tokens.append(count)
        if len(self._tokens) > 1000:
            self._tokens = self._tokens[-1000:]

    def _percentile(self, data: List[float], p: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    async def get_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot"""
        snapshot = PerformanceSnapshot()

        # Calculate node latencies
        for node, latencies in self._node_latencies.items():
            snapshot.node_latencies[node] = {
                "p50": self._percentile(latencies, 50),
                "p95": self._percentile(latencies, 95),
                "p99": self._percentile(latencies, 99),
            }

        # Calculate tool latencies
        for tool, latencies in self._tool_latencies.items():
            snapshot.tool_latencies[tool] = {
                "p50": self._percentile(latencies, 50),
                "p95": self._percentile(latencies, 95),
                "p99": self._percentile(latencies, 99),
            }

        # Calculate rates
        total_cache = self._cache_hits + self._cache_misses
        snapshot.cache_hit_rate = (
            self._cache_hits / total_cache if total_cache > 0 else 0.0
        )

        snapshot.prefetch_hit_rate = (
            self._prefetch_hits / self._prefetch_starts
            if self._prefetch_starts > 0
            else 0.0
        )

        snapshot.tool_timeout_rate = (
            self._tool_timeouts / self._tool_calls if self._tool_calls > 0 else 0.0
        )

        # Token usage
        if self._tokens:
            snapshot.avg_tokens_per_turn = sum(self._tokens) / len(self._tokens)
            snapshot.total_tokens_last_hour = sum(self._tokens[-100:])  # Approx

        return snapshot


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# =============================================================================
# EXAMPLE INTEGRATION
# =============================================================================

"""
COMPLETE INTEGRATION EXAMPLE
============================

from uuid import uuid4
from Module4_NiruAPI.agents.prefetch import (
    PrefetchMiddleware,
    TelemetryMetrics,
    traced_node,
    get_metrics_collector,
)

# Initialize prefetch middleware
prefetch_middleware = PrefetchMiddleware()

async def init_prefetch():
    # Set up the search function
    from Module4_NiruAPI.agents.optimization import CachedKBSearch
    cached_search = CachedKBSearch()
    await cached_search.initialize()
    
    async def search_fn(query: str, namespaces: List[str]) -> Dict[str, Any]:
        return await cached_search.search(query, namespace=namespaces)
    
    await prefetch_middleware.initialize(search_fn)


# Instrumented supervisor node
@traced_node("supervisor")
async def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # ... supervisor logic ...
    return {
        **state,
        "supervisor_decision": decision,
        "token_usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
    }


# Instrumented tool executor that uses prefetch
@traced_node("tool_executor")
async def tool_executor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    request_id = state.get("request_id")
    supervisor_decision = state.get("supervisor_decision", {})
    
    # Try to get prefetch results
    prefetch_results = await prefetch_middleware.get_prefetch_if_needed(
        request_id,
        supervisor_decision
    )
    
    # If we have prefetch results for kb_search, use them
    tool_plan = supervisor_decision.get("tool_plan", [])
    results = []
    
    for tool_call in tool_plan:
        tool_name = tool_call.get("tool_name")
        
        if tool_name == "kb_search" and prefetch_results:
            # Use prefetch results
            results.append({
                "tool_name": tool_name,
                "status": "success",
                "data": prefetch_results,
                "from_prefetch": True,
            })
            get_metrics_collector().record_prefetch(started=True, used=True)
        else:
            # Execute tool normally
            async with TelemetryMetrics.trace_tool(tool_name, tool_call.get("query", "")):
                result = await execute_tool(tool_call)
                results.append(result)
    
    return {**state, "tool_results": results}


# API endpoint with prefetch
@app.post("/chat")
async def chat(request: ChatRequest):
    request_id = str(uuid4())
    
    # Start prefetch immediately (non-blocking)
    prefetch_started = await prefetch_middleware.on_message_received(
        request_id,
        request.message
    )
    
    # Process through LangGraph (supervisor runs in parallel with prefetch)
    result = await graph.ainvoke({
        "request_id": request_id,
        "messages": [{"role": "user", "content": request.message}],
        # ... other state
    })
    
    return result


# Metrics endpoint for dashboard
@app.get("/metrics")
async def metrics():
    collector = get_metrics_collector()
    snapshot = await collector.get_snapshot()
    
    return {
        "node_latencies": snapshot.node_latencies,
        "tool_latencies": snapshot.tool_latencies,
        "cache_hit_rate": f"{snapshot.cache_hit_rate:.1%}",
        "prefetch_hit_rate": f"{snapshot.prefetch_hit_rate:.1%}",
        "tool_timeout_rate": f"{snapshot.tool_timeout_rate:.1%}",
        "avg_tokens_per_turn": snapshot.avg_tokens_per_turn,
    }
"""


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Config
    "PrefetchConfig",
    "LEGAL_TRIGGER_PATTERNS",
    "TRIGGER_REGEX",
    # Telemetry
    "TelemetryMetrics",
    # Prefetch
    "PrefetchResult",
    "SpeculativePrefetcher",
    "PrefetchMiddleware",
    # Decorators
    "traced_node",
    # Metrics
    "PerformanceSnapshot",
    "MetricsCollector",
    "get_metrics_collector",
]
