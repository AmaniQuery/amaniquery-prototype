"""
AmaniQ v2 Tool Executor Node - Fault-Tolerant Parallel Execution
================================================================

Uses the ToolRegistry from tool_registry.py directly.
Available tools: kb_search, web_search, news_search, calculator,
                 url_fetch, youtube_search, twitter_search

Author: Eng. Onyango Benard
Version: 2.0
"""

import asyncio
import hashlib
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from collections import OrderedDict
from loguru import logger

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from Module4_NiruAPI.agents.tools.tool_registry import ToolRegistry

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class ToolExecutorConfig:
    """Configuration for tool execution"""

    default_timeout: float = 8.0
    max_retries: int = 3
    retry_base_delay: float = 0.5
    cache_ttl: int = 300
    cache_max_size: int = 1000
    max_parallel_tools: int = 4


class ToolStatus(str, Enum):
    """Tool execution status"""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    CACHED = "cached"


@dataclass
class ToolResult:
    """Result from a single tool execution"""

    tool_name: str
    query: str
    status: ToolStatus
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    cached: bool = False
    retry_count: int = 0


# =============================================================================
# SIMPLE IN-MEMORY CACHE
# =============================================================================


class CacheManager:
    """Simple in-memory cache with TTL"""

    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()

    def _make_key(self, tool_name: str, query: str) -> str:
        content = f"{tool_name}:{query.lower().strip()}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, tool_name: str, query: str) -> Optional[Any]:
        key = self._make_key(tool_name, query)
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl:
                self._cache.move_to_end(key)
                return value
            del self._cache[key]
        return None

    def set(self, tool_name: str, query: str, value: Any) -> None:
        key = self._make_key(tool_name, query)
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[key] = (value, time.time())


# =============================================================================
# TOOL EXECUTOR
# =============================================================================


class ToolExecutor:
    """Executes tools from ToolRegistry with fault tolerance"""

    def __init__(
        self,
        config: Optional[ToolExecutorConfig] = None,
        registry: Optional[ToolRegistry] = None,
    ):
        self.config = config or ToolExecutorConfig()
        self.cache = CacheManager(self.config.cache_max_size, self.config.cache_ttl)
        self._registry = registry  # Use provided registry (from global startup)

    @property
    def registry(self) -> ToolRegistry:
        if self._registry is None:
            # Fallback: create new registry if none provided (shouldn't happen in production)
            logger.warning(
                "Creating new ToolRegistry - this should be passed from global startup!"
            )
            self._registry = ToolRegistry()
            logger.info(f"ToolRegistry initialized: {self._registry.list_tools()}")
        return self._registry

    def _build_args(self, tool_name: str, query: str) -> Dict[str, Any]:
        """Build tool arguments based on tool type"""
        if tool_name == "kb_search":
            return {"query": query, "top_k": 5}
        elif tool_name == "web_search":
            return {"query": query, "max_results": 5}
        elif tool_name == "news_search":
            return {"query": query, "max_results": 5}
        elif tool_name == "twitter_search":
            return {"query": query, "max_results": 10}
        elif tool_name == "youtube_search":
            return {"query": query, "max_results": 5}
        elif tool_name == "url_fetch":
            return {"url": query}
        elif tool_name == "calculator":
            return {"expression": query}
        elif tool_name == "file_write":
            return {"filename": f"research_{int(time.time())}.md", "content": query}
        elif tool_name == "email_draft":
            return {"to": "", "subject": query[:80], "body": query}
        elif tool_name == "bill_status":
            return {"bill_name": query}
        elif tool_name == "hansard":
            return {"topic": query}
        elif tool_name == "fees_calculator":
            return {"query": query}
        elif tool_name == "legal_citation":
            return {"query": query}
        return {"query": query}

    async def _execute_one(self, tool_name: str, query: str) -> ToolResult:
        """Execute single tool with timeout and retries"""
        start = time.time()

        # Check cache
        cached = self.cache.get(tool_name, query)
        if cached is not None:
            return ToolResult(
                tool_name=tool_name,
                query=query,
                status=ToolStatus.CACHED,
                data=cached,
                cached=True,
                latency_ms=(time.time() - start) * 1000,
            )

        # Check tool exists
        if tool_name not in self.registry.list_tools():
            return ToolResult(
                tool_name=tool_name,
                query=query,
                status=ToolStatus.ERROR,
                error=f"Unknown tool: {tool_name}. Available: {self.registry.list_tools()}",
                latency_ms=(time.time() - start) * 1000,
            )

        # Execute with retries
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                args = self._build_args(tool_name, query)

                # Call async method directly if available, otherwise use sync method in executor
                tool = self.registry.get_tool(tool_name)
                if hasattr(tool, "aexecute"):
                    # Direct async call - preferred for async tools like kb_search
                    result = await asyncio.wait_for(
                        tool.aexecute(**args), timeout=self.config.default_timeout
                    )
                else:
                    # Fall back to sync method in thread executor
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, lambda: self.registry.execute_tool(tool_name, args)
                        ),
                        timeout=self.config.default_timeout,
                    )

                self.cache.set(tool_name, query, result)
                return ToolResult(
                    tool_name=tool_name,
                    query=query,
                    status=ToolStatus.SUCCESS,
                    data=result,
                    latency_ms=(time.time() - start) * 1000,
                    retry_count=attempt,
                )
            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.config.default_timeout}s"
                logger.warning(f"{tool_name} timeout (attempt {attempt + 1})")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"{tool_name} error (attempt {attempt + 1}): {e}")

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_base_delay * (2**attempt))

        return ToolResult(
            tool_name=tool_name,
            query=query,
            status=(
                ToolStatus.TIMEOUT
                if "Timeout" in (last_error or "")
                else ToolStatus.ERROR
            ),
            error=last_error,
            latency_ms=(time.time() - start) * 1000,
            retry_count=self.config.max_retries,
        )

    async def execute_parallel(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[ToolResult]:
        """Execute tools in parallel"""
        if not tool_calls:
            return []

        tool_calls = tool_calls[: self.config.max_parallel_tools]

        tasks = [
            self._execute_one(
                tc.get("tool_name", tc.get("tool", "")), tc.get("query", "")
            )
            for tc in tool_calls
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [
            (
                r
                if isinstance(r, ToolResult)
                else ToolResult(
                    tool_name=tool_calls[i].get("tool_name", "unknown"),
                    query=tool_calls[i].get("query", ""),
                    status=ToolStatus.ERROR,
                    error=str(r),
                )
            )
            for i, r in enumerate(results)
        ]


# =============================================================================
# LANGGRAPH NODE
# =============================================================================

_executor: Optional[ToolExecutor] = None


def get_executor() -> ToolExecutor:
    """Get or create tool executor with global registry"""
    global _executor
    if _executor is None:
        # Get global tool registry from API startup
        try:
            from Module4_NiruAPI.api import tool_registry as global_registry

            if global_registry:
                _executor = ToolExecutor(registry=global_registry)
                logger.info("ToolExecutor using global registry (fast path)")
            else:
                _executor = ToolExecutor()
                logger.warning("Global registry not available, creating new one")
        except ImportError:
            _executor = ToolExecutor()
            logger.warning("Could not import global registry, creating new one")
    return _executor


async def tool_executor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for tool execution.

    Reads tool_plan from supervisor, executes in parallel, returns results.
    """
    executor = get_executor()

    supervisor_decision = state.get("supervisor_decision", {})
    tool_plan = supervisor_decision.get("tool_plan", [])

    if not tool_plan:
        logger.info("No tools to execute")
        return {**state, "tool_results": [], "tool_execution_status": "no_tools"}

    logger.info(f"Executing {len(tool_plan)} tools in parallel")

    # Convert Pydantic models to dicts if needed
    tool_calls = []
    for tc in tool_plan:
        if isinstance(tc, dict):
            tool_calls.append(tc)
        else:
            tool_calls.append(
                {
                    "tool_name": (
                        tc.tool_name.value
                        if hasattr(tc.tool_name, "value")
                        else str(tc.tool_name)
                    ),
                    "query": tc.query,
                    "priority": getattr(tc, "priority", 2),
                }
            )

    start = time.time()
    results = await executor.execute_parallel(tool_calls)
    total_ms = (time.time() - start) * 1000

    serialized = [
        {
            "tool_name": r.tool_name,
            "query": r.query,
            "status": r.status.value,
            "data": r.data,
            "error": r.error,
            "latency_ms": r.latency_ms,
            "cached": r.cached,
            "retry_count": r.retry_count,
        }
        for r in results
    ]

    success_count = sum(
        1 for r in results if r.status in (ToolStatus.SUCCESS, ToolStatus.CACHED)
    )
    logger.info(
        f"Tools done: {success_count}/{len(results)} success, {total_ms:.0f}ms total"
    )

    return {
        **state,
        "tool_results": serialized,
        "tool_execution_status": "complete",
        "tool_execution_latency_ms": total_ms,
        "tool_success_rate": success_count / len(results) if results else 0,
    }


__all__ = [
    "ToolExecutorConfig",
    "ToolStatus",
    "ToolResult",
    "CacheManager",
    "ToolExecutor",
    "tool_executor_node",
    "get_executor",
]
