"""
Streaming Tool Executor
=======================
Executes tools with real-time SSE event emission so the frontend
can display tool progress as it happens (like Gemini's "Searching..." cards).

Event types emitted:
  - tool_start:   Tool execution begins
  - tool_progress: Intermediate progress update
  - tool_result:  Tool completed successfully
  - tool_error:   Tool failed
  - tool_cached:  Tool returned cached result
"""

import asyncio
import time
import json
from typing import Dict, Any, AsyncGenerator, List, Optional
from dataclasses import dataclass, asdict, field
from loguru import logger

from .tool_registry import ToolRegistry
from .tool_schema import get_tool_schema


@dataclass
class ToolEvent:
    """SSE event emitted during tool execution lifecycle."""

    event: str  # tool_start | tool_progress | tool_result | tool_error | tool_cached
    tool_name: str
    query: str
    data: Any = None
    latency_ms: float = 0.0
    error: Optional[str] = None
    request_id: str = ""


def _build_args(tool_name: str, query: str) -> Dict[str, Any]:
    """Build tool arguments based on tool type."""
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


async def stream_execute_tool(
    tool_name: str,
    query: str,
    registry: ToolRegistry,
    request_id: str = "",
    timeout: float = 8.0,
) -> AsyncGenerator[str, None]:
    """
    Execute a single tool and yield SSE-formatted events throughout its lifecycle.

    Yields JSON strings suitable for SSE `data:` transmission.
    """
    start = time.time()
    base_event = {"request_id": request_id, "tool_name": tool_name, "query": query}

    # tool_start
    yield _sse_event(
        "tool_start",
        {
            **base_event,
            "description": _get_tool_description(tool_name, registry),
        },
    )

    # Check tool exists
    if tool_name not in registry.list_tools():
        yield _sse_event(
            "tool_error",
            {
                **base_event,
                "error": f"Unknown tool: {tool_name}",
                "latency_ms": (time.time() - start) * 1000,
            },
        )
        return

    # Execute
    try:
        args = _build_args(tool_name, query)
        tool = registry.get_tool(tool_name)

        if hasattr(tool, "aexecute"):
            result = await asyncio.wait_for(
                tool.aexecute(**args),
                timeout=timeout,
            )
        elif hasattr(tool, "invoke"):
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: tool.invoke(args)
                ),
                timeout=timeout,
            )
        else:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: registry.execute_tool(tool_name, args)
                ),
                timeout=timeout,
            )

        elapsed = (time.time() - start) * 1000
        yield _sse_event(
            "tool_result",
            {
                **base_event,
                "data": _summarize_result(result),
                "latency_ms": elapsed,
            },
        )

    except asyncio.TimeoutError:
        elapsed = (time.time() - start) * 1000
        yield _sse_event(
            "tool_error",
            {
                **base_event,
                "error": f"Timeout after {timeout}s",
                "latency_ms": elapsed,
            },
        )
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        yield _sse_event(
            "tool_error",
            {
                **base_event,
                "error": str(e),
                "latency_ms": elapsed,
            },
        )


async def stream_execute_parallel(
    tool_calls: List[Dict[str, Any]],
    registry: ToolRegistry,
    request_id: str = "",
    max_parallel: int = 4,
    timeout: float = 8.0,
) -> AsyncGenerator[str, None]:
    """
    Execute multiple tools in parallel, yielding events for each.

    Matches the parallel execution pattern of Gemini/Claude tool use.
    """
    semaphore = asyncio.Semaphore(max_parallel)

    async def _run_one(tc: Dict[str, Any]):
        tool_name = tc.get("tool_name", tc.get("tool", ""))
        query = tc.get("query", "")
        async for event in stream_execute_tool(
            tool_name, query, registry, request_id, timeout
        ):
            yield event

    tasks = [_run_one(tc) for tc in tool_calls[:max_parallel]]
    for coro in asyncio.as_completed(tasks):
        async for event in await coro:
            yield event


def _sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """Format as SSE data line."""
    payload = {"event": event_type, **data}
    return f"data: {json.dumps(payload, default=str)}\n\n"


def _summarize_result(result: Any) -> Any:
    """Summarize tool result to avoid sending massive payloads over SSE."""
    if isinstance(result, dict):
        return {
            k: v for k, v in result.items() if not isinstance(v, (bytes, bytearray))
        }
    if isinstance(result, list) and len(result) > 10:
        return {"count": len(result), "preview": result[:3]}
    return result


def _get_tool_description(tool_name: str, registry: ToolRegistry) -> str:
    """Get a human-readable description for a tool."""
    tool = registry.get_tool(tool_name)
    if tool is None:
        return tool_name
    try:
        return get_tool_schema(tool, tool_name).description
    except Exception:
        return tool_name


__all__ = [
    "ToolEvent",
    "stream_execute_tool",
    "stream_execute_parallel",
]
