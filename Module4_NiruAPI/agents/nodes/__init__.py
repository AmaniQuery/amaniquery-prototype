"""
AmaniQ v2 LangGraph Nodes
=========================

This package contains all LangGraph nodes for the AmaniQ v2 agent.

Nodes:
- tool_executor: Fault-tolerant parallel tool execution using ToolRegistry
- clarification: Human-in-the-loop clarification sub-graph

Uses actual tools from tool_registry.py: kb_search, web_search, news_search,
calculator, url_fetch, youtube_search, twitter_search

Author: Eng. Onyango Benard
Version: 2.0
"""

from .tool_executor import (
    ToolExecutorConfig,
    ToolStatus,
    ToolResult,
    CacheManager,
    ToolExecutor,
    tool_executor_node,
    get_executor,
)

from .clarification import (
    # State
    ClarificationState,
    ClarificationStatus,
    MAX_CLARIFICATION_ROUNDS,
    # Node functions
    clarification_entry_node,
    clarification_resume_node,
    max_clarification_node,
    # Routing functions
    should_clarify,
    clarification_route,
    after_clarification_route,
    # Helpers
    build_clarification_question,
    add_clarification_subgraph,
)

__all__ = [
    # Tool executor
    "ToolExecutorConfig",
    "ToolStatus",
    "ToolResult",
    "CacheManager",
    "ToolExecutor",
    "tool_executor_node",
    "get_executor",
    # Clarification
    "ClarificationState",
    "ClarificationStatus",
    "MAX_CLARIFICATION_ROUNDS",
    "clarification_entry_node",
    "clarification_resume_node",
    "max_clarification_node",
    "should_clarify",
    "clarification_route",
    "after_clarification_route",
    "build_clarification_question",
    "add_clarification_subgraph",
]
