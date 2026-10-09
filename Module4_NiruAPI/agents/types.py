"""
AmaniQ Agent Type Definitions
=============================

Canonical type definitions for the entire agent system.
All agent components should import types from this file to ensure consistency.

Author: Eng. Onyango Benard
Version: 2.1.0
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal, TypedDict, Annotated, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator

# LangGraph imports
try:
    from langgraph.graph.message import add_messages
    from langchain_core.messages import (
        BaseMessage,
        HumanMessage,
        AIMessage,
        SystemMessage,
    )

    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback for environments without LangGraph
    add_messages = lambda x: x
    BaseMessage = Any
    HumanMessage = Any
    AIMessage = Any
    SystemMessage = Any
    LANGGRAPH_AVAILABLE = False


# =============================================================================
# ENUMS
# =============================================================================


class IntentType(str, Enum):
    """Exact intent classifications for supervisor routing"""

    LEGAL_RESEARCH = "LEGAL_RESEARCH"
    NEWS_SUMMARY = "NEWS_SUMMARY"
    GENERAL_CHAT = "GENERAL_CHAT"
    CLARIFY = "CLARIFY"
    ESCALATE = "ESCALATE"
    CACHED = "CACHED"  # Response from cache


class ToolName(str, Enum):
    """Registered tool names - supervisor cannot hallucinate others"""

    KB_SEARCH = "kb_search"
    WEB_SEARCH = "web_search"
    NEWS_SEARCH = "news_search"
    CALCULATOR = "calculator"
    URL_FETCH = "url_fetch"
    YOUTUBE_SEARCH = "youtube_search"
    TWITTER_SEARCH = "twitter_search"
    FILE_WRITE = "file_write"
    EMAIL_DRAFT = "email_draft"
    BILL_STATUS = "bill_status"
    HANSARD = "hansard"
    FEES_CALCULATOR = "fees_calculator"
    LEGAL_CITATION = "legal_citation"


class ClarificationStatus(str, Enum):
    """Status of clarification process"""

    NEEDED = "needed"
    WAITING = "waiting"
    RECEIVED = "received"
    RESOLVED = "resolved"
    MAX_REACHED = "max_reached"


class ToolStatus(str, Enum):
    """Tool execution status"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


# =============================================================================
# PYDANTIC MODELS FOR SUPERVISOR
# =============================================================================


class ToolCall(BaseModel):
    """Single tool invocation in the parallel plan"""

    tool_name: ToolName = Field(..., description="Exact tool name from allowed list")
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Optimized search query for this specific tool",
    )
    priority: Literal[1, 2, 3] = Field(
        default=2,
        description="Execution priority: 1=critical, 2=important, 3=supplementary",
    )

    @field_validator("tool_name", mode="before")
    @classmethod
    def validate_tool_name(cls, v):
        """Ensure tool name is exactly one of the allowed values"""
        if isinstance(v, str):
            normalized = v.lower().strip()
            valid_tools = [t.value for t in ToolName]
            if normalized not in valid_tools:
                raise ValueError(
                    f"Invalid tool name '{v}'. Must be one of: {valid_tools}"
                )
            return normalized
        return v


class ClarificationRequest(BaseModel):
    """Structured clarification when more info needed"""

    missing_fields: List[str] = Field(
        ..., min_length=1, description="List of specific missing information"
    )
    clarification_question: str = Field(
        ...,
        min_length=10,
        max_length=300,
        description="Clear, specific question to ask the user",
    )
    partial_understanding: str = Field(
        ..., max_length=200, description="What was understood so far from the query"
    )


class SupervisorDecision(BaseModel):
    """
    The ONLY output format for the Supervisor node.
    Strict JSON validation - any deviation will be rejected.
    """

    # Required fields
    intent: IntentType = Field(
        ...,
        description="Exactly one of: LEGAL_RESEARCH, NEWS_SUMMARY, GENERAL_CHAT, CLARIFY, ESCALATE",
    )

    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0"
    )

    reasoning: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Brief chain-of-thought explanation",
    )

    # NEW: Multi-hop indicator for ReAct routing
    requires_multi_hop: bool = Field(
        default=False,
        description="True if query requires multiple sequential tool calls (not parallel). "
        "Examples: 'Did bill X pass AND what does it say?' requires status check then content search.",
    )

    # Conditional fields based on intent
    tool_plan: Optional[List[ToolCall]] = Field(
        default=None, description="Required if intent is LEGAL_RESEARCH or NEWS_SUMMARY"
    )

    clarification: Optional[ClarificationRequest] = Field(
        default=None, description="Required if intent is CLARIFY"
    )

    direct_response: Optional[str] = Field(
        default=None, max_length=1000, description="Required if intent is GENERAL_CHAT"
    )

    escalation_reason: Optional[str] = Field(
        default=None, max_length=200, description="Required if intent is ESCALATE"
    )

    # Metadata
    detected_language: Literal["en", "sw", "mixed"] = Field(
        default="en", description="Detected query language"
    )

    detected_entities: List[str] = Field(
        default_factory=list,
        description="Extracted legal entities: case names, statute refs, dates",
    )

    token_count: int = Field(
        default=0, ge=0, description="Estimated token count of the input context"
    )

    context_overflow: bool = Field(
        default=False,
        description="True if context exceeds limit - must refuse processing",
    )

    @model_validator(mode="after")
    def validate_conditional_fields(self):
        """Ensure required fields are present based on intent"""
        if self.context_overflow and self.intent != IntentType.ESCALATE:
            raise ValueError("Context overflow requires ESCALATE intent")

        if self.intent in (IntentType.LEGAL_RESEARCH, IntentType.NEWS_SUMMARY):
            if not self.tool_plan:
                raise ValueError(f"{self.intent} intent requires tool_plan")

        if self.intent == IntentType.CLARIFY:
            if not self.clarification:
                raise ValueError("CLARIFY intent requires clarification details")

        if self.intent == IntentType.GENERAL_CHAT:
            if not self.direct_response:
                raise ValueError("GENERAL_CHAT intent requires direct_response")

        if self.intent == IntentType.ESCALATE:
            if not self.escalation_reason:
                raise ValueError("ESCALATE intent requires escalation_reason")

        return self


class Citation(BaseModel):
    """Citation extracted from response"""

    source_name: str
    title: str
    url: Optional[str] = None
    excerpt: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


# =============================================================================
# STATE SCHEMA - CANONICAL DEFINITION
# =============================================================================


class AmaniQState(TypedDict, total=False):
    """
    Master state schema for AmaniQ v2 agent graph.

    This is the CANONICAL state definition - all nodes must use this.
    Uses proper LangGraph reducers for message handling.
    """

    # =========================================================================
    # Request Tracking
    # =========================================================================
    request_id: str
    thread_id: str
    user_id: str

    # =========================================================================
    # Input / Query
    # =========================================================================
    current_query: str
    original_question: str
    # Use add_messages reducer for proper message handling
    messages: Annotated[List[BaseMessage], add_messages]
    user_profile: Dict[str, Any]

    # =========================================================================
    # Supervisor Decision
    # =========================================================================
    supervisor_decision: Dict[str, Any]  # Serialized SupervisorDecision
    intent: str  # IntentType value
    confidence: float
    requires_multi_hop: bool  # NEW: For ReAct routing

    # =========================================================================
    # Tool Execution
    # =========================================================================
    tool_plan: List[Dict[str, Any]]  # List of ToolCall dicts
    tool_results: List[Dict[str, Any]]
    tool_execution_status: str  # ToolStatus value
    tool_execution_latency_ms: float
    tool_success_rate: float

    # =========================================================================
    # Prefetch / Cache
    # =========================================================================
    prefetch_used: bool
    prefetch_results: Optional[Dict[str, Any]]
    from_cache: bool
    cached_answer: Optional[Dict[str, Any]]

    # =========================================================================
    # Clarification
    # =========================================================================
    clarification_count: int
    clarification_history: List[Dict[str, str]]
    current_clarification_question: str
    waiting_for_user: bool
    clarification_resolved: bool
    clarification_status: str  # ClarificationStatus value
    final_enriched_query: str
    max_clarification_reached: bool

    # =========================================================================
    # ReAct Agent
    # =========================================================================
    react_iterations: List[Dict[str, Any]]
    react_final_answer: Optional[str]
    react_success: bool
    react_failed: bool
    react_tool_results: List[Dict[str, Any]]  # Tool results from ReAct

    # =========================================================================
    # Response
    # =========================================================================
    final_response: str
    analysis: str  # Internal reasoning (not shown to user)
    citations: List[Dict[str, Any]]
    response_confidence: float
    human_review_required: bool
    completed_at: str  # ISO timestamp

    # =========================================================================
    # Metadata
    # =========================================================================
    iteration_count: int
    max_iterations: int
    detected_language: str
    detected_entities: List[str]
    token_usage: Dict[str, int]

    # =========================================================================
    # Errors
    # =========================================================================
    error: Optional[str]
    error_details: List[str]


# =============================================================================
# STATE FACTORY FUNCTIONS
# =============================================================================


def create_initial_state(
    query: str,
    thread_id: str,
    user_id: str = "anonymous",
    request_id: Optional[str] = None,
    messages: Optional[List[BaseMessage]] = None,
    user_profile: Optional[Dict[str, Any]] = None,
) -> AmaniQState:
    """
    Create initial state for a new conversation turn.

    Args:
        query: User's question
        thread_id: Unique conversation thread ID
        user_id: User identifier
        request_id: Optional request tracking ID
        messages: Optional existing message history
        user_profile: Optional user context/preferences

    Returns:
        Initialized AmaniQState
    """
    from uuid import uuid4

    return AmaniQState(
        # Request tracking
        request_id=request_id or str(uuid4()),
        thread_id=thread_id,
        user_id=user_id,
        # Input
        current_query=query,
        original_question=query,
        messages=messages or [],
        user_profile=user_profile or {},
        # Supervisor
        supervisor_decision={},
        intent="",
        confidence=0.0,
        requires_multi_hop=False,
        # Tools
        tool_plan=[],
        tool_results=[],
        tool_execution_status="pending",
        tool_execution_latency_ms=0.0,
        tool_success_rate=0.0,
        # Cache/Prefetch
        prefetch_used=False,
        prefetch_results=None,
        from_cache=False,
        cached_answer=None,
        # Clarification
        clarification_count=0,
        clarification_history=[],
        current_clarification_question="",
        waiting_for_user=False,
        clarification_resolved=False,
        clarification_status="",
        final_enriched_query="",
        max_clarification_reached=False,
        # ReAct
        react_iterations=[],
        react_final_answer=None,
        react_success=False,
        react_failed=False,
        react_tool_results=[],
        # Response
        final_response="",
        analysis="",
        citations=[],
        response_confidence=0.0,
        human_review_required=False,
        completed_at="",
        # Metadata
        iteration_count=0,
        max_iterations=3,
        detected_language="en",
        detected_entities=[],
        token_usage={},
        # Errors
        error=None,
        error_details=[],
    )


def merge_state_updates(current: AmaniQState, updates: Dict[str, Any]) -> AmaniQState:
    """
    Merge state updates into current state.
    Handles special cases like message reducer.

    Args:
        current: Current state
        updates: Updates to apply

    Returns:
        Updated state
    """
    merged = dict(current)

    for key, value in updates.items():
        if key == "messages" and key in merged:
            # Messages use add_messages reducer - append, don't replace
            # The reducer handles deduplication by message ID
            merged[key] = merged[key] + value if value else merged[key]
        elif key == "error_details" and key in merged:
            # Error details accumulate
            merged[key] = merged.get(key, []) + (
                value if isinstance(value, list) else [value]
            )
        else:
            merged[key] = value

    return AmaniQState(**merged)


# =============================================================================
# TOOL RESULT TYPES
# =============================================================================


class ToolResult(TypedDict):
    """Standard tool result format"""

    tool_name: str
    query: str
    status: str  # ToolStatus value
    data: Any
    latency_ms: float
    error: Optional[str]
    from_prefetch: bool
    from_cache: bool


class ReActIteration(TypedDict):
    """Single ReAct iteration record"""

    iteration: int
    thought: str
    action: str
    observation: str
    tool_result: Optional[Dict[str, Any]]
    success: bool


# =============================================================================
# CHECKPOINT / PERSISTENCE TYPES
# =============================================================================


class CheckpointConfig(TypedDict):
    """Configuration for state checkpointing"""

    thread_id: str
    checkpoint_ns: str
    checkpoint_id: str


class InterruptConfig(TypedDict):
    """Configuration for interrupt/resume"""

    interrupt_before: List[str]  # Node names to interrupt before
    interrupt_after: List[str]  # Node names to interrupt after


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "IntentType",
    "ToolName",
    "ClarificationStatus",
    "ToolStatus",
    # Pydantic models
    "ToolCall",
    "ClarificationRequest",
    "SupervisorDecision",
    "Citation",
    # State
    "AmaniQState",
    "create_initial_state",
    "merge_state_updates",
    # Result types
    "ToolResult",
    "ReActIteration",
    # Config types
    "CheckpointConfig",
    "InterruptConfig",
    # LangGraph types (re-export for convenience)
    "BaseMessage",
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
    "add_messages",
    "LANGGRAPH_AVAILABLE",
]
