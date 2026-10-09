"""
AmaniQ v2 - Production-Ready Legal Research Agent Graph
========================================================

Complete LangGraph implementation with:
- Moonshot AI Supervisor for intent routing
- Parallel tool execution with fault tolerance
- Human-in-the-loop clarification (max 3 rounds)
- Redis caching (48h answers, 24h vectors)
- Speculative pre-fetch for legal queries
- OpenTelemetry tracing for latency debugging
- Natural Kenyan English/Swahili responses

Architecture:
    User Message
         │
         ├──► [Speculative Prefetch] ──► (runs in parallel)
         │
         ▼
    ┌─────────────┐
    │ SUPERVISOR  │──► intent routing
    └──────┬──────┘
           │
    ┌──────┴──────┬────────────┬───────────┐
    │             │            │           │
    ▼             ▼            ▼           ▼
 CLARIFY     TOOL_EXEC     RESPOND     ESCALATE
    │             │            │
    │             ▼            │
    │        RESPONDER         │
    │             │            │
    └─────────────┴────────────┘
                  │
                  ▼
                 END

Author: Eng. Onyango Benard
Version: 2.0.0
Date: 2025
"""

import asyncio
import os
import time
import json
import threading
import httpx
from typing import Dict, Any, List, Optional, Literal, TypedDict, Annotated
from datetime import datetime
from uuid import uuid4
from dataclasses import dataclass, field
from loguru import logger

import operator

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver

    LANGGRAPH_AVAILABLE = True
except ImportError:
    logger.warning("LangGraph not installed. Run: pip install langgraph")
    StateGraph = None
    END = None
    MemorySaver = None
    LANGGRAPH_AVAILABLE = False

# OpenAI client for Moonshot
from openai import OpenAI

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import AmaniQ v2 components
from .prompts.supervisor_prompt import (
    SUPERVISOR_SYSTEM_PROMPT,
    SupervisorDecision,
    IntentType,
    ToolName,
    build_supervisor_messages,
    get_moonshot_config,
    parse_supervisor_response,
    count_tokens,
    check_context_overflow,
)

from .prompts.responder_prompt import (
    RESPONDER_SYSTEM_PROMPT,
    build_responder_messages,
    format_tool_data,
    get_responder_config,
    extract_analysis,
)

from .nodes.tool_executor import (
    ToolExecutor,
    ToolExecutorConfig,
    ToolStatus,
    ToolResult,
    tool_executor_node,
    get_executor,
)

from .nodes.clarification import (
    ClarificationState,
    ClarificationStatus,
    MAX_CLARIFICATION_ROUNDS,
    clarification_entry_node,
    clarification_resume_node,
    max_clarification_node,
    should_clarify,
    clarification_route,
    after_clarification_route,
    build_clarification_question,
)

from .optimization import (
    RedisCache,
    get_cache,
    AnswerCache,
    VectorSearchCache,
    ConversationSummarizer,
    CachedKBSearch,
    CachingMiddleware,
    startup_prewarm,
    TOP_LEGAL_QUERIES,
)

from .prefetch import (
    TelemetryMetrics,
    SpeculativePrefetcher,
    PrefetchMiddleware,
    traced_node,
    get_metrics_collector,
    MetricsCollector,
)

from .tools.tool_registry import ToolRegistry

# ReAct Agent
from .nodes.react_node import react_reasoning_node, react_tool_node
from .tools.agentic_tools import initialize_agentic_tools, get_agentic_tools

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class AmaniQConfig:
    """Configuration for AmaniQ v2 agent"""

    # LLM provider selection
    llm_provider: str = field(
        default_factory=lambda: os.getenv("AMANIQ_LLM_PROVIDER", "moonshot")
    )

    # Moonshot AI settings
    moonshot_api_key: str = field(
        default_factory=lambda: os.getenv("MOONSHOT_API_KEY", "")
    )
    moonshot_base_url: str = field(
        default_factory=lambda: os.getenv(
            "MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1"
        )
    )

    # NVIDIA NIM settings
    nvidia_nim_api_key: str = field(
        default_factory=lambda: os.getenv("NVIDIA_NIM_API_KEY", "")
    )
    nvidia_nim_base_url: str = field(
        default_factory=lambda: os.getenv(
            "NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
    )
    nvidia_nim_model: str = field(
        default_factory=lambda: os.getenv(
            "NVIDIA_NIM_MODEL", "meta/llama-3.1-70b-instruct"
        )
    )

    # Model settings
    supervisor_model: str = "moonshot-v1-32k"
    responder_model: str = "moonshot-v1-128k"

    # Execution settings
    max_iterations: int = 3
    max_clarification_rounds: int = 3
    tool_timeout_seconds: float = 8.0
    max_parallel_tools: int = 4

    # Caching settings
    enable_caching: bool = True
    enable_prefetch: bool = True

    # Tracing settings
    enable_telemetry: bool = True

    # Persistence
    enable_persistence: bool = False
    checkpoint_path: str = "./checkpoints/amaniq_v2.db"


# =============================================================================
# STATE SCHEMA
# =============================================================================


class AmaniQState(TypedDict, total=False):
    """Master state schema for AmaniQ v2 agent graph"""

    # Request tracking
    request_id: str
    thread_id: str

    # Input
    current_query: str
    original_question: str
    messages: Annotated[List[Dict[str, Any]], operator.add]

    # Supervisor output
    supervisor_decision: Dict[str, Any]
    intent: str
    confidence: float

    # Tool execution
    tool_plan: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    tool_execution_status: str
    tool_execution_latency_ms: float
    tool_success_rate: float

    # Prefetch
    prefetch_used: bool
    prefetch_results: Optional[Dict[str, Any]]

    # Clarification
    clarification_count: int
    clarification_history: List[Dict[str, str]]
    current_clarification_question: str
    waiting_for_user: bool
    clarification_resolved: bool
    final_enriched_query: str
    max_clarification_reached: bool

    # Response
    final_response: str
    analysis: str
    citations: List[Dict[str, Any]]

    # Caching
    from_cache: bool
    cached_answer: Optional[Dict[str, Any]]

    # Metadata
    iteration_count: int
    max_iterations: int
    detected_language: str
    detected_entities: List[str]
    token_usage: Dict[str, int]

    # Quality
    response_confidence: float
    human_review_required: bool

    # Errors
    error: Optional[str]
    error_details: List[str]

    # User Context
    user_id: str
    user_profile: Dict[str, Any]

    # ReAct Agent (Modern)
    react_messages: List[Dict[str, Any]]  # OpenAI-format message history for ReAct
    react_last_message: Any  # Last AIMessage (for tool calls)
    react_status: Literal["continue", "done", "error"]
    react_final_answer: Optional[str]
    react_success: bool
    react_failed: bool
    requires_multi_hop: bool

    # Timestamps
    started_at: str
    completed_at: Optional[str]


# =============================================================================
# MOONSHOT CLIENT
# =============================================================================


class CircuitBreaker:
    """Circuit breaker for LLM API calls to prevent cascading failures"""

    STATE_CLOSED = "closed"
    STATE_OPEN = "open"
    STATE_HALF_OPEN = "half_open"

    def __init__(
        self,
        threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_requests: int = 1,
    ):
        self.threshold = threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_requests = half_open_max_requests
        self.state = self.STATE_CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.half_open_requests = 0
        self._lock = threading.Lock()

    def record_success(self):
        with self._lock:
            if self.state == self.STATE_HALF_OPEN:
                self.half_open_requests -= 1
                self.success_count += 1
                if self.success_count >= 2:
                    self.state = self.STATE_CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info("Circuit breaker reset to CLOSED")
            elif self.state == self.STATE_CLOSED:
                self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.state == self.STATE_HALF_OPEN:
                self.state = self.STATE_OPEN
                self.half_open_requests = 0
                logger.warning(f"Circuit breaker returned to OPEN (half-open failure)")
            elif (
                self.state == self.STATE_CLOSED and self.failure_count >= self.threshold
            ):
                self.state = self.STATE_OPEN
                logger.warning(
                    f"Circuit breaker OPEN after {self.failure_count} failures"
                )

    def allow_request(self) -> bool:
        with self._lock:
            if self.state == self.STATE_CLOSED:
                return True
            if self.state == self.STATE_OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = self.STATE_HALF_OPEN
                    self.half_open_requests = 0
                    self.success_count = 0
                    logger.info("Circuit breaker HALF_OPEN (attempting recovery)")
                    return True
                return False
            if self.state == self.STATE_HALF_OPEN:
                if self.half_open_requests < self.half_open_max_requests:
                    self.half_open_requests += 1
                    return True
                return False
            return True


class TokenBucketRateLimiter:
    """Simple token bucket rate limiter for API calls"""

    def __init__(self, rate: float = 10.0, burst: int = 20):
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> float:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_refill = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0
            deficit = tokens - self.tokens
            wait = deficit / self.rate
            self.tokens = 0.0
            return wait


class MoonshotClient:
    """Singleton client for OpenAI-compatible APIs (Moonshot, NVIDIA NIM, etc.)

    Supports multiple OpenAI-compatible backends via AmaniQConfig.llm_provider.
    Features: connection pooling, rate limiting, circuit breaker.
    """

    _instances: Dict[str, OpenAI] = {}
    _async_client: Optional["httpx.AsyncClient"] = None
    _config: Optional[AmaniQConfig] = None
    _rate_limiter: TokenBucketRateLimiter = TokenBucketRateLimiter(rate=10.0, burst=20)
    _circuit_breaker: CircuitBreaker = CircuitBreaker(
        threshold=5, recovery_timeout=30.0
    )

    @classmethod
    def get_client(cls, config: Optional[AmaniQConfig] = None) -> OpenAI:
        """Get or create client for the configured provider (Moonshot, NVIDIA NIM, etc.)"""
        if not cls._circuit_breaker.allow_request():
            raise RuntimeError(
                "LLM API circuit breaker is OPEN. Please try again later."
            )

        cfg = config or AmaniQConfig()
        provider = cfg.llm_provider
        provider_key = f"{provider}:{cfg.moonshot_base_url if provider == 'moonshot' else cfg.nvidia_nim_base_url}"

        if provider_key not in cls._instances or config != cls._config:
            cls._config = cfg
            if provider == "nvidia_nim":
                cls._instances[provider_key] = OpenAI(
                    api_key=cfg.nvidia_nim_api_key,
                    base_url=cfg.nvidia_nim_base_url,
                    http_client=cls._get_http_client(),
                )
                logger.info(
                    f"NVIDIA NIM client initialized: {cfg.nvidia_nim_base_url} ({cfg.nvidia_nim_model})"
                )
            else:
                # Default: Moonshot
                cls._instances[provider_key] = OpenAI(
                    api_key=cfg.moonshot_api_key,
                    base_url=cfg.moonshot_base_url,
                    http_client=cls._get_http_client(),
                )
                logger.info(f"Moonshot client initialized: {cfg.moonshot_base_url}")
        return cls._instances[provider_key]

    @classmethod
    def _get_http_client(cls) -> Any:
        """Get or create shared httpx client with connection pooling"""
        import httpx

        if cls._async_client is None:
            cls._async_client = httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
                timeout=httpx.Timeout(30.0, connect=5.0),
            )
        return cls._async_client

    @classmethod
    def rate_limit(cls):
        """Block until rate limit allows the request"""
        wait = cls._rate_limiter.acquire()
        if wait > 0:
            logger.debug(f"Rate limited, waiting {wait:.2f}s")
            time.sleep(wait)

    @classmethod
    def record_success(cls):
        cls._circuit_breaker.record_success()

    @classmethod
    def record_failure(cls):
        cls._circuit_breaker.record_failure()

    @classmethod
    def reset(cls):
        """Reset all clients (for testing)"""
        cls._instances.clear()
        cls._async_client = None
        cls._config = None
        cls._rate_limiter = TokenBucketRateLimiter(rate=10.0, burst=20)
        cls._circuit_breaker = CircuitBreaker(threshold=5, recovery_timeout=30.0)


# =============================================================================
# NODE FUNCTIONS
# =============================================================================


async def entry_node(state: AmaniQState) -> AmaniQState:
    """
    Entry node - Initialize request tracking and check cache.
    """
    logger.info("=== ENTRY NODE ===")

    # Generate request ID if not present
    request_id = state.get("request_id") or str(uuid4())
    thread_id = state.get("thread_id") or str(uuid4())

    # Extract current query from messages
    messages = state.get("messages", [])
    current_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            current_query = msg.get("content", "")
            break

    # Initialize state
    updates = {
        "request_id": request_id,
        "thread_id": thread_id,
        "current_query": current_query,
        "original_question": state.get("original_question") or current_query,
        "iteration_count": 0,
        "max_iterations": 3,
        "clarification_count": state.get("clarification_count", 0),
        "clarification_history": state.get("clarification_history", []),
        "waiting_for_user": False,
        "clarification_resolved": False,
        "error_details": [],
        "from_cache": False,
        "prefetch_used": False,
        "started_at": datetime.utcnow().isoformat(),
        "user_id": state.get("user_id"),
        "user_profile": state.get("user_profile"),
    }

    # Build User Profile if missing
    user_id = state.get("user_id")
    if user_id and not updates.get("user_profile"):
        try:
            from Module3_NiruDB.chat_manager_v2 import get_chat_manager

            chat_manager = get_chat_manager()
            history = chat_manager.get_user_interaction_history(user_id, limit=100)

            if history:
                logger.info(
                    f"Building user profile from {len(history)} interactions for {user_id}"
                )
                config = AmaniQConfig()
                client = MoonshotClient.get_client(config)

                # Fetch current task clusters for better classification
                try:
                    from Module4_NiruAPI.agents.task_clustering import ClusterAnalyzer

                    analyzer = ClusterAnalyzer()
                    clusters = analyzer.get_current_clusters(limit=12)
                    cluster_names = [c["cluster_name"] for c in clusters]
                    cluster_descriptions = "\n".join(
                        [f"- {c['cluster_name']}: {c['description']}" for c in clusters]
                    )
                except Exception as e:
                    logger.warning(f"Could not fetch clusters: {e}")
                    cluster_names = []
                    cluster_descriptions = "No clusters available"

                profile_prompt = f"""
                Build a user profile from the last {len(history)} interactions of user {user_id}.
                
                Return a JSON object with EXACTLY these keys:
                - "expertise_level": (layperson / lawyer / researcher / journalist)
                - "task_groups": [list of top 3 task groups from: {cluster_names if cluster_names else 'general legal research, case law lookup, constitutional queries, news tracking'}]
                - "preferred_answer_style": (concise / detailed / bullet_points / kenya_law_format)
                - "frequent_topics": [list of frequently tracked bills or topics]
                
                Available task clusters:
                {cluster_descriptions}
                
                Interactions:
                {json.dumps(history[-20:], default=str)}
                """

                MoonshotClient.rate_limit()
                try:
                    response = client.chat.completions.create(
                        model="moonshot-v1-8k",
                        messages=[{"role": "user", "content": profile_prompt}],
                        response_format={"type": "json_object"},
                    )
                    MoonshotClient.record_success()
                except Exception:
                    MoonshotClient.record_failure()
                    raise

                user_profile = json.loads(response.choices[0].message.content)
                updates["user_profile"] = user_profile
                logger.info(f"Built user profile: {user_profile}")
        except Exception as e:
            logger.warning(f"Failed to build user profile: {e}")

    # Check answer cache
    try:
        cache = await get_cache()
        answer_cache = AnswerCache(cache)
        cached = await answer_cache.get_cached_answer(current_query)

        if cached:
            logger.info(f"Cache HIT for query: {current_query[:50]}...")
            TelemetryMetrics.record_cache_hit("answer")
            updates["cached_answer"] = cached
            updates["from_cache"] = True
        else:
            TelemetryMetrics.record_cache_miss("answer")

    except Exception as e:
        logger.warning(f"Cache check failed: {e}")

    return {**state, **updates}


async def supervisor_node(state: AmaniQState) -> AmaniQState:
    """
    Supervisor node - Route query to appropriate handler.
    Uses Moonshot AI with strict JSON output.
    """
    logger.info("=== SUPERVISOR NODE ===")

    # Skip if using cached answer
    if state.get("from_cache") and state.get("cached_answer"):
        logger.info("Skipping supervisor - using cached answer")
        return {
            **state,
            "supervisor_decision": {"intent": "CACHED"},
            "intent": "CACHED",
        }

    query = state.get("current_query", "")
    messages_history = state.get("messages", [])

    try:
        # Build supervisor messages
        supervisor_messages = build_supervisor_messages(
            user_query=query,
            message_history=messages_history,
            user_context=state.get("user_profile"),
        )

        # Check token count
        token_count, overflow = check_context_overflow(supervisor_messages)

        if overflow:
            logger.warning(f"Context overflow: {token_count} tokens")
            return {
                **state,
                "supervisor_decision": {
                    "intent": "ESCALATE",
                    "confidence": 1.0,
                    "reasoning": "Context window exceeded",
                    "escalation_reason": "Your conversation exceeds the token limit. Please start a new conversation.",
                    "context_overflow": True,
                    "token_count": token_count,
                },
                "intent": "ESCALATE",
            }

        # Call Moonshot with rate limiting and circuit breaker
        config = AmaniQConfig()
        MoonshotClient.rate_limit()
        client = MoonshotClient.get_client(config)
        moonshot_config = get_moonshot_config()

        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=moonshot_config["model"],
                messages=supervisor_messages,
                temperature=moonshot_config["temperature"],
                max_tokens=moonshot_config["max_tokens"],
                response_format=moonshot_config["response_format"],
            )
            MoonshotClient.record_success()
        except Exception as e:
            MoonshotClient.record_failure()
            raise
        latency_ms = (time.time() - start_time) * 1000

        # Parse response
        response_text = response.choices[0].message.content
        decision = parse_supervisor_response(response_text)

        # Record metrics
        if config.enable_telemetry:
            TelemetryMetrics.record_tokens(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                "supervisor",
            )

        logger.info(
            f"Supervisor decision: {decision.intent.value} "
            f"(confidence: {decision.confidence:.2f}, {latency_ms:.0f}ms)"
        )

        # Build tool plan from decision
        tool_plan = []
        if decision.tool_plan:
            for tc in decision.tool_plan:
                tool_plan.append(
                    {
                        "tool_name": tc.tool_name.value,
                        "query": tc.query,
                        "priority": tc.priority,
                    }
                )

        # Extract requires_multi_hop from decision (defaults to False if not present)
        requires_multi_hop = getattr(decision, "requires_multi_hop", False)

        return {
            **state,
            "supervisor_decision": decision.model_dump(),
            "intent": decision.intent.value,
            "confidence": decision.confidence,
            "tool_plan": tool_plan,
            "requires_multi_hop": requires_multi_hop,
            "detected_language": decision.detected_language,
            "detected_entities": decision.detected_entities,
            "token_usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        }

    except Exception as e:
        logger.error(f"Supervisor error: {e}")
        return {
            **state,
            "supervisor_decision": {
                "intent": "ESCALATE",
                "confidence": 0.0,
                "reasoning": f"Supervisor error: {str(e)}",
                "escalation_reason": "I encountered an error processing your request. Please try again.",
            },
            "intent": "ESCALATE",
            "error": str(e),
            "error_details": state.get("error_details", []) + [f"Supervisor: {e}"],
        }


async def tool_executor_wrapper(state: AmaniQState) -> AmaniQState:
    """
    Tool executor wrapper - Calls the tool_executor_node with prefetch support.
    """
    logger.info("=== TOOL EXECUTOR NODE ===")

    # Check if we have prefetch results
    prefetch_results = state.get("prefetch_results")
    tool_plan = state.get("tool_plan", [])

    if prefetch_results:
        # Use prefetch results for kb_search
        modified_results = []
        for tc in tool_plan:
            if tc.get("tool_name") == "kb_search" and prefetch_results:
                modified_results.append(
                    {
                        "tool_name": "kb_search",
                        "query": tc.get("query", ""),
                        "status": "success",
                        "data": prefetch_results,
                        "from_prefetch": True,
                        "latency_ms": 0,
                    }
                )
                logger.info("Using prefetch results for kb_search")
                TelemetryMetrics.record_prefetch_hit()
            else:
                # Will be executed below
                pass

        # Execute remaining tools
        remaining_plan = [
            tc
            for tc in tool_plan
            if tc.get("tool_name") != "kb_search" or not prefetch_results
        ]

        if remaining_plan:
            result = await tool_executor_node(
                {
                    **state,
                    "supervisor_decision": {
                        **state.get("supervisor_decision", {}),
                        "tool_plan": remaining_plan,
                    },
                }
            )
            tool_results = modified_results + result.get("tool_results", [])
        else:
            tool_results = modified_results

        return {
            **state,
            "tool_results": tool_results,
            "tool_execution_status": "complete",
            "prefetch_used": True,
        }

    # No prefetch - execute normally
    result = await tool_executor_node(state)
    return {**state, **result}


async def responder_node(state: AmaniQState) -> AmaniQState:
    """
    Responder node - Synthesize final response from tool results or ReAct output.
    Uses Moonshot AI with large context window.
    """
    logger.info("=== RESPONDER NODE ===")

    # Handle cached response
    if state.get("from_cache") and state.get("cached_answer"):
        cached = state["cached_answer"]
        return {
            **state,
            "final_response": cached.get("answer", ""),
            "citations": cached.get("citations", []),
            "response_confidence": 0.95,  # High confidence for cached
            "completed_at": datetime.utcnow().isoformat(),
        }

    # Handle ReAct agent results FIRST (priority over tool results)
    if state.get("react_success") and state.get("react_final_answer"):
        logger.info("[Responder] Using ReAct agent final answer")
        react_answer = state["react_final_answer"]
        return {
            **state,
            "final_response": react_answer,
            "response_confidence": 0.85,  # ReAct answers are generally reliable
            "completed_at": datetime.utcnow().isoformat(),
            "analysis": f"ReAct agent completed in {len(state.get('react_iterations', []))} iterations",
        }

    # Handle ReAct failure - provide graceful degradation
    if state.get("react_failed"):
        logger.warning("[Responder] ReAct agent failed - falling back to error message")
        return {
            **state,
            "final_response": (
                "I encountered difficulties processing your query. "
                "This may be due to tool limitations or the query requiring information "
                "I don't currently have access to. Please try rephrasing your question "
                "or breaking it into simpler parts."
            ),
            "response_confidence": 0.3,
            "human_review_required": True,
            "completed_at": datetime.utcnow().isoformat(),
            "error": "ReAct agent failed",
        }

    # Handle escalation
    intent = state.get("intent", "")
    if intent == "ESCALATE":
        supervisor_decision = state.get("supervisor_decision", {})
        escalation_reason = supervisor_decision.get(
            "escalation_reason",
            "I'm unable to process this request. Please consult a qualified legal professional.",
        )
        return {
            **state,
            "final_response": f"⚠️ **Notice**: {escalation_reason}",
            "response_confidence": 0.3,
            "human_review_required": True,
            "completed_at": datetime.utcnow().isoformat(),
        }

    # Handle direct response (GENERAL_CHAT)
    if intent == "GENERAL_CHAT":
        supervisor_decision = state.get("supervisor_decision", {})
        direct_response = supervisor_decision.get("direct_response", "")
        return {
            **state,
            "final_response": direct_response,
            "response_confidence": state.get("confidence", 0.9),
            "completed_at": datetime.utcnow().isoformat(),
        }

    # Build responder prompt for normal tool results
    tool_results = state.get("tool_results", [])
    supervisor_decision = state.get("supervisor_decision", {})
    original_question = state.get("original_question", state.get("current_query", ""))

    responder_messages = build_responder_messages(
        original_question=original_question,
        tool_results=tool_results,
        supervisor_decision=supervisor_decision,
        message_history=state.get("messages", []),
        user_context=state.get("user_profile"),
    )

    try:
        # Call Moonshot with large context model
        config = AmaniQConfig()
        MoonshotClient.rate_limit()
        client = MoonshotClient.get_client(config)
        responder_config = get_responder_config()

        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=responder_config["model"],
                messages=responder_messages,
                temperature=responder_config["temperature"],
                max_tokens=responder_config["max_tokens"],
                stream=False,  # For now, non-streaming
            )
            MoonshotClient.record_success()
        except Exception as e:
            MoonshotClient.record_failure()
            raise
        latency_ms = (time.time() - start_time) * 1000

        response_text = response.choices[0].message.content

        # Extract analysis section (internal thinking)
        analysis, user_response = extract_analysis(response_text)

        # Record metrics
        if config.enable_telemetry:
            TelemetryMetrics.record_tokens(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                "responder",
            )

        logger.info(
            f"Responder generated {len(user_response)} chars in {latency_ms:.0f}ms"
        )

        # Cache the answer
        if config.enable_caching:
            try:
                cache = await get_cache()
                answer_cache = AnswerCache(cache)
                await answer_cache.cache_answer(
                    query=original_question,
                    answer=user_response,
                    citations=[],  # Could extract from response
                )
            except Exception as e:
                logger.warning(f"Failed to cache answer: {e}")

        # Calculate confidence based on tool success rate
        tool_success = state.get("tool_success_rate", 1.0)
        supervisor_confidence = state.get("confidence", 0.8)
        response_confidence = tool_success * 0.4 + supervisor_confidence * 0.6

        return {
            **state,
            "final_response": user_response,
            "analysis": analysis,
            "response_confidence": response_confidence,
            "token_usage": {
                **state.get("token_usage", {}),
                "responder_prompt": response.usage.prompt_tokens,
                "responder_completion": response.usage.completion_tokens,
            },
            "completed_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Responder error: {e}")
        return {
            **state,
            "final_response": (
                "Samahani, I encountered an issue generating a complete response. "
                f"Based on my search, here's what I found:\n\n"
                f"{_format_fallback_response(tool_results)}"
            ),
            "error": str(e),
            "error_details": state.get("error_details", []) + [f"Responder: {e}"],
            "response_confidence": 0.4,
            "completed_at": datetime.utcnow().isoformat(),
        }


def _format_fallback_response(tool_results: List[Dict]) -> str:
    """Format fallback response from raw tool results"""
    if not tool_results:
        return "I couldn't find relevant information for your query."

    parts = []
    for result in tool_results[:3]:
        tool_name = result.get("tool_name", "search")
        data = result.get("data", {})

        if isinstance(data, dict):
            search_results = data.get("search_results", [])
            for item in search_results[:2]:
                content = item.get("content", "")[:300]
                parts.append(f"- {content}...")

    return "\n".join(parts) if parts else "Limited information available."


async def clarification_entry_wrapper(state: AmaniQState) -> AmaniQState:
    """Wrapper for clarification entry node"""
    return clarification_entry_node(state)


async def clarification_resume_wrapper(state: AmaniQState) -> AmaniQState:
    """Wrapper for clarification resume node"""
    return clarification_resume_node(state)


async def max_clarification_wrapper(state: AmaniQState) -> AmaniQState:
    """Wrapper for max clarification node"""
    return max_clarification_node(state)


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================


def route_from_entry(state: AmaniQState) -> Literal["supervisor", "respond"]:
    """Route from entry - check if cached"""
    if state.get("from_cache") and state.get("cached_answer"):
        return "respond"
    return "supervisor"


def route_from_supervisor(
    state: AmaniQState,
) -> Literal["clarify", "tools", "react", "respond", "escalate"]:
    """
    Route based on supervisor intent and LLM-detected multi-hop requirement.

    Uses requires_multi_hop field from SupervisorDecision instead of regex heuristics.
    """
    intent = state.get("intent", "")

    # LLM-based multi-hop detection (from SupervisorDecision.requires_multi_hop)
    requires_multi_hop = state.get("requires_multi_hop", False)

    if intent == "CLARIFY":
        return "clarify"
    elif intent in ("LEGAL_RESEARCH", "NEWS_SUMMARY"):
        # Route multi-hop queries to ReAct for sequential reasoning
        if requires_multi_hop:
            logger.info(
                f"[Router] Multi-hop query detected (LLM) - routing to ReAct agent"
            )
            return "react"

        return "tools"
    elif intent == "GENERAL_CHAT":
        return "respond"
    elif intent == "ESCALATE":
        return "escalate"
    elif intent == "CACHED":
        return "respond"
    else:
        # Default to tools for unknown intents
        return "tools"


def route_from_react(state: AmaniQState) -> Literal["respond", "fallback_tools"]:
    """
    Route from ReAct agent based on success/failure.

    If ReAct failed, fall back to parallel tool execution.
    If ReAct succeeded, proceed to responder.
    """
    if state.get("react_failed"):
        logger.warning("[Router] ReAct failed - falling back to parallel tools")
        return "fallback_tools"
    return "respond"


def route_clarification(
    state: AmaniQState,
) -> Literal["wait", "resume", "max_reached", "done"]:
    """Route within clarification sub-graph"""
    return clarification_route(state)


def route_after_clarification(
    state: AmaniQState,
) -> Literal["supervisor", "tools", "respond"]:
    """Route after clarification resolved"""
    return after_clarification_route(state)


# =============================================================================
# GRAPH BUILDER
# =============================================================================


def create_amaniq_v2_graph(
    config: Optional[AmaniQConfig] = None,
    enable_persistence: bool = False,
) -> StateGraph:
    """
    Create the AmaniQ v2 agent graph.

    Args:
        config: Optional configuration
        enable_persistence: Enable state checkpointing

    Returns:
        Compiled LangGraph StateGraph
    """
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("LangGraph not installed. Run: pip install langgraph")

    cfg = config or AmaniQConfig()
    logger.info("Building AmaniQ v2 Agent Graph...")

    # Create graph
    workflow = StateGraph(AmaniQState)

    # Add nodes
    workflow.add_node("entry", entry_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("tool_executor", tool_executor_wrapper)
    workflow.add_node("responder", responder_node)

    # ReAct Agent Nodes (Modern)
    workflow.add_node("react_reasoning", react_reasoning_node)
    workflow.add_node("react_tool_node", react_tool_node)

    # Clarification nodes
    workflow.add_node("clarification_entry", clarification_entry_wrapper)
    workflow.add_node("clarification_resume", clarification_resume_wrapper)
    workflow.add_node("max_clarification", max_clarification_wrapper)

    # Set entry point
    workflow.set_entry_point("entry")

    # Entry routing
    workflow.add_conditional_edges(
        "entry",
        route_from_entry,
        {
            "supervisor": "supervisor",
            "respond": "responder",
        },
    )

    # Supervisor routing
    supervisor_routes = {
        "clarify": "clarification_entry",
        "tools": "tool_executor",
        "react": "react_reasoning",  # Route to new reasoning node
        "respond": "responder",
        "escalate": "responder",
    }

    workflow.add_conditional_edges(
        "supervisor", route_from_supervisor, supervisor_routes
    )

    # Tool executor → Responder
    workflow.add_edge("tool_executor", "responder")

    # ReAct Agent Routing
    workflow.add_conditional_edges(
        "react_reasoning",
        lambda state: state.get("react_status", "done"),
        {
            "continue": "react_tool_node",
            "done": "responder",
            "error": "responder",  # Fallback to responder on error
        },
    )

    # Tool output loop back to reasoning
    workflow.add_edge("react_tool_node", "react_reasoning")

    # Clarification sub-graph
    workflow.add_conditional_edges(
        "clarification_entry",
        route_clarification,
        {
            "wait": END,  # Pause for user input (interrupt)
            "resume": "clarification_resume",
            "max_reached": "max_clarification",
            "done": "supervisor",
        },
    )

    workflow.add_conditional_edges(
        "clarification_resume",
        route_after_clarification,
        {
            "supervisor": "supervisor",
            "tools": "tool_executor",
            "respond": "responder",
        },
    )

    workflow.add_edge("max_clarification", "tool_executor")

    # Responder → END
    workflow.add_edge("responder", END)

    # Compile graph with appropriate checkpointer
    checkpointer = None
    interrupt_nodes = ["clarification_entry"]

    # Add HITL interrupt for tool execution in ReAct loop
    # This allows human review before tools are executed
    interrupt_nodes.append("react_tool_node")

    if enable_persistence:
        # Try PostgresSaver for production, fall back to MemorySaver
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            import os

            postgres_uri = os.getenv("POSTGRES_URI") or os.getenv("DATABASE_URL")
            if postgres_uri:
                checkpointer = PostgresSaver.from_conn_string(postgres_uri)
                logger.info("Using PostgresSaver for production checkpointing")
            else:
                checkpointer = MemorySaver()
                logger.warning(
                    "POSTGRES_URI not set - using MemorySaver (state not persisted across restarts)"
                )
        except ImportError:
            checkpointer = MemorySaver()
            logger.warning(
                "langgraph.checkpoint.postgres not available - using MemorySaver"
            )
        except Exception as e:
            checkpointer = MemorySaver()
            logger.warning(
                f"PostgresSaver failed to initialize: {e} - using MemorySaver"
            )

    if checkpointer:
        # Compile with interrupt support
        graph = workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=interrupt_nodes,  # Pause before clarification and tool execution
        )
    else:
        # Without persistence, interrupts won't work effectively across HTTP requests
        # but we compile anyway
        graph = workflow.compile()

    logger.info("AmaniQ v2 Graph built successfully!")
    return graph


# =============================================================================
# AMANIQ V2 AGENT CLASS
# =============================================================================


class AmaniQAgent:
    """
    Main AmaniQ v2 Agent class for API integration.

    Usage:
        agent = AmaniQAgent()
        await agent.initialize()

        response = await agent.chat(
            message="What does Article 27 say about equality?",
            thread_id="user-123"
        )
    """

    def __init__(self, config: Optional[AmaniQConfig] = None):
        """Initialize agent with configuration"""
        self.config = config or AmaniQConfig()
        self.graph: Optional[StateGraph] = None
        self.prefetch_middleware: Optional[PrefetchMiddleware] = None
        self.caching_middleware: Optional[CachingMiddleware] = None
        self.metrics_collector: MetricsCollector = get_metrics_collector()
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize agent components"""
        if self._initialized:
            return True

        try:
            # Build graph
            logger.info("Building AmaniQ v2 graph...")
            self.graph = create_amaniq_v2_graph(
                config=self.config,
                enable_persistence=self.config.enable_persistence,
            )

            if self.graph is None:
                raise RuntimeError("Graph creation returned None")

            logger.info("Graph built successfully")

            # Initialize caching (optional - don't fail if this fails)
            if self.config.enable_caching:
                try:
                    self.caching_middleware = CachingMiddleware()
                    await self.caching_middleware.initialize()
                    logger.info("Caching middleware initialized")
                except Exception as e:
                    logger.warning(
                        f"Caching middleware failed to initialize: {e}, continuing without cache"
                    )
                    self.caching_middleware = None

            # Initialize prefetch (optional - don't fail if this fails)
            if self.config.enable_prefetch:
                try:
                    self.prefetch_middleware = PrefetchMiddleware()

                    # Set up search function for prefetch
                    cached_search = CachedKBSearch()
                    await cached_search.initialize()

                    async def search_fn(
                        query: str, namespaces: List[str]
                    ) -> Dict[str, Any]:
                        return await cached_search.search(query, namespace=namespaces)

                    await self.prefetch_middleware.initialize(search_fn)
                    logger.info("Prefetch middleware initialized")
                except Exception as e:
                    logger.warning(
                        f"Prefetch middleware failed to initialize: {e}, continuing without prefetch"
                    )
                    self.prefetch_middleware = None

            # Initialize telemetry (optional - don't fail if this fails)
            if self.config.enable_telemetry:
                try:
                    TelemetryMetrics.initialize()
                    logger.info("Telemetry initialized")
                except Exception as e:
                    logger.warning(
                        f"Telemetry failed to initialize: {e}, continuing without telemetry"
                    )

            # Initialize agentic tools for ReAct agent (optional - don't fail if this fails)
            try:
                # Get dependencies from initialized services
                from Module3_NiruDB.qdrant_cloud import get_vector_store

                vector_store = await get_vector_store()

                # Initialize RAG pipeline if available
                rag_pipeline = None
                try:
                    from Module3_NiruDB.rag_pipeline import get_rag_pipeline

                    rag_pipeline = await get_rag_pipeline()
                except Exception:
                    logger.warning("RAG pipeline not available for agentic tools")

                # Initialize metadata manager if available
                metadata_manager = None
                try:
                    from Module3_NiruDB.metadata_manager import get_metadata_manager

                    metadata_manager = await get_metadata_manager()
                except Exception:
                    logger.warning("Metadata manager not available for agentic tools")

                # Initialize the agentic tool registry
                initialize_agentic_tools(
                    vector_store=vector_store,
                    rag_pipeline=rag_pipeline,
                    metadata_manager=metadata_manager,
                )
                logger.info("Agentic tools initialized for ReAct agent")
            except Exception as e:
                logger.warning(
                    f"Agentic tools failed to initialize: {e}, ReAct agent may have limited capability"
                )

            self._initialized = True
            logger.info("✅ AmaniQ v2 Agent initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ CRITICAL: Agent initialization failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            # Re-raise to ensure the API startup fails if the brain can't initialize
            raise RuntimeError(f"Failed to initialize AmaniQ v2 Agent: {e}") from e

    async def chat(
        self,
        message: str,
        thread_id: Optional[str] = None,
        message_history: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a chat message and return response.

        Args:
            message: User's message
            thread_id: Optional thread ID for conversation tracking
            message_history: Optional previous messages
            metadata: Optional metadata

        Returns:
            Response dict with answer, confidence, sources, etc.
        """
        # Ensure agent is initialized
        if not self._initialized:
            logger.warning("Agent not initialized, initializing now...")
            await self.initialize()

        # Double-check graph is available
        if self.graph is None:
            logger.error("CRITICAL: Graph is None after initialization!")
            raise RuntimeError("AmaniQ v2 graph failed to initialize properly")

        request_id = str(uuid4())
        thread_id = thread_id or str(uuid4())

        logger.info(
            f"[AmaniQ v2] Processing chat request {request_id[:8]}... for thread {thread_id[:8]}..."
        )

        # Start prefetch if enabled
        if self.prefetch_middleware:
            try:
                await self.prefetch_middleware.on_message_received(request_id, message)
            except Exception as e:
                logger.warning(f"Prefetch failed: {e}, continuing without prefetch")

        # Build initial state
        messages = message_history or []
        messages.append({"role": "user", "content": message})

        initial_state: AmaniQState = {
            "request_id": request_id,
            "thread_id": thread_id,
            "messages": messages,
            "original_question": message,
            "current_query": message,
            "user_id": user_id,
        }

        try:
            # Run graph with 60-second timeout
            start_time = time.time()

            logger.info(f"[AmaniQ v2] Invoking graph for: {message[:100]}...")
            result = await asyncio.wait_for(
                self.graph.ainvoke(initial_state), timeout=60.0
            )

            total_latency_ms = (time.time() - start_time) * 1000

            logger.info(f"[AmaniQ v2] Graph completed in {total_latency_ms:.0f}ms")

            # Format response
            return self._format_response(result, total_latency_ms)

        except Exception as e:
            logger.error(f"[AmaniQ v2] Chat error: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return {
                "answer": f"I apologize, but I encountered an error processing your request: {str(e)}. Please try again.",
                "confidence": 0.0,
                "error": str(e),
                "request_id": request_id,
                "thread_id": thread_id,
            }

    async def resume_clarification(
        self,
        thread_id: str,
        user_response: str,
        previous_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Resume graph after user provides clarification.

        Args:
            thread_id: Thread ID
            user_response: User's clarification response
            previous_state: Previous graph state

        Returns:
            Response dict
        """
        if not self._initialized:
            await self.initialize()

        # Add user response to messages
        messages = previous_state.get("messages", [])
        messages.append({"role": "user", "content": user_response})

        # Resume state
        resume_state = {
            **previous_state,
            "messages": messages,
            "waiting_for_user": False,
        }

        try:
            result = await asyncio.wait_for(
                self.graph.ainvoke(
                    resume_state, config={"configurable": {"thread_id": thread_id}}
                ),
                timeout=60.0,
            )
            return self._format_response(result, 0)

        except Exception as e:
            logger.error(f"Resume error: {e}")
            return {
                "answer": f"Error resuming: {str(e)}",
                "confidence": 0.0,
                "error": str(e),
            }

    def _format_response(
        self, state: AmaniQState, total_latency_ms: float
    ) -> Dict[str, Any]:
        """Format graph state into API response"""

        # Check if waiting for clarification
        if state.get("waiting_for_user"):
            return {
                "type": "clarification",
                "question": state.get("current_clarification_question", ""),
                "partial_understanding": state.get("supervisor_decision", {})
                .get("clarification", {})
                .get("partial_understanding", ""),
                "clarification_round": state.get("clarification_count", 1),
                "max_rounds": MAX_CLARIFICATION_ROUNDS,
                "request_id": state.get("request_id"),
                "thread_id": state.get("thread_id"),
                "state": dict(state),  # Return state for resume
            }

        # Normal response
        return {
            "type": "answer",
            "answer": state.get("final_response", ""),
            "confidence": state.get("response_confidence", 0.0),
            "intent": state.get("intent", ""),
            "detected_language": state.get("detected_language", "en"),
            "detected_entities": state.get("detected_entities", []),
            "citations": state.get("citations", []),
            "sources_count": len(state.get("tool_results", [])),
            "from_cache": state.get("from_cache", False),
            "prefetch_used": state.get("prefetch_used", False),
            "human_review_required": state.get("human_review_required", False),
            "request_id": state.get("request_id"),
            "thread_id": state.get("thread_id"),
            "metadata": {
                "started_at": state.get("started_at"),
                "completed_at": state.get("completed_at"),
                "total_latency_ms": total_latency_ms,
                "token_usage": state.get("token_usage", {}),
                "tool_success_rate": state.get("tool_success_rate", 0),
                "iteration_count": state.get("iteration_count", 0),
            },
            "error": state.get("error"),
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        snapshot = await self.metrics_collector.get_snapshot()
        return {
            "node_latencies": snapshot.node_latencies,
            "tool_latencies": snapshot.tool_latencies,
            "cache_hit_rate": f"{snapshot.cache_hit_rate:.1%}",
            "prefetch_hit_rate": f"{snapshot.prefetch_hit_rate:.1%}",
            "tool_timeout_rate": f"{snapshot.tool_timeout_rate:.1%}",
            "avg_tokens_per_turn": snapshot.avg_tokens_per_turn,
            "timestamp": snapshot.timestamp.isoformat(),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global agent instance
_agent: Optional[AmaniQAgent] = None


async def get_agent(config: Optional[AmaniQConfig] = None) -> AmaniQAgent:
    """Get or create global agent instance"""
    global _agent
    if _agent is None:
        _agent = AmaniQAgent(config)
        await _agent.initialize()
    return _agent


async def query_amaniq(
    query: str,
    thread_id: Optional[str] = None,
    config: Optional[AmaniQConfig] = None,
) -> Dict[str, Any]:
    """
    Convenience function to query AmaniQ agent.

    Args:
        query: User's question
        thread_id: Optional thread ID
        config: Optional configuration

    Returns:
        Response dictionary
    """
    agent = await get_agent(config)
    return await agent.chat(message=query, thread_id=thread_id)


async def prewarm_agent():
    """Pre-warm agent caches at startup"""
    try:
        agent = await get_agent()
        await startup_prewarm()
        logger.info("Agent pre-warming complete")
    except Exception as e:
        logger.warning(f"Agent pre-warming failed: {e}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Config
    "AmaniQConfig",
    # State
    "AmaniQState",
    # Agent
    "AmaniQAgent",
    "get_agent",
    # Graph
    "create_amaniq_v2_graph",
    # Convenience
    "query_amaniq",
    "prewarm_agent",
    # Client
    "MoonshotClient",
    # Re-exports from components
    "IntentType",
    "ToolName",
    "SupervisorDecision",
    "ToolStatus",
    "ToolResult",
    "ClarificationStatus",
    "MAX_CLARIFICATION_ROUNDS",
    "TelemetryMetrics",
    "get_metrics_collector",
    "TOP_LEGAL_QUERIES",
]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        print("=" * 80)
        print("AmaniQ v2 Agent - Example Usage")
        print("=" * 80)

        # Check for API key
        if not os.getenv("MOONSHOT_API_KEY"):
            print("\n⚠️  MOONSHOT_API_KEY not set. Please set it to run examples.")
            print("   export MOONSHOT_API_KEY=your-key-here")
            return

        # Initialize agent
        agent = AmaniQAgent()
        await agent.initialize()

        # Example queries
        examples = [
            "What does Article 27 of the Constitution say about equality?",
            "Hello, habari yako!",
            "Tell me about that land case",  # Should trigger clarification
            "What's the latest news on Finance Bill 2024?",
        ]

        for i, query in enumerate(examples, 1):
            print(f"\n{'=' * 80}")
            print(f"Example {i}: {query}")
            print("=" * 80)

            response = await agent.chat(message=query)

            if response.get("type") == "clarification":
                print(f"\n🔄 Clarification Needed:")
                print(f"   Question: {response.get('question')}")
                print(
                    f"   Round: {response.get('clarification_round')}/{response.get('max_rounds')}"
                )
            else:
                print(f"\n✅ Response:")
                print(f"   Intent: {response.get('intent')}")
                print(f"   Confidence: {response.get('confidence', 0):.2f}")
                print(f"   From Cache: {response.get('from_cache')}")
                print(f"   Answer Preview: {response.get('answer', '')[:200]}...")

        # Get metrics
        print(f"\n{'=' * 80}")
        print("Performance Metrics")
        print("=" * 80)
        metrics = await agent.get_metrics()
        for key, value in metrics.items():
            print(f"   {key}: {value}")

    asyncio.run(main())
