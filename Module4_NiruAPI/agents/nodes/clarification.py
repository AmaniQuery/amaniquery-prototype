"""
AmaniQ v2 Clarification Node - Human-in-the-Loop Cycle
======================================================

This module implements the clarification sub-graph for when the Supervisor
outputs "CLARIFY" intent. It pauses the graph, asks the user a specific
question, and resumes from the same state when they reply.

Features:
- Max 3 clarification rounds before forcing final answer
- Short, specific questions (max 2 sentences)
- State preservation across pause/resume
- Automatic escalation after max rounds

Author: Eng. Onyango Benard
Version: 2.0
"""

from typing import Dict, Any, List, Optional, Literal, TypedDict, Annotated
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from loguru import logger

import operator

# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_CLARIFICATION_ROUNDS = 3


# =============================================================================
# STATE DEFINITIONS
# =============================================================================


class ClarificationState(TypedDict, total=False):
    """State for the clarification sub-graph"""

    # Original context (preserved across rounds)
    original_question: str
    thread_id: str

    # Clarification tracking
    clarification_count: int
    clarification_history: List[Dict[str, str]]  # [{question: ..., answer: ...}]

    # Current clarification
    current_clarification_question: str
    waiting_for_user: bool

    # Supervisor decision that triggered clarification
    supervisor_decision: Dict[str, Any]

    # Accumulated context from partial understanding
    accumulated_context: str

    # Resolution
    clarification_resolved: bool
    final_enriched_query: str

    # Messages for LangGraph
    messages: Annotated[List[Dict[str, Any]], operator.add]


class ClarificationStatus(str, Enum):
    """Status of the clarification process"""

    PENDING = "pending"  # Waiting for user response
    ANSWERED = "answered"  # User has responded
    MAX_ROUNDS = "max_rounds"  # Hit max clarification limit
    RESOLVED = "resolved"  # Clarification complete


# =============================================================================
# CLARIFICATION NODE FUNCTIONS
# =============================================================================


def clarification_entry_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry node when Supervisor outputs CLARIFY intent.
    Extracts the clarification question and prepares to pause.

    This node:
    1. Increments clarification counter
    2. Extracts the specific question from supervisor decision
    3. Sets waiting_for_user = True to pause graph

    Args:
        state: Current graph state with supervisor_decision

    Returns:
        Updated state with clarification question
    """
    supervisor_decision = state.get("supervisor_decision", {})
    clarification = supervisor_decision.get("clarification", {})

    # Get or initialize clarification count
    clarification_count = state.get("clarification_count", 0) + 1
    clarification_history = state.get("clarification_history", [])

    # Extract clarification details
    clarification_question = clarification.get(
        "clarification_question",
        "Could you please provide more details about your question?",
    )
    partial_understanding = clarification.get("partial_understanding", "")
    missing_fields = clarification.get("missing_fields", [])

    # Ensure question is short (max 2 sentences)
    clarification_question = _truncate_to_two_sentences(clarification_question)

    # Build accumulated context from partial understanding
    accumulated = state.get("accumulated_context", "")
    if partial_understanding and partial_understanding not in accumulated:
        accumulated = f"{accumulated}\n{partial_understanding}".strip()

    logger.info(
        f"Clarification round {clarification_count}/{MAX_CLARIFICATION_ROUNDS}: "
        f"Missing fields: {missing_fields}"
    )

    # Prepare the assistant message for the user
    assistant_message = {
        "role": "assistant",
        "content": clarification_question,
        "metadata": {
            "type": "clarification",
            "round": clarification_count,
            "missing_fields": missing_fields,
        },
    }

    return {
        **state,
        "clarification_count": clarification_count,
        "clarification_history": clarification_history,
        "current_clarification_question": clarification_question,
        "waiting_for_user": True,  # This pauses the graph
        "accumulated_context": accumulated,
        "clarification_resolved": False,
        "messages": [assistant_message],
    }


def clarification_resume_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resume node after user provides clarification.

    This node:
    1. Captures the user's response
    2. Adds to clarification history
    3. Builds enriched query for re-routing

    Args:
        state: State with user's clarification response

    Returns:
        Updated state ready for supervisor re-routing
    """
    messages = state.get("messages", [])
    clarification_history = state.get("clarification_history", [])

    # Get the last user message (their clarification response)
    user_response = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_response = msg.get("content", "")
            break

    if not user_response:
        logger.warning("No user response found in clarification resume")
        user_response = "[No response provided]"

    # Add to history
    clarification_history = clarification_history.copy()
    clarification_history.append(
        {
            "question": state.get("current_clarification_question", ""),
            "answer": user_response,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    # Build enriched query combining original + clarifications
    original_question = state.get("original_question", "")
    accumulated_context = state.get("accumulated_context", "")

    # Create enriched query for supervisor
    enriched_parts = [original_question]
    for clarif in clarification_history:
        enriched_parts.append(f"[Clarification: {clarif['answer']}]")

    final_enriched_query = " ".join(enriched_parts)

    logger.info(
        f"Clarification received. Enriched query length: {len(final_enriched_query)}"
    )

    return {
        **state,
        "clarification_history": clarification_history,
        "waiting_for_user": False,  # Resume the graph
        "clarification_resolved": True,
        "final_enriched_query": final_enriched_query,
        "current_query": final_enriched_query,  # For supervisor re-routing
    }


def max_clarification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node triggered when max clarification rounds reached.
    Forces a final answer with whatever context we have.

    Args:
        state: Current state after max rounds

    Returns:
        State forcing tool execution with partial context
    """
    clarification_history = state.get("clarification_history", [])
    original_question = state.get("original_question", "")
    accumulated_context = state.get("accumulated_context", "")

    # Build the best query we can from available context
    query_parts = [original_question]
    for clarif in clarification_history:
        if clarif.get("answer"):
            query_parts.append(clarif["answer"])

    best_effort_query = " ".join(query_parts)

    logger.warning(
        f"Max clarification rounds ({MAX_CLARIFICATION_ROUNDS}) reached. "
        f"Proceeding with best-effort query."
    )

    # Create a message explaining we'll do our best
    assistant_message = {
        "role": "assistant",
        "content": (
            "Sawa, let me try to help with the information I have. "
            "I may not have all the details, but I'll do my best to answer."
        ),
        "metadata": {
            "type": "max_clarification_reached",
            "rounds_used": MAX_CLARIFICATION_ROUNDS,
        },
    }

    # Override supervisor decision to force tool execution
    forced_decision = {
        "intent": "LEGAL_RESEARCH",  # Default to research
        "confidence": 0.6,  # Lower confidence due to incomplete info
        "reasoning": f"Forced after {MAX_CLARIFICATION_ROUNDS} clarification rounds",
        "tool_plan": [
            {"tool_name": "kb_search", "query": best_effort_query, "priority": 1}
        ],
        "detected_language": state.get("supervisor_decision", {}).get(
            "detected_language", "en"
        ),
        "detected_entities": [],
        "token_count": len(best_effort_query.split()) * 2,
        "context_overflow": False,
    }

    return {
        **state,
        "waiting_for_user": False,
        "clarification_resolved": True,
        "final_enriched_query": best_effort_query,
        "current_query": best_effort_query,
        "supervisor_decision": forced_decision,
        "messages": [assistant_message],
        "max_clarification_reached": True,
    }


# =============================================================================
# CONDITIONAL EDGE FUNCTIONS
# =============================================================================


def should_clarify(state: Dict[str, Any]) -> Literal["clarify", "continue", "respond"]:
    """
    Conditional edge after Supervisor node.
    Determines if we need clarification or can proceed.

    Args:
        state: Current graph state with supervisor_decision

    Returns:
        "clarify" - Need more info from user
        "continue" - Proceed to tool execution
        "respond" - Direct response (no tools needed)
    """
    supervisor_decision = state.get("supervisor_decision", {})
    intent = supervisor_decision.get("intent", "")

    if intent == "CLARIFY":
        return "clarify"
    elif intent in ("LEGAL_RESEARCH", "NEWS_SUMMARY"):
        return "continue"
    elif intent == "GENERAL_CHAT":
        return "respond"
    elif intent == "ESCALATE":
        return "respond"
    else:
        # Default to continue if unknown
        return "continue"


def clarification_route(
    state: Dict[str, Any],
) -> Literal["wait", "resume", "max_reached", "done"]:
    """
    Conditional edge within clarification sub-graph.

    Args:
        state: Current clarification state

    Returns:
        "wait" - Waiting for user input (pause graph)
        "resume" - User responded, resume processing
        "max_reached" - Hit max rounds, force answer
        "done" - Clarification complete
    """
    waiting = state.get("waiting_for_user", False)
    clarification_count = state.get("clarification_count", 0)
    resolved = state.get("clarification_resolved", False)

    # Check if max rounds reached
    if clarification_count >= MAX_CLARIFICATION_ROUNDS and not resolved:
        return "max_reached"

    # Check if waiting for user
    if waiting:
        return "wait"

    # Check if resolved
    if resolved:
        return "done"

    # User has responded, resume
    return "resume"


def after_clarification_route(
    state: Dict[str, Any],
) -> Literal["supervisor", "tools", "respond"]:
    """
    Route after clarification is resolved.

    Args:
        state: State after clarification

    Returns:
        "supervisor" - Re-run supervisor with enriched query
        "tools" - Go directly to tools (max rounds case)
        "respond" - Direct response
    """
    max_reached = state.get("max_clarification_reached", False)

    if max_reached:
        # Skip supervisor, go straight to tools
        return "tools"

    # Re-run supervisor with enriched query
    return "supervisor"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _truncate_to_two_sentences(text: str) -> str:
    """
    Truncate text to maximum 2 sentences.

    Args:
        text: Input text

    Returns:
        Text with at most 2 sentences
    """
    import re

    # Split by sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    if len(sentences) <= 2:
        return text.strip()

    # Take first 2 sentences
    return " ".join(sentences[:2])


def build_clarification_question(
    missing_fields: List[str], partial_understanding: str, language: str = "en"
) -> str:
    """
    Build a natural clarification question.

    Args:
        missing_fields: What information is missing
        partial_understanding: What we understood so far
        language: "en", "sw", or "mixed"

    Returns:
        Natural question in appropriate language
    """
    # Field-specific question templates
    field_questions = {
        "case_name": {
            "en": "Which specific case are you asking about?",
            "sw": "Ni kesi gani hasa unayouliza?",
        },
        "year": {
            "en": "What year or time period are you interested in?",
            "sw": "Mwaka gani au kipindi gani?",
        },
        "parties": {
            "en": "Who are the parties involved?",
            "sw": "Ni nani wahusika?",
        },
        "court": {
            "en": "Which court handled this matter?",
            "sw": "Mahakama gani ilishughulikia hili?",
        },
        "act_name": {
            "en": "Which specific Act or law are you referring to?",
            "sw": "Ni sheria gani hasa unayorejelea?",
        },
        "section": {
            "en": "Which section or article specifically?",
            "sw": "Sehemu gani hasa?",
        },
        "county": {
            "en": "Which county or location?",
            "sw": "Kaunti gani au eneo gani?",
        },
        "amount": {
            "en": "What amount or value is involved?",
            "sw": "Kiasi gani?",
        },
    }

    # Select language
    lang_key = "sw" if language == "sw" else "en"

    # Build question from missing fields
    questions = []
    for field in missing_fields[:2]:  # Max 2 fields to keep it short
        field_lower = field.lower().replace(" ", "_")
        if field_lower in field_questions:
            questions.append(field_questions[field_lower][lang_key])
        else:
            # Generic question for unknown field
            if lang_key == "sw":
                questions.append(f"Tafadhali eleza zaidi kuhusu {field}?")
            else:
                questions.append(f"Could you specify the {field.replace('_', ' ')}?")

    # Combine and add context
    if partial_understanding and lang_key == "en":
        prefix = f"I understand you're asking about {partial_understanding}. "
    elif partial_understanding and lang_key == "sw":
        prefix = f"Naelewa unaomba kuhusu {partial_understanding}. "
    else:
        prefix = ""

    return prefix + " ".join(questions)


# =============================================================================
# GRAPH BUILDER HELPER
# =============================================================================


def add_clarification_subgraph(graph_builder):
    """
    Add the clarification sub-graph to a LangGraph StateGraph.

    Usage:
        from langgraph.graph import StateGraph

        graph = StateGraph(AgentState)
        # ... add other nodes ...

        # Add clarification handling
        add_clarification_subgraph(graph)

        # Add conditional edge from supervisor
        graph.add_conditional_edges(
            "supervisor",
            should_clarify,
            {
                "clarify": "clarification_entry",
                "continue": "tool_executor",
                "respond": "responder",
            }
        )

    Args:
        graph_builder: LangGraph StateGraph builder
    """
    # Add clarification nodes
    graph_builder.add_node("clarification_entry", clarification_entry_node)
    graph_builder.add_node("clarification_resume", clarification_resume_node)
    graph_builder.add_node("max_clarification", max_clarification_node)

    # Add edges within clarification sub-graph
    graph_builder.add_conditional_edges(
        "clarification_entry",
        clarification_route,
        {
            "wait": "__interrupt__",  # Pause for user input
            "resume": "clarification_resume",
            "max_reached": "max_clarification",
            "done": "supervisor",
        },
    )

    # After user responds (graph resumes at clarification_resume)
    graph_builder.add_conditional_edges(
        "clarification_resume",
        after_clarification_route,
        {
            "supervisor": "supervisor",  # Re-route with enriched query
            "tools": "tool_executor",
            "respond": "responder",
        },
    )

    # Max clarification goes directly to tools
    graph_builder.add_edge("max_clarification", "tool_executor")

    logger.info("Clarification sub-graph added to graph builder")


# =============================================================================
# EXAMPLE GRAPH STRUCTURE
# =============================================================================

"""
COMPLETE GRAPH STRUCTURE WITH CLARIFICATION
============================================

                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ SUPERVISOR  │◄────────────────────┐
                    └──────┬──────┘                     │
                           │                            │
            should_clarify() │                          │
         ┌─────────┬───────┴────────┐                  │
         │         │                │                   │
         ▼         ▼                ▼                   │
    ┌────────┐ ┌────────┐    ┌───────────┐             │
    │CLARIFY │ │CONTINUE│    │  RESPOND  │             │
    └────┬───┘ └────┬───┘    └─────┬─────┘             │
         │         │               │                    │
         ▼         │               │                    │
┌─────────────────┐│               │                    │
│ CLARIFICATION   ││               │                    │
│     ENTRY       ││               │                    │
└────────┬────────┘│               │                    │
         │         │               │                    │
    clarification_route()          │                    │
    ┌────┴────┬───────┐            │                    │
    │         │       │            │                    │
    ▼         ▼       ▼            │                    │
┌──────┐ ┌───────┐ ┌────────┐      │                    │
│ WAIT │ │RESUME │ │MAX_HIT │      │                    │
│(pause)│ └───┬───┘ └───┬────┘     │                    │
└──┬───┘     │         │           │                    │
   │         │         │           │                    │
   │ (user   │         │           │                    │
   │ responds)│        │           │                    │
   │         │         │           │                    │
   └────►┌───┴───┐     │           │                    │
         │CLARIF │     │           │                    │
         │RESUME │     │           │                    │
         └───┬───┘     │           │                    │
             │         │           │                    │
             │ after_clarification_route()              │
             │         │           │                    │
     ┌───────┴───┐     │           │                    │
     │           │     │           │                    │
     ▼           ▼     ▼           │                    │
┌──────────┐ ┌──────────┐          │                    │
│SUPERVISOR│ │  TOOLS   │◄─────────┘                    │
│(re-route)│ └────┬─────┘                               │
└────┬─────┘      │                                     │
     │            │                                     │
     └────────────┼────────────────┐                    │
                  │                │                    │
                  ▼                ▼                    │
            ┌───────────┐   ┌───────────┐              │
            │ RESPONDER │   │ RESPONDER │              │
            └─────┬─────┘   └─────┬─────┘              │
                  │               │                    │
                  ▼               ▼                    │
            ┌───────────────────────┐                  │
            │         END           │                  │
            └───────────────────────┘


STATE FLOW DURING CLARIFICATION
================================

1. User: "Tell me about that land case"
   
2. Supervisor → intent: CLARIFY
   - missing_fields: ["case_name", "year", "parties"]
   - partial_understanding: "land dispute case"

3. clarification_entry_node:
   - clarification_count: 1
   - waiting_for_user: True
   - current_clarification_question: "Which specific land case? Jina la kesi ni gani?"
   
4. Graph PAUSES (returns to user)

5. User: "The one in Kiambu about the Kamau family"

6. Graph RESUMES at clarification_resume_node:
   - waiting_for_user: False
   - clarification_history: [{"question": "...", "answer": "Kiambu, Kamau family"}]
   - final_enriched_query: "Tell me about that land case [Clarification: Kiambu, Kamau family]"

7. → Back to Supervisor with enriched query

8. Supervisor → intent: LEGAL_RESEARCH (now has enough info)
   - tool_plan: [{tool: "kb_search", query: "Kamau family land case Kiambu"}]

9. → tool_executor → responder → END
"""


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # State
    "ClarificationState",
    "ClarificationStatus",
    "MAX_CLARIFICATION_ROUNDS",
    # Node functions
    "clarification_entry_node",
    "clarification_resume_node",
    "max_clarification_node",
    # Routing functions
    "should_clarify",
    "clarification_route",
    "after_clarification_route",
    # Helpers
    "build_clarification_question",
    "add_clarification_subgraph",
]
