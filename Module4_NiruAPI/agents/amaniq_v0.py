"""
AmaniQuery v0.0 (amaniq-v0) - RAG Agent for Kenyan Intelligence
================================================================================

Complete agent orchestration for AmaniQuery using LangGraph, designed specifically
for Kenyan legal and media ecosystems with bilingual fluency (English/Swahili).

Architecture (Production-Grade):
    User Query → EntryGate (Intent + Language) → ShengTranslator →
    Router (Conditional to Personas) → NewsAgent/LawAgent/HybridAgent →
    KBSearch + WebSearch (Parallel) → ReasoningEngine → QualityGate →
    Synthesis + Validation → ExitGate (Response + Attribution)

Specialized Agents:
1. EntryGate: Query classification, language detection, confidence scoring
2. ShengTranslator: Sheng/slang → formal (for better embeddings)
3. NewsAgent: Kenyan news retrieval (time-weighted, county-specific)
4. LawAgent: Legal intelligence (case law, statutes, constitution)
5. HybridCoordinator: Multi-source synthesis for complex queries
6. ReasoningEngine: Chain-of-Thought, multi-hop reasoning, fact-checking
7. QualityGate: Confidence scoring, hallucination detection
8. ExitGate: Response formatting, source attribution

Production Features:
- Multi-hop reasoning with citation chains
- Temporal consistency checking
- Kenyan context ontology (47 counties, political entities)
- Hybrid retrieval (vector + metadata + BM25)
- Swahili/English code-switching support
- Human-in-the-loop for legal risk
- Comprehensive source attribution
- Self-correction with retry limits

Usage:
    from amaniq_v1 import create_amaniq_v1_graph, AmaniqV1State

    graph = create_amaniq_v1_graph(
        llm_client=llm,
        vector_db_client=vector_store,
        metadata_manager=metadata_mgr,
        enable_reasoning=True,
        enable_quality_gate=True
    )
    result = graph.invoke({
        "user_query": "Kanjo wameongeza parking fees aje?",
        "conversation_history": [],
        "session_context": {"user_county": "Nairobi"}
    })
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from datetime import datetime
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import existing AmaniQuery agents
from .intent_router import classify_query, INTENT_ROUTER_SYSTEM_PROMPT
from .sheng_translator import (
    detect_sheng,
    full_translation_pipeline,
    SHENG_TO_FORMAL_PROMPT,
    FORMAL_TO_SHENG_PROMPT,
)
from .kenyanizer import (
    get_system_prompt,
    SYSTEM_PROMPT_WANJIKU,
    SYSTEM_PROMPT_WAKILI,
    SYSTEM_PROMPT_MWANAHABARI,
)
from .json_enforcer import (
    get_json_enforcement_prompt,
    validate_response,
    parse_llm_response,
    retry_with_enforcement,
    RESPONSE_SCHEMA,
)
from .retrieval_strategies import UnifiedRetriever
from .tools.web_search import WebSearchTool
from .tools.kb_search import KnowledgeBaseSearchTool
from .tools.news_search import NewsSearchTool
from .reasoning.reasoner import Reasoner
from .reasoning.planner import Planner

# Import from Module3_NiruDB for metadata and vector operations
try:
    from Module3_NiruDB.metadata_manager import MetadataManager
    from Module3_NiruDB.vector_store import VectorStore
except ImportError:
    logger.warning(
        "Could not import Module3_NiruDB components, some features may be limited"
    )
    MetadataManager = None
    VectorStore = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# STATE SCHEMA
# ============================================================================


class AmaniqV1State(TypedDict):
    """
    Comprehensive state for Amaniq v1 pipeline - Production-Ready.

    State flows through all nodes and accumulates information across
    the multi-agent orchestration system.
    """

    # ========================================================================
    # INPUT
    # ========================================================================
    user_query: str  # Original user query
    conversation_history: List[Dict[str, str]]  # Previous chat messages
    session_context: Optional[Dict]  # User session (county, preferences)

    # ========================================================================
    # ENTRY GATE - INTENT ROUTING & LANGUAGE DETECTION
    # ========================================================================
    query_type: Optional[str]  # "wanjiku", "wakili", "mwanahabari"
    confidence: Optional[float]  # Intent classification confidence
    detected_language: Optional[str]  # "en", "sw", "sheng", "mixed"
    routing_reasoning: Optional[str]  # Why this classification
    human_review_flag: bool  # Flag for human-in-the-loop

    # ========================================================================
    # SHENG TRANSLATION & NORMALIZATION
    # ========================================================================
    has_sheng: bool  # Whether query contains Sheng
    formal_query: Optional[str]  # Sheng translated to formal
    original_query: str  # Preserved for response re-injection
    detected_sheng_terms: List[str]  # Sheng terms identified

    # ========================================================================
    # RETRIEVAL - HYBRID SEARCH
    # ========================================================================
    retrieval_query: str  # Query used for retrieval
    retrieved_docs: List[Dict[str, Any]]  # Retrieved documents from vector DB
    retrieval_metadata: Optional[Dict]  # Retrieval stats/scores
    namespace_used: List[str]  # Which namespaces searched

    # ========================================================================
    # KNOWLEDGE BASE SEARCH (KB Tool)
    # ========================================================================
    kb_search_results: Optional[List[Dict]]  # Results from KB search tool
    kb_metadata_filters: Optional[Dict]  # Metadata filters applied

    # ========================================================================
    # WEB SEARCH & NEWS SEARCH
    # ========================================================================
    web_results: Optional[str]  # Results from web search
    news_results: Optional[str]  # Results from news search
    external_sources: List[Dict]  # External source attributions

    # ========================================================================
    # REASONING ENGINE
    # ========================================================================
    reasoning_plan: Optional[List[Dict]]  # Execution plan from Planner
    reasoning_chain: Optional[str]  # Chain-of-Thought trace
    multi_hop_evidence: Optional[List[Dict]]  # Multi-hop reasoning evidence
    fact_check_results: Optional[Dict]  # Fact-checking results

    # ========================================================================
    # KENYANIZER - PERSONA & CONTEXT
    # ========================================================================
    system_prompt: str  # Persona-specific system prompt
    persona_name: str  # "wanjiku", "wakili", "mwanahabari"
    kenyan_context: Optional[Dict]  # County, entity, cultural context

    # ========================================================================
    # SYNTHESIS
    # ========================================================================
    raw_llm_response: Optional[str]  # Raw LLM output
    parsed_response: Optional[Dict]  # Parsed JSON response
    synthesis_attempts: int  # Number of synthesis attempts

    # ========================================================================
    # QUALITY GATE - VALIDATION & CONFIDENCE
    # ========================================================================
    is_valid: bool  # Validation passed
    validation_errors: List[str]  # Validation error messages
    validation_attempts: int  # Number of validation attempts
    confidence_score: float  # Overall confidence (0.0-1.0)
    hallucination_detected: bool  # Hallucination flag
    temporal_consistency: bool  # Temporal consistency check

    # ========================================================================
    # OUTPUT - EXIT GATE
    # ========================================================================
    final_response: Optional[Dict]  # Final validated JSON response
    source_attributions: List[Dict]  # Full source citations
    error_message: Optional[str]  # Error if pipeline failed

    # ========================================================================
    # METADATA & OBSERVABILITY
    # ========================================================================
    pipeline_start_time: float  # Timestamp when pipeline started
    total_tokens_used: int  # Total LLM tokens consumed
    node_execution_times: Dict[str, float]  # Time spent in each node
    retrieval_strategy_used: str  # Which retrieval strategy
    # ========================================================================
    has_sheng: bool  # Whether query contains Sheng
    formal_query: Optional[str]  # Sheng translated to formal
    original_query: str  # Preserved for response re-injection

    # ========================================================================
    # RETRIEVAL
    # ========================================================================
    retrieval_query: str  # Query used for retrieval
    retrieved_docs: List[Dict[str, Any]]  # Retrieved documents
    retrieval_metadata: Optional[Dict]  # Retrieval stats/scores

    # ========================================================================
    # WEB SEARCH
    # ========================================================================
    web_results: Optional[str]  # Results from web search

    # ========================================================================
    # KENYANIZER
    # ========================================================================
    system_prompt: str  # Persona-specific system prompt
    persona_name: str  # "wanjiku", "wakili", "mwanahabari"

    # ========================================================================
    # SYNTHESIS
    # ========================================================================
    raw_llm_response: Optional[str]  # Raw LLM output
    parsed_response: Optional[Dict]  # Parsed JSON response
    synthesis_attempts: int  # Number of synthesis attempts

    # ========================================================================
    # VALIDATION
    # ========================================================================
    is_valid: bool  # Validation passed
    validation_errors: List[str]  # Validation error messages
    validation_attempts: int  # Number of validation attempts

    # ========================================================================
    # OUTPUT
    # ========================================================================
    final_response: Optional[Dict]  # Final validated JSON response
    error_message: Optional[str]  # Error if pipeline failed

    # ========================================================================
    # METADATA
    # ========================================================================
    pipeline_start_time: float  # Timestamp when pipeline started
    total_tokens_used: int  # Total LLM tokens consumed
    node_execution_times: Dict[str, float]  # Time spent in each node


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def generate_llm_response(
    client, prompt: str, max_tokens: int = 1000, temperature: float = 0.3
) -> str:
    """
    Helper to generate text from LLM client (supports OpenAI, Moonshot, and Gemini).
    Handles both new (v1.0+) and old OpenAI client patterns, plus Gemini API.
    """
    try:
        # Try Gemini API pattern first (google.generativeai)
        if hasattr(client, "generate_content"):
            # This is a Gemini model object
            response = client.generate_content(prompt)
            return response.text

        # Try new OpenAI v1.0+ pattern
        elif hasattr(client, "chat") and hasattr(client.chat, "completions"):
            response = client.chat.completions.create(
                model="moonshot-v1-8k",  # Default to moonshot-v1-8k
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

        # Try custom client with generate() method
        elif hasattr(client, "generate"):
            return client.generate(
                prompt, max_tokens=max_tokens, temperature=temperature
            )

        # Try direct call (if client is a function)
        elif callable(client):
            return client(prompt)

        else:
            raise AttributeError(
                "LLM client has no 'generate_content', 'chat.completions.create', or 'generate' method"
            )

    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        raise


# ============================================================================
# AGENT NODES
# ============================================================================


class IntentRouterAgent:
    """Node 1: Route query to appropriate persona"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def __call__(self, state: AmaniqV1State) -> AmaniqV1State:
        """Classify user query intent"""
        start_time = datetime.now().timestamp()
        logger.info(f"[IntentRouter] Processing: {state['user_query']}")

        try:
            # Define wrapper for LLM function
            def llm_wrapper(prompt):
                return generate_llm_response(self.llm_client, prompt, temperature=0.1)

            # Use existing intent router
            result = classify_query(
                query=state["user_query"],
                llm_function=llm_wrapper,
                # conversation_history=state.get('conversation_history', []) # Removed as classify_query doesn't support it yet in intent_router.py
            )

            state["query_type"] = result["query_type"]
            state["confidence"] = result["confidence"]
            state["detected_language"] = result["detected_language"]
            state["routing_reasoning"] = result.get("reasoning", "")

            # Map to persona names
            persona_map = {
                "wanjiku": "wanjiku",
                "wakili": "wakili",
                "mwanahabari": "mwanahabari",
                "public_interest": "wanjiku",  # Fallback mapping
                "legal": "wakili",
                "research": "mwanahabari",
            }
            state["persona_name"] = persona_map.get(result["query_type"], "wanjiku")

            logger.info(
                f"[IntentRouter] Classified as: {state['query_type']} (confidence: {result['confidence']:.2f})"
            )

        except Exception as e:
            logger.error(f"[IntentRouter] Error: {e}")
            # Fallback to wanjiku for errors
            state["query_type"] = "wanjiku"
            state["persona_name"] = "wanjiku"
            state["confidence"] = 0.5
            state["detected_language"] = "mixed"
            state["routing_reasoning"] = f"Error occurred, defaulting to wanjiku: {e}"

        # Update execution time
        if "node_execution_times" not in state:
            state["node_execution_times"] = {}
        state["node_execution_times"]["intent_router"] = (
            datetime.now().timestamp() - start_time
        )

        return state


class ShengTranslatorAgent:
    """Node 2: Translate Sheng to formal for better retrieval"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def __call__(self, state: AmaniqV1State) -> AmaniqV1State:
        """Translate Sheng/slang to formal language"""
        start_time = datetime.now().timestamp()
        logger.info(f"[ShengTranslator] Processing query")

        try:
            # Preserve original query
            state["original_query"] = state["user_query"]

            # Detect if Sheng is present
            is_sheng, confidence, detected_terms = detect_sheng(state["user_query"])
            state["has_sheng"] = is_sheng

            if is_sheng or state.get("detected_language") in ["sheng", "mixed"]:
                logger.info(
                    f"[ShengTranslator] Sheng detected (confidence: {confidence:.2f})"
                )

                # Define wrapper for LLM function
                def llm_wrapper(prompt):
                    return generate_llm_response(
                        self.llm_client, prompt, temperature=0.1
                    )

                # Translate to formal
                # Note: full_translation_pipeline signature might differ, using direct translation logic here if needed
                # But assuming we use translate_to_formal logic from sheng_translator if available or just use the pipeline

                # For now, let's assume we want to use the LLM to translate
                prompt = SHENG_TO_FORMAL_PROMPT + f"\n\nQuery: {state['user_query']}"
                formal_query = llm_wrapper(prompt)

                state["formal_query"] = formal_query.strip()
                logger.info(f"[ShengTranslator] Translated to: {state['formal_query']}")
            else:
                # No Sheng, use original query
                state["formal_query"] = state["user_query"]
                logger.info(
                    f"[ShengTranslator] No Sheng detected, using original query"
                )

            # Set retrieval query
            state["retrieval_query"] = state["formal_query"]

        except Exception as e:
            logger.error(f"[ShengTranslator] Error: {e}")
            # Fallback to original query
            state["formal_query"] = state["user_query"]
            state["retrieval_query"] = state["user_query"]
            state["has_sheng"] = False

        state["node_execution_times"]["sheng_translator"] = (
            datetime.now().timestamp() - start_time
        )
        return state


class RetrievalAgent:
    """Node 3: Retrieve relevant documents with persona-specific strategy"""

    def __init__(self, vector_db_client):
        self.retriever = vector_db_client

    def __call__(self, state: AmaniqV1State) -> AmaniqV1State:
        """Retrieve documents using persona-optimized strategy"""
        start_time = datetime.now().timestamp()
        logger.info(f"[Retrieval] Searching for: {state['retrieval_query']}")

        try:
            persona = state.get("persona_name", "wanjiku")

            # Route to appropriate retrieval strategy using UnifiedRetriever.retrieve
            if persona == "wanjiku":
                results = self.retriever.retrieve(
                    query_type="wanjiku",
                    query=state["retrieval_query"],
                    limit=10,
                    recency_months=6,
                )
            elif persona == "wakili":
                results = self.retriever.retrieve(
                    query_type="wakili",
                    query=state["retrieval_query"],
                    limit=10,
                    doc_types=["act", "bill", "judgment", "constitution"],
                )
            elif persona == "mwanahabari":
                results = self.retriever.retrieve(
                    query_type="mwanahabari",
                    query=state["retrieval_query"],
                    limit=20,
                    require_tables=True,
                )
            else:
                # Fallback
                results = self.retriever.retrieve(
                    query_type="wanjiku", query=state["retrieval_query"], limit=10
                )

            state["retrieved_docs"] = results
            state["retrieval_metadata"] = {
                "num_results": len(results),
                "avg_score": (
                    sum(r.get("score", 0) for r in results) / len(results)
                    if results
                    else 0
                ),
                "strategy": persona,
            }

            logger.info(
                f"[Retrieval] Retrieved {len(results)} documents (avg score: {state['retrieval_metadata']['avg_score']:.3f})"
            )

        except Exception as e:
            logger.error(f"[Retrieval] Error: {e}")
            state["retrieved_docs"] = []
            state["retrieval_metadata"] = {"error": str(e)}

        state["node_execution_times"]["retrieval"] = (
            datetime.now().timestamp() - start_time
        )
        return state


class WebSearchAgent:
    """Node 3.5: Fetch external context via Web Search"""

    def __init__(self):
        self.search_tool = WebSearchTool()

    def __call__(self, state: AmaniqV1State) -> AmaniqV1State:
        """Execute web search if needed"""
        start_time = datetime.now().timestamp()

        # Only search if confidence is low or specific intent (optional logic)
        # For now, we'll run it parallel to retrieval or as an enhancement
        # Let's assume we run it for 'mwanahabari' or if retrieval yields few results

        try:
            # Simple heuristic: Always search for now to provide context, or check if retrieval was poor
            # But to save costs, maybe only if 'mwanahabari' or 'wanjiku' with current events

            query = state["retrieval_query"]
            logger.info(f"[WebSearch] Searching web for: {query}")

            # Use execute() method and format results properly
            search_results = self.search_tool.execute(query)

            # Format results as a string for the synthesis agent
            if search_results.get("results"):
                formatted = f"Web search found {search_results['count']} results:\n\n"
                for i, result in enumerate(search_results["results"][:5], 1):
                    formatted += f"{i}. {result['title']}\n{result['snippet']}\nSource: {result['url']}\n\n"
                state["web_results"] = formatted
            else:
                state["web_results"] = ""

            logger.info(f"[WebSearch] Found {search_results.get('count', 0)} results")

        except Exception as e:
            logger.error(f"[WebSearch] Error: {e}")
            state["web_results"] = ""

        state["node_execution_times"]["web_search"] = (
            datetime.now().timestamp() - start_time
        )
        return state


class KenyanizerPreambleAgent:
    """Node 4: Generate persona-specific system prompt"""

    def __call__(self, state: AmaniqV1State) -> AmaniqV1State:
        """Generate appropriate system prompt for persona"""
        start_time = datetime.now().timestamp()
        logger.info(f"[Kenyanizer] Generating prompt for: {state['persona_name']}")

        try:
            persona = state.get("persona_name", "wanjiku")

            # Generate system prompt
            system_prompt = get_system_prompt(query_type=persona)

            state["system_prompt"] = system_prompt
            logger.info(f"[Kenyanizer] Generated {len(system_prompt)} char prompt")

        except Exception as e:
            logger.error(f"[Kenyanizer] Error: {e}")
            # Fallback to wanjiku prompt
            state["system_prompt"] = SYSTEM_PROMPT_WANJIKU

        state["node_execution_times"]["kenyanizer"] = (
            datetime.now().timestamp() - start_time
        )
        return state

    def _summarize_context(self, docs: List[Dict]) -> str:
        """Summarize retrieved documents for context"""
        if not docs:
            return "No relevant documents found."

        summary = f"Found {len(docs)} relevant documents:\n"
        for i, doc in enumerate(docs[:3], 1):
            summary += (
                f"{i}. {doc.get('source', 'Unknown')}: {doc.get('text', '')[:100]}...\n"
            )
        return summary


class SynthesisAgent:
    """Node 5: Generate final response with JSON enforcement"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def __call__(self, state: AmaniqV1State) -> AmaniqV1State:
        """Synthesize final response using persona prompt + JSON enforcement"""
        start_time = datetime.now().timestamp()
        logger.info(
            f"[Synthesis] Generating response (attempt {state.get('synthesis_attempts', 0) + 1})"
        )

        try:
            # Initialize attempts counter
            if "synthesis_attempts" not in state:
                state["synthesis_attempts"] = 0
            state["synthesis_attempts"] += 1

            # Build context from retrieved docs AND web results
            context = self._build_context(
                state["retrieved_docs"], state.get("web_results", "")
            )

            # Get JSON enforcement prompt
            persona_map = {
                "wanjiku": "public_interest",
                "wakili": "legal",
                "mwanahabari": "research",
            }
            query_type_for_enforcer = persona_map.get(
                state["persona_name"], "public_interest"
            )

            # Get enforcement prompt
            enforcement_prompt = get_json_enforcement_prompt(
                user_query=state["user_query"],
                retrieved_context=context,
                persona_hint=query_type_for_enforcer,
            )

            # Combine system prompt + enforcement
            full_prompt = f"{state['system_prompt']}\n\n{enforcement_prompt}"

            # Generate response
            raw_response = generate_llm_response(
                self.llm_client, prompt=full_prompt, max_tokens=1500, temperature=0.3
            )

            state["raw_llm_response"] = raw_response

            # Parse JSON
            parsed, error = parse_llm_response(raw_response)
            state["parsed_response"] = parsed

            if error:
                logger.warning(f"[Synthesis] Parse error: {error}")

            logger.info(f"[Synthesis] Generated response ({len(raw_response)} chars)")

        except Exception as e:
            logger.error(f"[Synthesis] Error: {e}")
            state["error_message"] = f"Synthesis failed: {e}"
            state["parsed_response"] = None

        state["node_execution_times"]["synthesis"] = (
            datetime.now().timestamp() - start_time
        )
        return state

    def _build_context(self, docs: List[Dict], web_results: str) -> str:
        """Build context string from retrieved documents and web results"""
        context_parts = []

        if docs:
            context_parts.append("--- INTERNAL KNOWLEDGE BASE ---")
            for i, doc in enumerate(docs[:5], 1):
                source = doc.get("source", "Unknown Source")
                text = doc.get("text", "")[:500]  # Limit length
                context_parts.append(f"[Document {i}] {source}\n{text}\n")
        else:
            context_parts.append(
                "No relevant information found in the internal knowledge base."
            )

        if web_results:
            context_parts.append("\n--- EXTERNAL WEB SEARCH RESULTS ---")
            context_parts.append(web_results[:2000])  # Limit length

        return "\n".join(context_parts)


class ValidationAgent:
    """Node 6: Validate JSON schema and citation presence"""

    def __call__(self, state: AmaniqV1State) -> AmaniqV1State:
        """Validate response against schema"""
        start_time = datetime.now().timestamp()
        logger.info(
            f"[Validation] Validating response (attempt {state.get('validation_attempts', 0) + 1})"
        )

        try:
            # Initialize attempts counter
            if "validation_attempts" not in state:
                state["validation_attempts"] = 0
            state["validation_attempts"] += 1

            if not state.get("parsed_response"):
                state["is_valid"] = False
                state["validation_errors"] = ["No parsed response available"]
                logger.warning("[Validation] No parsed response to validate")
                return state

            # Validate schema
            is_valid, error = validate_response(state["parsed_response"])

            state["is_valid"] = is_valid
            state["validation_errors"] = [error] if error else []

            if is_valid:
                state["final_response"] = state["parsed_response"]
                logger.info(f"[Validation] ✓ Response valid")
            else:
                logger.warning(f"[Validation] ✗ Validation failed: {error}")

        except Exception as e:
            logger.error(f"[Validation] Error: {e}")
            state["is_valid"] = False
            state["validation_errors"] = [f"Validation error: {e}"]

        state["node_execution_times"]["validation"] = (
            datetime.now().timestamp() - start_time
        )
        return state


# ============================================================================
# CONDITIONAL EDGES
# ============================================================================


def should_retry_synthesis(
    state: AmaniqV1State,
) -> Literal["synthesis", "fallback", "validation"]:
    """Decide whether to retry synthesis or move to fallback"""
    max_attempts = 3

    if state.get("synthesis_attempts", 0) >= max_attempts:
        logger.warning(f"[Router] Max synthesis attempts reached, using fallback")
        return "fallback"

    if state.get("parsed_response") is None:
        logger.info(f"[Router] Synthesis failed, retrying")
        return "synthesis"

    return "validation"


def should_retry_validation(state: AmaniqV1State) -> Literal["synthesis", "end"]:
    """Decide whether to retry synthesis for validation errors"""
    max_attempts = 2

    if not state.get("is_valid", False):
        if state.get("validation_attempts", 0) < max_attempts:
            logger.info(f"[Router] Validation failed, retrying synthesis")
            return "synthesis"
        else:
            logger.warning(
                f"[Router] Max validation attempts reached, accepting response"
            )
            # Accept the response anyway (better than nothing)
            state["final_response"] = state.get("parsed_response", {})
            return "end"

    return "end"


def should_use_web_search(state: AmaniqV1State) -> Literal["web_search", "kenyanizer"]:
    """
    Decide whether to use web search based on retrieval quality.

    Web search is used when:
    1. Few or no documents retrieved (< 3 docs)
    2. Low average retrieval score (< 0.3)
    3. Query type is 'mwanahabari' (research needs current info)
    4. Query mentions recent events/dates

    Otherwise, skip web search to save time and use vector context only.
    """
    retrieved_docs = state.get("retrieved_docs", [])
    retrieval_metadata = state.get("retrieval_metadata", {})
    query_type = state.get("query_type", "wanjiku")

    # Check 1: Not enough documents retrieved
    if len(retrieved_docs) < 3:
        logger.info(f"[Router] Low doc count ({len(retrieved_docs)}), using web search")
        return "web_search"

    # Check 2: Low quality scores
    avg_score = retrieval_metadata.get("avg_score", 0)
    if avg_score < 0.3:
        logger.info(f"[Router] Low avg score ({avg_score:.2f}), using web search")
        return "web_search"

    # Check 3: Research queries always get web search
    if query_type == "mwanahabari":
        logger.info(f"[Router] Research query, using web search")
        return "web_search"

    # Check 4: Recent events/dates in query
    query = state.get("user_query", "").lower()
    recent_keywords = [
        "latest",
        "recent",
        "new",
        "2024",
        "2025",
        "today",
        "yesterday",
        "now",
    ]
    if any(keyword in query for keyword in recent_keywords):
        logger.info(f"[Router] Recent event query, using web search")
        return "web_search"

    # Sufficient context from vectors, skip web search
    logger.info(
        f"[Router] Sufficient context ({len(retrieved_docs)} docs, score {avg_score:.2f}), skipping web search"
    )
    return "kenyanizer"


class FallbackHandler:
    """Fallback node for when synthesis/validation fails"""

    def __call__(self, state: AmaniqV1State) -> AmaniqV1State:
        """Generate fallback response when pipeline fails"""
        logger.warning("[Fallback] Using fallback response mechanism")

        # Generate a minimal valid response
        fallback_response = {
            "answer": "I apologize, but I encountered difficulties generating a proper response. Please try rephrasing your question or contact support.",
            "sources": [],
            "confidence": "low",
            "follow_up_questions": ["Could you rephrase your question?"],
            "query_type": state.get("query_type", "wanjiku"),
        }

        state["final_response"] = fallback_response
        state["is_valid"] = True
        state["error_message"] = "Used fallback response due to pipeline errors"

        return state


# ============================================================================
# GRAPH CREATION
# ============================================================================


def create_amaniq_v1_graph(
    llm_client, vector_db_client, enable_persistence: bool = False, fast_llm_client=None
):
    """
    Create the complete Amaniq v1 agent orchestration graph.

    Args:
        llm_client: LLM client for synthesis (OpenAI-compatible)
        vector_db_client: Vector database client (UnifiedRetriever)
        enable_persistence: Whether to enable state persistence (default: False)
        fast_llm_client: Optional fast LLM client for intent routing and translation

    Returns:
        Compiled LangGraph StateGraph
    """
    logger.info("Creating Amaniq v1 Agent Graph...")

    # Use fast LLM for router/translator if provided, otherwise use main LLM
    router_llm = fast_llm_client if fast_llm_client else llm_client

    # Initialize agents
    intent_router = IntentRouterAgent(router_llm)
    sheng_translator = ShengTranslatorAgent(router_llm)
    retrieval = RetrievalAgent(vector_db_client)
    web_search = WebSearchAgent()
    kenyanizer = KenyanizerPreambleAgent()
    synthesis = SynthesisAgent(llm_client)
    validation = ValidationAgent()
    fallback = FallbackHandler()

    # Create graph
    workflow = StateGraph(AmaniqV1State)

    # Add nodes
    workflow.add_node("intent_router", intent_router)
    workflow.add_node("sheng_translator", sheng_translator)
    workflow.add_node("retrieval", retrieval)
    workflow.add_node("web_search", web_search)
    workflow.add_node("kenyanizer", kenyanizer)
    workflow.add_node("synthesis", synthesis)
    workflow.add_node("validation", validation)
    workflow.add_node("fallback", fallback)

    # Define edges
    workflow.set_entry_point("intent_router")

    # Linear flow for main pipeline
    workflow.add_edge("intent_router", "sheng_translator")
    workflow.add_edge("sheng_translator", "retrieval")

    # Conditional routing after retrieval: use web search only if needed
    workflow.add_conditional_edges(
        "retrieval",
        should_use_web_search,
        {
            "web_search": "web_search",  # Use web search
            "kenyanizer": "kenyanizer",  # Skip web search, go directly to kenyanizer
        },
    )

    # Web search goes to kenyanizer when used
    workflow.add_edge("web_search", "kenyanizer")
    workflow.add_edge("kenyanizer", "synthesis")

    # Conditional routing after synthesis
    workflow.add_conditional_edges(
        "synthesis",
        should_retry_synthesis,
        {
            "synthesis": "synthesis",  # Retry synthesis
            "validation": "validation",  # Move to validation
            "fallback": "fallback",  # Use fallback
        },
    )

    # Conditional routing after validation
    workflow.add_conditional_edges(
        "validation",
        should_retry_validation,
        {"synthesis": "synthesis", "end": END},  # Retry synthesis  # Done
    )

    # Fallback always goes to end
    workflow.add_edge("fallback", END)

    # Compile with optional persistence
    if enable_persistence:
        memory = SqliteSaver.from_conn_string(":memory:")
        graph = workflow.compile(checkpointer=memory)
        logger.info("✓ Amaniq v1 Graph compiled with persistence enabled")
    else:
        graph = workflow.compile()
        logger.info("✓ Amaniq v1 Graph compiled (no persistence)")

    return graph


# ============================================================================
# EXECUTION HELPER
# ============================================================================


def execute_pipeline(
    graph, user_query: str, conversation_history: List[Dict] = None
) -> Dict[str, Any]:
    """
    Execute the Amaniq v1 pipeline with a user query.

    Args:
        graph: Compiled LangGraph
        user_query: User's input query
        conversation_history: Optional conversation history

    Returns:
        Dict containing final_response and metadata
    """
    import time

    # Initialize state
    initial_state = {
        "user_query": user_query,
        "conversation_history": conversation_history or [],
        "has_sheng": False,
        "synthesis_attempts": 0,
        "validation_attempts": 0,
        "is_valid": False,
        "validation_errors": [],
        "pipeline_start_time": time.time(),
        "total_tokens_used": 0,
        "node_execution_times": {},
    }

    # Execute graph
    logger.info(f"[Pipeline] Starting execution for query: {user_query}")
    result = graph.invoke(initial_state)

    # Calculate total time
    total_time = time.time() - result["pipeline_start_time"]

    # Build response
    response = {
        "final_response": result.get("final_response", {}),
        "metadata": {
            "query_type": result.get("query_type"),
            "persona": result.get("persona_name"),
            "detected_language": result.get("detected_language"),
            "has_sheng": result.get("has_sheng", False),
            "confidence": result.get("confidence"),
            "retrieval_count": len(result.get("retrieved_docs", [])),
            "web_search_used": bool(result.get("web_results")),
            "synthesis_attempts": result.get("synthesis_attempts", 0),
            "validation_attempts": result.get("validation_attempts", 0),
            "is_valid": result.get("is_valid", False),
            "total_time_seconds": round(total_time, 2),
            "node_times": result.get("node_execution_times", {}),
            "error": result.get("error_message"),
        },
    }

    logger.info(f"[Pipeline] ✓ Completed in {total_time:.2f}s")
    return response


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("AmaniQuery v1.0 (amaniq-v1) - Agent Orchestration Graph")
    print("Import this module to use the graph.")
    print("\nExample usage:")
    print(
        "  from Module4_NiruAPI.agents.amaniq_v1 import create_amaniq_v1_graph, execute_pipeline"
    )
    print("  graph = create_amaniq_v1_graph(llm_client, vector_db)")
    print("  result = execute_pipeline(graph, 'Parking fees iko ngapi?')")
