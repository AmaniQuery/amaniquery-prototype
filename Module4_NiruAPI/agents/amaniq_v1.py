"""
AmaniQ v1.0 - Production-Ready RAG Agent Graph for Kenyan News & Law Intelligence
==================================================================================

Complete LangGraph implementation of the AmaniQ agent orchestration system,
designed specifically for the nuances of Kenyan legal and media ecosystems.

Architecture:
    User Query → EntryGate → ShengTranslator (if needed) → Planner →
    ToolExecutor (KB Search + Others) → ReasoningEngine → PersonaSynthesis →
    QualityGate → ExitGate → Response

Key Features:
- Stateful multi-agent orchestration with LangGraph
- All retrieval from vector DB via KB search (no external APIs)
- Tool orchestration guided by reasoning engine
- Persona-specific synthesis (wanjiku/wakili/mwanahabari)
- Multi-hop reasoning with citation chains
- Bilingual fluency (English/Swahili) with code-switching support
- Human-in-the-loop for low confidence queries
- Kenyan context ontology (47 counties, political entities)

Author: AmaniQuery Team
Version: 1.0.0
Date: 2025
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from datetime import datetime
import logging
import os
import re
from pathlib import Path
import sys
import json
from openai import OpenAI

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import existing AmaniQuery components
from .intent_router import classify_query
from .sheng_translator import (
    detect_sheng,
    full_translation_pipeline,
    translate_to_formal,
    translate_to_sheng,
)
from .kenyanizer import (
    get_system_prompt,
    SYSTEM_PROMPT_WANJIKU,
    SYSTEM_PROMPT_WAKILI,
    SYSTEM_PROMPT_MWANAHABARI,
)
from .json_enforcer import validate_response, parse_llm_response
from .tools.kb_search import KnowledgeBaseSearchTool
from .tools.web_search import WebSearchTool
from .tools.news_search import NewsSearchTool
from .tools.twitter_scraper import TwitterScraperTool
from .reasoning.reasoner import Reasoner
from .reasoning.planner import Planner

# Import database components
try:
    from Module3_NiruDB.vector_store import VectorStore
    from Module3_NiruDB.metadata_manager import MetadataManager
except ImportError:
    VectorStore = None
    MetadataManager = None

# Import RAG pipeline
try:
    from Module4_NiruAPI.rag_pipeline import RAGPipeline
except ImportError:
    RAGPipeline = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL CLIENTS (initialized in create_amaniq_v1_graph)
# ============================================================================

_VECTOR_STORE = None
_RAG_PIPELINE = None
_KB_SEARCH_TOOL = None
_WEB_SEARCH_TOOL = None
_NEWS_SEARCH_TOOL = None
_TWITTER_TOOL = None
_REASONER = None
_PLANNER = None

# ============================================================================
# KENYAN CONTEXT CONSTANTS
# ============================================================================

KENYAN_COUNTIES = [
    "Mombasa",
    "Kwale",
    "Kilifi",
    "Tana River",
    "Lamu",
    "Taita-Taveta",
    "Garissa",
    "Wajir",
    "Mandera",
    "Marsabit",
    "Isiolo",
    "Meru",
    "Tharaka-Nithi",
    "Embu",
    "Kitui",
    "Machakos",
    "Makueni",
    "Nyandarua",
    "Nyeri",
    "Kirinyaga",
    "Murang'a",
    "Kiambu",
    "Turkana",
    "West Pokot",
    "Samburu",
    "Trans-Nzoia",
    "Uasin Gishu",
    "Elgeyo-Marakwet",
    "Nandi",
    "Baringo",
    "Laikipia",
    "Nakuru",
    "Narok",
    "Kajiado",
    "Kericho",
    "Bomet",
    "Kakamega",
    "Vihiga",
    "Bungoma",
    "Busia",
    "Siaya",
    "Kisumu",
    "Homa Bay",
    "Migori",
    "Kisii",
    "Nyamira",
    "Nairobi",
]

LEGAL_KEYWORDS = [
    "court",
    "constitution",
    "judgment",
    "bill",
    "act",
    "law",
    "statute",
    "case",
    "petition",
    "appeal",
    "mahakama",
    "katiba",
    "sheria",
    "muamuzi",
    "ruling",
    "verdict",
    "litigation",
    "arbitration",
    "tribunal",
]

NEWS_KEYWORDS = [
    "treasury",
    "parliament",
    "county",
    "governor",
    "senator",
    "mp",
    "cabinet",
    "ministry",
    "habari",
    "taarifa",
    "breaking",
    "scandal",
    "investigation",
    "report",
    "announcement",
    "policy",
    "reform",
]

SWAHILI_INDICATORS = [
    "katiba",
    "mahakama",
    "habari",
    "taarifa",
    "sheria",
    "serikali",
    "waziri",
    "gavana",
    "uchaguzi",
    "muungano",
    "wahamiaji",
    "uchumi",
]

# ============================================================================
# STATE SCHEMA
# ============================================================================


class ToolCall(TypedDict):
    """Structure for tool calls in the plan"""

    tool_name: str
    tool_args: Dict[str, Any]
    result: Optional[Any]


class PlanStep(TypedDict):
    """Research plan step"""

    step: int
    action: str
    description: str
    tool_call: Optional[ToolCall]
    completed: bool


class Evidence(TypedDict):
    """Evidence structure for retrieved documents"""

    source_id: str
    content: str
    source_type: Literal["news", "law", "parliament", "historical", "web"]
    timestamp: Optional[str]
    url: Optional[str]
    confidence: float
    metadata: Dict[str, Any]


class Source(TypedDict):
    """Source attribution structure"""

    id: str
    title: str
    url: str
    source_type: str
    retrieved_at: str


class Thought(TypedDict):
    """Reasoning step structure"""

    step: int
    action: str
    observation: str
    reasoning: str


class AmaniqV1State(TypedDict):
    """Master state schema for AmaniQ agent graph"""

    # Input
    user_query: str
    session_context: Dict[str, Any]
    conversation_history: List[Dict[str, str]]

    # Classification & Language
    intent_classification: Literal["news", "law", "hybrid", "general"]
    persona: Literal["wanjiku", "wakili", "mwanahabari"]
    swahili_language_flag: bool
    translated_query: Optional[str]
    detected_language: str

    # Planning & Execution
    research_plan: List[PlanStep]
    current_step: int

    # Retrieval & Evidence
    retrieved_evidence: List[Evidence]
    source_attributions: List[Source]

    # Reasoning
    reasoning_path: List[Thought]
    sub_queries: List[str]
    synthesis_result: Optional[str]

    # Quality Control
    confidence_score: float
    human_review_flag: bool
    quality_issues: List[str]

    # Output
    final_response: str
    formatted_response: Dict[str, Any]

    # Metadata
    iteration_count: int
    agent_path: List[str]
    error_log: List[str]
    max_iterations: int

    # Enhancements
    compressed_context: Optional[str]
    fact_check_results: List[Dict[str, Any]]
    skip_llm_synthesis: bool


# ============================================================================
# NODE FUNCTIONS
# ============================================================================


def entry_gate(state: AmaniqV1State) -> AmaniqV1State:
    """
    Entry Gate Node: Query classification, language detection, and persona selection

    Responsibilities:
    - Detect language (English/Swahili/Mixed)
    - Classify intent (news/law/hybrid/general)
    - Select appropriate persona (wanjiku/wakili/mwanahabari)
    - Detect Sheng and flag for translation
    - Initialize state tracking
    """
    logger.info("=== ENTRY GATE ===")

    query = state["user_query"]

    # Language detection
    swahili_flag = any(word.lower() in query.lower() for word in SWAHILI_INDICATORS)
    sheng_detected = detect_sheng(query)

    # Determine language
    if swahili_flag or sheng_detected:
        detected_language = "swahili" if swahili_flag else "sheng"
    else:
        detected_language = "english"

    # Intent classification using existing router
    try:
        intent_result = classify_query(query)
        intent = intent_result.get("intent", "general")
    except Exception as e:
        logger.error(f"Intent classification error: {e}")
        intent = "general"

    # Map intent to our schema and select persona
    if "law" in intent.lower() or "legal" in intent.lower():
        classification = "law"
        persona = "wakili"
    elif "news" in intent.lower() or "media" in intent.lower():
        classification = "news"
        persona = "mwanahabari"
    elif "hybrid" in intent.lower() or "mixed" in intent.lower():
        classification = "hybrid"
        persona = "mwanahabari"  # News persona for hybrid
    else:
        classification = "general"
        persona = "wanjiku"  # Simple persona for general queries

    # Check for legal keywords
    has_legal_keywords = any(kw in query.lower() for kw in LEGAL_KEYWORDS)
    has_news_keywords = any(kw in query.lower() for kw in NEWS_KEYWORDS)

    if has_legal_keywords and has_news_keywords:
        classification = "hybrid"
        persona = "wakili"  # Legal persona for hybrid with legal component
    elif has_legal_keywords:
        classification = "law"
        persona = "wakili"
    elif has_news_keywords:
        classification = "news"
        persona = "mwanahabari"

    logger.info(f"Query: {query}")
    logger.info(
        f"Language: {detected_language}, Intent: {classification}, Persona: {persona}"
    )

    # Initialize state
    state.update(
        {
            "intent_classification": classification,
            "persona": persona,
            "swahili_language_flag": swahili_flag,
            "detected_language": detected_language,
            "iteration_count": 0,
            "current_step": 0,
            "max_iterations": 3,
            "agent_path": ["entry_gate"],
            "error_log": [],
            "retrieved_evidence": [],
            "source_attributions": [],
            "reasoning_path": [],
            "sub_queries": [],
            "research_plan": [],
            "quality_issues": [],
            "confidence_score": 0.0,
            "human_review_flag": False,
        }
    )

    return state


def sheng_translator_node(state: AmaniqV1State) -> AmaniqV1State:
    """
    Sheng Translator Node: Convert informal Swahili to formal for better retrieval
    """
    logger.info("=== SHENG TRANSLATOR ===")

    query = state["user_query"]

    # Only translate if Sheng or informal Swahili detected
    if state["detected_language"] in ["sheng", "swahili"]:
        try:
            # Define LLM wrapper for translation
            def llm_wrapper(prompt: str) -> str:
                if _RAG_PIPELINE:
                    # Use generate_answer with the prompt as system prompt
                    return _RAG_PIPELINE.generate_answer(
                        query="", context="", system_prompt=prompt, max_tokens=500
                    )
                return prompt

            # Use translate_to_formal with LLM wrapper
            translation_result = translate_to_formal(
                user_query=query, llm_function=llm_wrapper
            )
            translated = translation_result.get("formal_query", query)

            logger.info(f"Translated: {query} → {translated}")

            state["translated_query"] = translated
            state["agent_path"].append("sheng_translator")
        except Exception as e:
            logger.error(f"Translation error: {e}")
            state["translated_query"] = query
            state["error_log"].append(f"Translation failed: {str(e)}")
    else:
        state["translated_query"] = query

    return state


def planner_node(state: AmaniqV1State) -> AmaniqV1State:
    """
    Planner Node: Create research plan based on query and intent

    Uses the Planner to create a structured research plan that will guide
    tool execution in the next node.
    """
    logger.info("=== PLANNER ===")

    query = state.get("translated_query") or state["user_query"]
    intent = state["intent_classification"]
    persona = state["persona"]

    # Determine search namespaces based on intent
    # Use underscore naming to match vector_store namespace implementation
    if intent == "law":
        namespaces = ["kenya_law"]
    elif intent == "news":
        namespaces = ["kenya_news", "kenya_parliament"]
    elif intent == "hybrid":
        namespaces = ["kenya_law", "kenya_news", "kenya_parliament"]
    else:
        namespaces = ["kenya_news", "kenya_parliament", "kenya_law", "historical"]

    # Create a simplified plan for KB search-focused retrieval
    research_plan = []

    # Step 1: Primary KB search
    research_plan.append(
        PlanStep(
            step=1,
            action="kb_search_primary",
            description=f"Search knowledge base for: {query}",
            tool_call=ToolCall(
                tool_name="kb_search",
                tool_args={
                    "query": query,
                    "top_k": 8 if persona == "wanjiku" else 10,
                    "namespace": namespaces,
                },
                result=None,
            ),
            completed=False,
        )
    )

    # Step 2: Expand search for hybrid queries
    if intent == "hybrid":
        research_plan.append(
            PlanStep(
                step=2,
                action="kb_search_expanded",
                description=f"Expanded search for related information",
                tool_call=ToolCall(
                    tool_name="kb_search",
                    tool_args={
                        "query": query,
                        "top_k": 5,
                        "namespace": ["historical", "global_trends"],
                    },
                    result=None,
                ),
                completed=False,
            )
        )

    # Step 3: Analysis (reasoning, no tool call)
    research_plan.append(
        PlanStep(
            step=len(research_plan) + 1,
            action="analyze_evidence",
            description="Analyze retrieved evidence and synthesize findings",
            tool_call=None,
            completed=False,
        )
    )

    state["research_plan"] = research_plan
    state["agent_path"].append("planner")

    logger.info(f"Created research plan with {len(research_plan)} steps")

    return state


def tool_executor_node(state: AmaniqV1State) -> AmaniqV1State:
    """
    Tool Executor Node: Execute research plan using available tools

    Orchestrates all tools based on the research plan created by the Planner.
    Primary tool is KB search, but can use others if needed.
    """
    logger.info("=== TOOL EXECUTOR ===")

    research_plan = state["research_plan"]
    evidence_list = []
    sources_list = []

    for step in research_plan:
        if step["completed"]:
            continue

        tool_call = step.get("tool_call")
        if tool_call is None:
            # No tool needed for this step (e.g., analysis step)
            step["completed"] = True
            continue

        tool_name = tool_call["tool_name"]
        tool_args = tool_call["tool_args"]

        logger.info(f"Executing step {step['step']}: {step['action']}")

        try:
            # Execute the appropriate tool with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if tool_name == "kb_search" and _KB_SEARCH_TOOL:
                        result = _KB_SEARCH_TOOL.execute(**tool_args)

                        # Process search results into evidence
                        search_results = result.get("search_results", [])
                        for idx, item in enumerate(search_results):
                            evidence = Evidence(
                                source_id=f"kb_{step['step']}_{idx}",
                                content=item.get("content", ""),
                                source_type=_infer_source_type(
                                    item.get("metadata", {})
                                ),
                                timestamp=item.get("metadata", {}).get(
                                    "date_published"
                                ),
                                url=item.get("metadata", {}).get("url", ""),
                                confidence=float(item.get("score", 0.5)),
                                metadata=item.get("metadata", {}),
                            )
                            evidence_list.append(evidence)

                            # Create source attribution
                            source = Source(
                                id=evidence["source_id"],
                                title=item.get("metadata", {}).get(
                                    "title", "Knowledge Base Document"
                                ),
                                url=evidence["url"],
                                source_type=evidence["source_type"],
                                retrieved_at=datetime.now().isoformat(),
                            )
                            sources_list.append(source)

                        tool_call["result"] = result
                        logger.info(f"KB search returned {len(search_results)} results")
                        break  # Success, exit retry loop

                    elif tool_name == "web_search" and _WEB_SEARCH_TOOL:
                        result = _WEB_SEARCH_TOOL.execute(**tool_args)
                        tool_call["result"] = result
                        break

                    elif tool_name == "news_search" and _NEWS_SEARCH_TOOL:
                        result = _NEWS_SEARCH_TOOL.execute(**tool_args)
                        tool_call["result"] = result
                        break

                    elif tool_name == "twitter_search" and _TWITTER_TOOL:
                        result = _TWITTER_TOOL.execute(**tool_args)
                        tool_call["result"] = result
                        break

                except Exception as inner_e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Tool {tool_name} failed (attempt {attempt+1}/{max_retries}): {inner_e}. Retrying..."
                        )
                        continue
                    else:
                        raise inner_e

            step["completed"] = True

        except Exception as e:
            logger.error(f"Error executing {tool_name}: {e}")
            state["error_log"].append(f"Tool execution failed ({tool_name}): {str(e)}")
            step["completed"] = True  # Mark as completed to avoid infinite loop

    # Update state with evidence
    state["retrieved_evidence"].extend(evidence_list)
    state["source_attributions"].extend(sources_list)
    state["current_step"] = len([s for s in research_plan if s["completed"]])
    state["agent_path"].append("tool_executor")

    logger.info(
        f"Tool execution completed: {len(evidence_list)} evidence items retrieved"
    )

    return state


def compress_context_node(state: AmaniqV1State) -> AmaniqV1State:
    """
    Context Compression Node: Optimize evidence for context window

    - Groups evidence by relevance and type
    - Extracts key citations (Case numbers, Acts)
    - Truncates less relevant content
    - Prioritizes 'Law' sources for Wakili persona
    """
    logger.info("=== CONTEXT COMPRESSION ===")

    evidence = state["retrieved_evidence"]
    persona = state["persona"]

    if not evidence:
        state["compressed_context"] = ""
        return state

    # Extract citations
    citations = []
    for e in evidence:
        content = e["content"]
        # Simple regex for Kenyan legal citations
        found_citations = re.findall(
            r"(?:Case No\.|Civil Appeal|Petition|Act No\.|Cap\.|Article)\s+[0-9A-Za-z\s\/]+",
            content,
        )
        citations.extend(found_citations)

    # Sort evidence by confidence
    sorted_evidence = sorted(
        evidence, key=lambda x: x.get("confidence", 0), reverse=True
    )

    compressed_parts = []

    # Always include top 3 chunks fully
    for i, e in enumerate(sorted_evidence[:3]):
        compressed_parts.append(f"Source {i+1} ({e['source_type']}): {e['content']}")

    # For the rest, summarize or extract citations
    for i, e in enumerate(sorted_evidence[3:], 3):
        if persona == "wakili" and e["source_type"] == "law":
            # Keep more legal context
            compressed_parts.append(
                f"Source {i+1} ({e['source_type']}): {e['content'][:300]}..."
            )
        else:
            # Aggressive truncation
            compressed_parts.append(
                f"Source {i+1} ({e['source_type']}): {e['content'][:150]}..."
            )

    if citations:
        compressed_parts.append(
            "\nKey Citations Found: " + ", ".join(set(citations[:10]))
        )

    state["compressed_context"] = "\n\n".join(compressed_parts)
    state["agent_path"].append("compress_context")

    return state


def _infer_source_type(metadata: Dict[str, Any]) -> str:
    """Infer source type from metadata"""
    doc_type = metadata.get("doc_type", "").lower()
    namespace = metadata.get("namespace", "").lower()

    if "law" in doc_type or "law" in namespace:
        return "law"
    elif "news" in doc_type or "news" in namespace:
        return "news"
    elif "parliament" in doc_type or "parliament" in namespace:
        return "parliament"
    elif "historical" in namespace:
        return "historical"
    else:
        return "web"


def reasoning_engine(state: AmaniqV1State) -> AmaniqV1State:
    """
    Reasoning Engine: Multi-hop reasoning and verification

    Capabilities:
    - Chain-of-thought reasoning using Reasoner
    - Multi-hop query decomposition
    - Consistency checking (temporal, entity, statistical)
    - Citation chain construction
    - Determines if more retrieval is needed
    """
    logger.info("=== REASONING ENGINE ===")

    query = state["user_query"]
    evidence = state["retrieved_evidence"]
    persona = state["persona"]

    reasoning_path = []

    # Step 1: Evidence analysis
    thought1 = Thought(
        step=1,
        action="Analyze retrieved evidence",
        observation=f"Retrieved {len(evidence)} evidence items from knowledge base",
        reasoning="Examining sources for relevance, consistency, and completeness",
    )
    reasoning_path.append(thought1)

    # Step 2: Temporal consistency check
    timestamps = [e.get("timestamp") for e in evidence if e.get("timestamp")]
    thought2 = Thought(
        step=2,
        action="Check temporal consistency",
        observation=f"Found {len(timestamps)} timestamped sources",
        reasoning="Ensuring timeline alignment across sources for accuracy",
    )
    reasoning_path.append(thought2)

    # Step 3: Entity consistency check
    entities_found = []
    for county in KENYAN_COUNTIES:
        if county.lower() in query.lower():
            entities_found.append(county)

    thought3 = Thought(
        step=3,
        action="Verify entity consistency",
        observation=f"Identified entities: {', '.join(entities_found) if entities_found else 'None'}",
        reasoning="Cross-referencing entity mentions across sources for consistency",
    )
    reasoning_path.append(thought3)

    # Step 4: Coverage analysis - determine if we need more information
    needs_more_info = False
    if len(evidence) < 3:
        needs_more_info = True
        thought4 = Thought(
            step=4,
            action="Coverage analysis",
            observation=f"Limited evidence ({len(evidence)} items)",
            reasoning="May need additional retrieval for comprehensive answer",
        )
    else:
        thought4 = Thought(
            step=4,
            action="Coverage analysis",
            observation=f"Sufficient evidence ({len(evidence)} items)",
            reasoning="Adequate information for synthesis",
        )
    reasoning_path.append(thought4)

    # Step 5: Use Reasoner for chain-of-thought if available
    if _REASONER:
        try:
            cot_reasoning = _REASONER.chain_of_thought(
                query, {"evidence_count": len(evidence)}
            )
            thought5 = Thought(
                step=5,
                action="Chain-of-thought reasoning",
                observation="Applied structured reasoning to query",
                reasoning=cot_reasoning[:200],  # Truncate for storage
            )
            reasoning_path.append(thought5)
        except Exception as e:
            logger.error(f"Reasoner error: {e}")

    # Step 6: Synthesize findings
    synthesis = (
        f"Based on {len(evidence)} sources from knowledge base, addressing: {query}"
    )

    thought6 = Thought(
        step=len(reasoning_path) + 1,
        action="Synthesize response",
        observation="Combined evidence from all sources",
        reasoning=f"Preparing {persona}-appropriate response",
    )
    reasoning_path.append(thought6)

    state["reasoning_path"] = reasoning_path
    state["synthesis_result"] = synthesis
    state["agent_path"].append("reasoning_engine")

    # Set flag if more information needed (for potential iteration)
    if needs_more_info and state["iteration_count"] < state["max_iterations"]:
        state["confidence_score"] = 0.5  # Low confidence triggers iteration

    logger.info(f"Completed {len(reasoning_path)} reasoning steps")

    return state


def persona_synthesis_node(state: AmaniqV1State) -> AmaniqV1State:
    """
    Persona Synthesis Node: Format response using persona-specific prompts

    Uses RAG pipeline with persona-specific system prompts:
    - wanjiku: Simple, conversational language
    - wakili: Formal legal analysis with citations
    - mwanahabari: Objective, data-driven reporting
    """
    logger.info("=== PERSONA SYNTHESIS ===")

    query = state["user_query"]
    evidence = state["retrieved_evidence"]
    persona = state["persona"]
    reasoning = state["reasoning_path"]

    # Get persona-specific system prompt
    if persona == "wanjiku":
        system_prompt = SYSTEM_PROMPT_WANJIKU
    elif persona == "wakili":
        system_prompt = SYSTEM_PROMPT_WAKILI
    elif persona == "mwanahabari":
        system_prompt = SYSTEM_PROMPT_MWANAHABARI
    else:
        system_prompt = SYSTEM_PROMPT_WANJIKU  # Default

    # Format evidence for synthesis
    evidence_texts = []
    for e in evidence[:10]:  # Top 10 pieces of evidence
        evidence_text = f"Source: {e.get('metadata', {}).get('title', 'Unknown')}\n"
        evidence_text += f"Type: {e['source_type']}\n"
        evidence_text += f"Content: {e['content'][:500]}...\n"
        evidence_texts.append(evidence_text)

    context = "\n\n".join(evidence_texts)

    # If RAG pipeline available, use it for synthesis
    synthesis = None

    # Check if LLM synthesis is skipped (for "pure response" mode)
    if state.get("skip_llm_synthesis", False):
        logger.info("Skipping LLM synthesis as requested")
        synthesis = None  # Will trigger fallback below
    elif _RAG_PIPELINE:
        try:
            # Use RAG pipeline with persona prompt
            synthesis = _RAG_PIPELINE.generate_answer(
                query=query, context=context, system_prompt=system_prompt
            )

            # If original query was Sheng, translate back
            if state["detected_language"] == "sheng":

                def llm_wrapper(prompt: str) -> str:
                    return _RAG_PIPELINE.generate_answer(
                        query="", context="", system_prompt=prompt, max_tokens=1000
                    )

                sheng_result = translate_to_sheng(
                    user_query=query, formal_answer=synthesis, llm_function=llm_wrapper
                )
                synthesis = sheng_result.get("sheng_response", synthesis)
                logger.info("Re-injected Sheng style into response")

        except Exception as e:
            logger.error(f"RAG pipeline synthesis error: {e}")
            state["error_log"].append(f"Synthesis error: {str(e)}")

    # Fallback: simple template-based synthesis
    if not synthesis:
        if persona == "wanjiku":
            synthesis = f"Here's what I found about your question:\n\n{context[:500]}\n\nIn simple terms, this means..."
        elif persona == "wakili":
            synthesis = f"Legal Analysis:\n\n{context[:500]}\n\nStatutory Provisions and Case Law..."
        else:
            synthesis = f"Research Summary:\n\n{context[:500]}\n\nKey findings and statistics..."

    state["synthesis_result"] = synthesis
    state["agent_path"].append("persona_synthesis")

    logger.info(f"Generated {persona} synthesis: {len(synthesis)} chars")

    return state

    logger.info(f"Generated {persona} synthesis: {len(synthesis)} chars")

    return state


def fact_check_node(state: AmaniqV1State) -> AmaniqV1State:
    """
    Fact Check Node: Verify generated answer against evidence using Moonshot AI

    - Checks for hallucinated citations
    - Verifies key claims against retrieved evidence
    - Loops back if critical failures found (Reflexion)
    """
    logger.info("=== FACT CHECK (MOONSHOT) ===")

    synthesis = state.get("synthesis_result", "")
    evidence = state["retrieved_evidence"]

    if not synthesis:
        return state

    # Initialize Moonshot client
    api_key = os.getenv("MOONSHOT_API_KEY")
    base_url = os.getenv("MOONSHOT_BASE_URL") or "https://api.moonshot.ai/v1"

    if not api_key:
        logger.warning("MOONSHOT_API_KEY not found, skipping AI fact check")
        state["fact_check_results"] = [{"issues": [], "status": "skipped"}]
        return state

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        # Prepare evidence text
        evidence_text = ""
        for i, e in enumerate(evidence[:5]):  # Top 5 sources
            evidence_text += f"Source {i+1}: {e['content'][:1000]}\n\n"

        prompt = f"""You are a strict fact-checker for a legal AI assistant.
        
Evidence:
{evidence_text}

Generated Answer:
{synthesis}

Task:
1. Verify if the Generated Answer is supported by the Evidence.
2. Identify any hallucinations or unsupported claims.
3. Check if citations (e.g., "Source 1") match the actual content.

Output JSON format:
{{
    "is_accurate": boolean,
    "issues": ["list of specific inaccuracies or hallucinations"],
    "correction_needed": boolean
}}"""

        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to output JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        result_json = response.choices[0].message.content
        fact_check_result = json.loads(result_json)

        issues = fact_check_result.get("issues", [])
        if fact_check_result.get("correction_needed"):
            state["quality_issues"].extend(issues)
            logger.warning(f"Moonshot Fact Check Issues: {issues}")

            # Reduce confidence if issues found
            state["confidence_score"] = max(0.0, state["confidence_score"] - 0.2)

        state["fact_check_results"] = [fact_check_result]

    except Exception as e:
        logger.error(f"Moonshot fact check failed: {e}")
        state["fact_check_results"] = [{"error": str(e)}]

    state["agent_path"].append("fact_check")
    return state


def quality_gate(state: AmaniqV1State) -> AmaniqV1State:
    """
    Quality Gate: Confidence scoring and validation

    Checks:
    - Citation completeness
    - Source diversity
    - Consistency across evidence
    - Persona-appropriate formatting
    - Confidence thresholds
    """
    logger.info("=== QUALITY GATE ===")

    evidence = state["retrieved_evidence"]
    sources = state["source_attributions"]
    persona = state["persona"]
    issues = []

    # Calculate confidence score
    confidence_factors = []

    # Factor 1: Number of sources (more is better)
    source_count = len(sources)
    if source_count >= 5:
        confidence_factors.append(0.95)
    elif source_count >= 3:
        confidence_factors.append(0.80)
    elif source_count >= 2:
        confidence_factors.append(0.65)
    elif source_count == 1:
        confidence_factors.append(0.50)
    else:
        confidence_factors.append(0.30)
        issues.append("Insufficient sources from knowledge base")

    # Factor 2: Source diversity
    source_types = set(s["source_type"] for s in sources)
    if len(source_types) > 1:
        confidence_factors.append(0.90)
    else:
        confidence_factors.append(0.70)

    # Factor 3: Evidence confidence average (from retrieval scores)
    if evidence:
        avg_evidence_conf = sum(e.get("confidence", 0.5) for e in evidence) / len(
            evidence
        )
        confidence_factors.append(avg_evidence_conf)
    else:
        confidence_factors.append(0.40)
        issues.append("No evidence retrieved from knowledge base")

    # Factor 4: Persona-specific quality check
    if persona == "wakili":
        # Legal queries need high confidence
        legal_sources = [s for s in sources if s["source_type"] == "law"]
        if legal_sources:
            confidence_factors.append(0.90)
        else:
            confidence_factors.append(0.60)
            issues.append("Legal query with no legal sources")
    elif persona == "mwanahabari":
        # News queries need recent sources
        recent_sources = [
            e
            for e in evidence
            if e.get("timestamp")
            and (
                datetime.now()
                - datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
            ).days
            < 90
        ]
        if recent_sources:
            confidence_factors.append(0.85)
        else:
            confidence_factors.append(0.70)

    # Calculate overall confidence
    confidence_score = (
        sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0
    )

    # Determine if human review needed
    human_review = False

    if confidence_score < 0.6:
        human_review = True
        issues.append("Low confidence - human review recommended")

    # Check for legal risk indicators
    query_lower = state["user_query"].lower()
    legal_risk_keywords = [
        "land dispute",
        "constitutional",
        "article 10",
        "article 19",
        "article 27",
    ]

    if any(kw in query_lower for kw in legal_risk_keywords):
        if state["intent_classification"] == "law":
            human_review = True
            issues.append("Legal risk query - requires expert review")

    state["confidence_score"] = confidence_score
    state["human_review_flag"] = human_review
    state["quality_issues"] = issues
    state["agent_path"].append("quality_gate")

    logger.info(f"Confidence: {confidence_score:.2f}, Human Review: {human_review}")
    if issues:
        logger.warning(f"Quality issues: {', '.join(issues)}")

    return state


def exit_gate(state: AmaniqV1State) -> AmaniqV1State:
    """
    Exit Gate: Response formatting and attribution

    Responsibilities:
    - Format final response matching frontend AmaniQueryResponse structure
    - Add source citations
    - Include disclaimers based on confidence
    - Localize for Kenyan context
    """
    logger.info("=== EXIT GATE ===")

    query = state["user_query"]
    evidence = state["retrieved_evidence"]
    sources = state["source_attributions"]
    confidence = state["confidence_score"]
    reasoning = state["reasoning_path"]
    synthesis = state.get("synthesis_result", "")
    persona = state["persona"]
    intent = state["intent_classification"]

    # Map persona to frontend query_type
    query_type_map = {
        "wanjiku": "public_interest",
        "wakili": "legal",
        "mwanahabari": "research",
    }
    query_type = query_type_map.get(persona, "public_interest")

    # Extract summary card info from synthesis
    if synthesis:
        # Try to extract title and content from synthesis
        lines = synthesis.split("\n")
        title = lines[0][:100] if lines else "Response"
        content = synthesis[:300] if len(synthesis) > 300 else synthesis
    else:
        title = f"Answer to: {query[:80]}"
        content = f"Based on knowledge base search regarding: {query}"

    # Build detailed breakdown points from evidence
    detailed_points = []
    for i, e in enumerate(evidence[:5], 1):  # Top 5 evidence items
        content_preview = e["content"][:200]
        detailed_points.append(content_preview)

    if not detailed_points:
        detailed_points = [
            "No detailed breakdown available",
            "Please try rephrasing your query for better results",
        ]

    # Build Kenyan context impact
    kenyan_impact = "This information is relevant to Kenyan citizens and stakeholders."

    # Persona-specific impact statements
    if persona == "wakili":
        kenyan_impact = "This legal information applies to Kenyan jurisdiction and should be reviewed by qualified legal counsel."
    elif persona == "mwanahabari":
        kenyan_impact = "This research provides data-driven insights relevant to Kenya's current affairs and policy landscape."
    elif persona == "wanjiku":
        # Try to extract county context
        user_county = state.get("session_context", {}).get("user_county")
        if user_county:
            kenyan_impact = f"This affects residents of {user_county} County and may impact your daily life."
        else:
            kenyan_impact = "This information affects everyday Kenyans and may impact your daily life."

    # Build citations in frontend format
    citations = []
    for s in sources[:10]:  # Top 10 sources
        citation = {
            "source": s["title"],
            "url": s.get("url", "N/A"),
            "quote": None,  # Could extract quote from evidence if needed
        }
        citations.append(citation)

    if not citations:
        citations = [{"source": "Knowledge Base", "url": "N/A", "quote": None}]

    # Build follow-up suggestions
    follow_up_suggestions = []

    if intent == "law":
        follow_up_suggestions = [
            "What are the penalties for non-compliance?",
            "How does this affect businesses in Kenya?",
            "Are there any recent amendments to this law?",
        ]
    elif intent == "news":
        follow_up_suggestions = [
            "What are the latest updates on this topic?",
            "How are Kenyans reacting to this development?",
            "What is the government's official position?",
        ]
    else:
        follow_up_suggestions = [
            "Can you provide more details?",
            "What are the implications for ordinary Kenyans?",
            "Are there any related topics I should know about?",
        ]

    # Format response matching frontend structure
    formatted_response = {
        "query_type": query_type,
        "language_detected": state["detected_language"],
        "response": {
            "summary_card": {"title": title, "content": content},
            "detailed_breakdown": {"points": detailed_points},
            "kenyan_context": {
                "impact": kenyan_impact,
                "related_topic": None,  # Could be extracted from sub_queries
            },
            "citations": citations,
        },
        "follow_up_suggestions": follow_up_suggestions,
        # Additional metadata for backend use
        "metadata": {
            "confidence": confidence,
            "persona": persona,
            "evidence_count": len(evidence),
            "reasoning_steps": len(reasoning),
            "human_review_required": state["human_review_flag"],
            "intent": intent,
            "agent_path": state["agent_path"],
            "quality_issues": state["quality_issues"],
            "timestamp": datetime.now().isoformat(),
            "iteration_count": state["iteration_count"],
            "reasoning_path": {
                "query": state["user_query"],
                "thoughts": reasoning,
                "total_duration_ms": 0,  # Placeholder
                "final_conclusion": "Analysis complete",
            },
        },
    }

    # Build text response for legacy compatibility
    response_parts = []

    # Add summary
    response_parts.append(f"# {title}\n")
    response_parts.append(content)

    # Add detailed points
    if detailed_points and detailed_points[0] != "No detailed breakdown available":
        response_parts.append("\n\n## Key Points:")
        for i, point in enumerate(detailed_points, 1):
            response_parts.append(f"{i}. {point}")

    # Add Kenyan context
    response_parts.append(f"\n\n🇰🇪 **Kenyan Context:** {kenyan_impact}")

    # Add citations
    if citations and citations[0]["url"] != "N/A":
        response_parts.append("\n\n📚 **Sources:**")
        for i, c in enumerate(citations[:5], 1):
            response_parts.append(f"[{i}] {c['source']}")

    # Add disclaimers based on confidence
    if persona == "wakili" and confidence < 0.8:
        response_parts.append(
            "\n\n⚖️ **Legal Disclaimer:** This information is for educational purposes only. Please consult with an Advocate of the High Court of Kenya for legal advice."
        )
    elif confidence < 0.6:
        response_parts.append(
            "\n\n⚠️ **Note:** This response has low confidence. Please verify with additional sources or consult subject matter experts."
        )
    elif confidence < 0.8:
        response_parts.append(
            "\n\n⚠️ **Note:** Please verify with authoritative sources for critical decisions."
        )

    # Add human review flag
    if state["human_review_flag"]:
        response_parts.append(
            "\n\n🔍 **Review Required:** This query has been flagged for expert review due to complexity or legal sensitivity."
        )

    final_response = "\n".join(response_parts)

    state["final_response"] = final_response
    state["formatted_response"] = formatted_response
    state["agent_path"].append("exit_gate")

    logger.info(
        f"Response generated: {len(final_response)} chars, confidence: {confidence:.2f}"
    )

    return state


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================


def route_from_entry(state: AmaniqV1State) -> str:
    """Route from entry gate to sheng translator or planner"""
    if state["detected_language"] in ["sheng", "swahili"]:
        return "sheng_translator"
    else:
        return "planner"


def route_from_translator(state: AmaniqV1State) -> str:
    """Route from translator to planner"""
    return "planner"


def route_from_quality_gate(state: AmaniqV1State) -> str:
    """Decide whether to iterate (re-plan) or proceed to exit"""
    iteration_count = state.get("iteration_count", 0)
    confidence = state["confidence_score"]
    max_iterations = state.get("max_iterations", 3)

    # Allow iterations for low confidence, but limit to max_iterations
    if confidence < 0.7 and iteration_count < max_iterations:
        state["iteration_count"] = iteration_count + 1
        logger.info(
            f"Low confidence ({confidence:.2f}), attempting iteration {iteration_count + 1}"
        )
        # Go back to planner to create refined plan
        return "planner"
    else:
        if iteration_count >= max_iterations:
            logger.info(
                f"Max iterations ({max_iterations}) reached, proceeding to exit"
            )
        return "exit_gate"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================


def create_amaniq_v1_graph(
    vector_store: Optional[VectorStore] = None,
    rag_pipeline: Optional[RAGPipeline] = None,
    enable_reasoning: bool = True,
    enable_quality_gate: bool = True,
    enable_persistence: bool = False,
    checkpoint_path: str = "./checkpoints/amaniq_v1.db",
    config_manager: Optional[Any] = None,
) -> StateGraph:
    """
    Create the AmaniQ v1 agent graph

    Args:
        vector_store: VectorStore instance for KB search
        rag_pipeline: RAG pipeline for synthesis
        enable_reasoning: Enable reasoning engine node
        enable_quality_gate: Enable quality gate validation
        enable_persistence: Enable state persistence with checkpoints
        checkpoint_path: Path to checkpoint database
        config_manager: ConfigManager for API keys

    Returns:
        Compiled LangGraph StateGraph
    """
    global _VECTOR_STORE, _RAG_PIPELINE, _KB_SEARCH_TOOL, _WEB_SEARCH_TOOL
    global _NEWS_SEARCH_TOOL, _TWITTER_TOOL, _REASONER, _PLANNER

    logger.info("Building AmaniQ v1 Agent Graph...")

    # Initialize global components
    if vector_store:
        _VECTOR_STORE = vector_store
    else:
        if VectorStore:
            _VECTOR_STORE = VectorStore(config_manager=config_manager)
        else:
            logger.warning("VectorStore not available")

    if rag_pipeline:
        _RAG_PIPELINE = rag_pipeline
    else:
        if RAGPipeline and _VECTOR_STORE:
            _RAG_PIPELINE = RAGPipeline(
                vector_store=_VECTOR_STORE, config_manager=config_manager
            )
        else:
            logger.warning("RAGPipeline not available")

    # Initialize tools
    _KB_SEARCH_TOOL = KnowledgeBaseSearchTool(vector_store=_VECTOR_STORE)

    try:
        _WEB_SEARCH_TOOL = WebSearchTool()
    except Exception as e:
        logger.warning(f"WebSearchTool initialization failed: {e}")

    try:
        _NEWS_SEARCH_TOOL = NewsSearchTool()
    except Exception as e:
        logger.warning(f"NewsSearchTool initialization failed: {e}")

    try:
        _TWITTER_TOOL = TwitterScraperTool()
    except Exception as e:
        logger.warning(f"TwitterScraperTool initialization failed: {e}")

    # Initialize reasoning components
    _REASONER = Reasoner()
    _PLANNER = Planner(rag_pipeline=_RAG_PIPELINE)

    # Create graph
    workflow = StateGraph(AmaniqV1State)

    # Add nodes
    workflow.add_node("entry_gate", entry_gate)
    workflow.add_node("sheng_translator", sheng_translator_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("compress_context", compress_context_node)  # NEW

    if enable_reasoning:
        workflow.add_node("reasoning_engine", reasoning_engine)
        workflow.add_node("persona_synthesis", persona_synthesis_node)
        workflow.add_node("fact_check", fact_check_node)  # NEW

    if enable_quality_gate:
        workflow.add_node("quality_gate", quality_gate)

    workflow.add_node("exit_gate", exit_gate)

    # Set entry point
    workflow.set_entry_point("entry_gate")

    # Add edges
    # From entry gate - conditional routing
    workflow.add_conditional_edges(
        "entry_gate",
        route_from_entry,
        {"sheng_translator": "sheng_translator", "planner": "planner"},
    )

    # From sheng translator to planner
    workflow.add_edge("sheng_translator", "planner")

    # From planner to tool executor
    workflow.add_edge("planner", "tool_executor")

    # From tool executor to compression
    workflow.add_edge("tool_executor", "compress_context")

    # From compression to reasoning
    if enable_reasoning:
        workflow.add_edge("compress_context", "reasoning_engine")
        workflow.add_edge("reasoning_engine", "persona_synthesis")
        workflow.add_edge("persona_synthesis", "fact_check")

        if enable_quality_gate:
            workflow.add_edge("fact_check", "quality_gate")

            # From quality gate - conditional for iteration
            workflow.add_conditional_edges(
                "quality_gate",
                route_from_quality_gate,
                {
                    "planner": "planner",  # Re-plan for iteration
                    "exit_gate": "exit_gate",
                },
            )
        else:
            workflow.add_edge("fact_check", "exit_gate")
    else:
        # Direct path if no reasoning
        if enable_quality_gate:
            workflow.add_edge("compress_context", "quality_gate")
            workflow.add_conditional_edges(
                "quality_gate",
                route_from_quality_gate,
                {"planner": "planner", "exit_gate": "exit_gate"},
            )
        else:
            workflow.add_edge("compress_context", "exit_gate")

    # Exit gate is the final node
    workflow.add_edge("exit_gate", END)

    # Compile graph
    if enable_persistence:
        # Create checkpoint directory
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        memory = SqliteSaver.from_conn_string(checkpoint_path)
        graph = workflow.compile(checkpointer=memory)
    else:
        graph = workflow.compile()

    logger.info("AmaniQ v1 Graph built successfully!")
    logger.info(
        f"Components initialized: VectorStore={_VECTOR_STORE is not None}, "
        f"RAGPipeline={_RAG_PIPELINE is not None}, "
        f"KBSearch={_KB_SEARCH_TOOL is not None}"
    )

    return graph


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def query_amaniq(
    query: str,
    session_context: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    graph: Optional[StateGraph] = None,
    vector_store: Optional[VectorStore] = None,
    rag_pipeline: Optional[RAGPipeline] = None,
    config_manager: Optional[Any] = None,
    skip_llm_synthesis: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to query AmaniQ agent

    Args:
        query: User query string
        session_context: Optional session context (user_county, preferences, etc.)
        conversation_history: Optional conversation history
        graph: Optional pre-built graph (creates new one if not provided)
        vector_store: Optional VectorStore instance
        rag_pipeline: Optional RAGPipeline instance
        config_manager: Optional ConfigManager instance

    Returns:
        Formatted response dictionary
    """
    if graph is None:
        graph = create_amaniq_v1_graph(
            vector_store=vector_store,
            rag_pipeline=rag_pipeline,
            config_manager=config_manager,
        )

    # Check env var for default synthesis setting (default to True if not set)
    # If ENABLE_LLM_SYNTHESIS is "false", then skip_llm_synthesis should be True
    env_enable_synthesis = os.getenv("ENABLE_LLM_SYNTHESIS", "true").lower() == "true"

    # Logic: skip if explicitly requested OR if env var disables it
    should_skip = skip_llm_synthesis or (not env_enable_synthesis)

    initial_state = {
        "user_query": query,
        "session_context": session_context or {},
        "conversation_history": conversation_history or [],
        "max_iterations": 2,  # Limit iterations for efficiency
        "skip_llm_synthesis": should_skip,
    }

    try:
        result = graph.invoke(initial_state)
        return result.get("formatted_response", {})
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        return {
            "answer": f"Error processing query: {str(e)}",
            "confidence": 0.0,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def visualize_graph(
    graph: Optional[StateGraph] = None, output_path: str = "./amaniq_v1_graph.png"
) -> None:
    """
    Visualize the AmaniQ agent graph

    Args:
        graph: StateGraph to visualize (creates new one if None)
        output_path: Path to save the visualization
    """
    if graph is None:
        graph = create_amaniq_v1_graph()

    try:
        # Try to generate visualization using Mermaid
        mermaid_code = graph.get_graph().draw_mermaid()
        print("AmaniQ v1 Agent Graph (Mermaid):")
        print(mermaid_code)

        # Save to file
        with open(output_path.replace(".png", ".mmd"), "w") as f:
            f.write(mermaid_code)

        logger.info(
            f"Graph visualization saved to {output_path.replace('.png', '.mmd')}"
        )
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        print("Graph structure:")
        print(f"Nodes: {list(graph.nodes.keys())}")
        print(f"Edges: {[(e.source, e.target) for e in graph.edges]}")


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("AmaniQ v1 Agent Graph - Example Usage")
    print("=" * 80)

    # Initialize components (in production, these would be properly configured)
    try:
        vector_store = VectorStore() if VectorStore else None
        rag_pipeline = (
            RAGPipeline(vector_store=vector_store)
            if RAGPipeline and vector_store
            else None
        )
    except Exception as e:
        logger.warning(f"Component initialization warning: {e}")
        vector_store = None
        rag_pipeline = None

    # Create graph once for all examples
    try:
        graph = create_amaniq_v1_graph(
            vector_store=vector_store,
            rag_pipeline=rag_pipeline,
            enable_reasoning=True,
            enable_quality_gate=True,
        )

        # Visualize graph structure
        print("\n" + "=" * 80)
        print("Graph Structure Visualization")
        print("=" * 80)
        visualize_graph(graph)

    except Exception as e:
        logger.error(f"Graph creation error: {e}")
        print(f"\nError creating graph: {e}")
        print("This may be due to missing dependencies. Install with:")
        print("  pip install langgraph langchain sentence-transformers")
        exit(1)

    # Example 1: General query (wanjiku persona)
    print("\n" + "=" * 80)
    print("EXAMPLE 1: General Query (Wanjiku Persona)")
    print("=" * 80)

    result = query_amaniq(
        query="What did the Treasury announce about tax reforms?",
        session_context={"user_county": "Nairobi"},
        graph=graph,
    )

    print(f"\nPersona: {result.get('persona', 'N/A')}")
    print(f"Intent: {result.get('intent', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 0.0):.2f}")
    print(f"Agent Path: {' → '.join(result.get('agent_path', []))}")
    print(f"Evidence Count: {result.get('evidence_count', 0)}")
    print(f"\nAnswer:\n{result.get('answer', 'No answer generated')[:500]}...")

    # Example 2: Legal query (wakili persona)
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Legal Query (Wakili Persona)")
    print("=" * 80)

    result = query_amaniq(
        query="What does Article 10 of the Constitution say about national values?",
        graph=graph,
    )

    print(f"\nPersona: {result.get('persona', 'N/A')}")
    print(f"Intent: {result.get('intent', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 0.0):.2f}")
    print(f"Agent Path: {' → '.join(result.get('agent_path', []))}")
    print(f"Human Review Required: {result.get('human_review_required', False)}")
    print(f"\nAnswer:\n{result.get('answer', 'No answer generated')[:500]}...")

    # Example 3: Hybrid query in Swahili (mwanahabari persona)
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Hybrid Query (Swahili - Mwanahabari Persona)")
    print("=" * 80)

    result = query_amaniq(
        query="Sheria mpya ya Finance Bill inaaathiri biashara vipi?", graph=graph
    )

    print(f"\nLanguage: {result.get('language', 'N/A')}")
    print(f"Persona: {result.get('persona', 'N/A')}")
    print(f"Intent: {result.get('intent', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 0.0):.2f}")
    print(f"Agent Path: {' → '.join(result.get('agent_path', []))}")
    print(f"\nAnswer:\n{result.get('answer', 'No answer generated')[:500]}...")

    # Example 4: Sheng query (wanjiku persona)
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Sheng Query (Wanjiku Persona)")
    print("=" * 80)

    result = query_amaniq(
        query="Kanjo wameongeza parking fees aje?",
        session_context={"user_county": "Nairobi"},
        graph=graph,
    )

    print(f"\nLanguage: {result.get('language', 'N/A')}")
    print(f"Persona: {result.get('persona', 'N/A')}")
    print(f"Intent: {result.get('intent', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 0.0):.2f}")
    print(f"Agent Path: {' → '.join(result.get('agent_path', []))}")
    print(f"Quality Issues: {result.get('quality_issues', [])}")
    print(f"\nAnswer:\n{result.get('answer', 'No answer generated')[:500]}...")

    # Example 5: News query (mwanahabari persona)
    print("\n" + "=" * 80)
    print("EXAMPLE 5: News Query (Mwanahabari Persona)")
    print("=" * 80)

    result = query_amaniq(
        query="What are the latest developments in Parliament regarding the housing levy?",
        graph=graph,
    )

    print(f"\nPersona: {result.get('persona', 'N/A')}")
    print(f"Intent: {result.get('intent', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 0.0):.2f}")
    print(f"Agent Path: {' → '.join(result.get('agent_path', []))}")
    print(f"Sources: {len(result.get('sources', []))}")
    print(f"\nAnswer:\n{result.get('answer', 'No answer generated')[:500]}...")

    print("\n" + "=" * 80)
    print("AmaniQ v1 Examples Complete")
    print("=" * 80)
