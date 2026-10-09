"""
Reflection Node for Agentic RAG
Self-critique and targeted refinement of draft answers
"""

import json
from typing import Dict, Any, List
from loguru import logger

# =============================================================================
# REFLECTION PROMPT
# =============================================================================

REFLECTION_PROMPT = """You are critiquing a draft answer for quality and completeness.

Draft Answer:
{draft_answer}

Retrieved Context:
{context_summary}

Analyze the draft critically:
1. Which specific claims lack direct evidence from the context?
2. Are there contradictions or logical gaps?
3. What additional information would strengthen the answer?
4. Is the legal/parliamentary context accurate for Kenya?

Return JSON with this exact schema:
{{
  "weak_claims": ["claim 1 that lacks evidence", "claim 2 that needs support"],
  "missing_evidence": ["specific fact needed", "additional context needed"],
  "search_queries": ["targeted query 1", "targeted query 2"],
  "confidence": 0.85,
  "needs_refinement": true,
  "critique": "Brief explanation of main gaps"
}}

Be strict. Flag ANY claim not directly supported by the context.
If the answer is excellent (no gaps), set needs_refinement to false.
"""


# =============================================================================
# REFLECTION NODE
# =============================================================================


def reflection_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reflection node - critique draft answer and suggest improvements

    Args:
        state: AmaniQState with react_final_answer or initial tool results

    Returns:
        Updated state with reflection results
    """
    # Get draft answer (from ReAct or initial retrieval)
    draft_answer = state.get("react_final_answer") or state.get("final_response", "")

    if not draft_answer:
        logger.warning("[Reflection] No draft answer to critique")
        return {"reflection_needed": False, "reflection_confidence": 1.0}

    logger.info("[Reflection] Analyzing draft answer quality...")

    # Get context for evaluation
    tool_results = state.get("tool_results", [])
    context_summary = _summarize_context(tool_results)

    # Build reflection prompt
    prompt = REFLECTION_PROMPT.format(
        draft_answer=draft_answer[:1500],  # Truncate if too long
        context_summary=context_summary,
    )

    try:
        # Get LLM critique
        from Module4_NiruAPI.agents.amaniq_v2 import MoonshotClient, AmaniQConfig

        config = AmaniQConfig()
        client = MoonshotClient.get_client(config)

        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=800,
        )

        reflection = json.loads(response.choices[0].message.content)

        logger.info(
            f"[Reflection] Confidence: {reflection['confidence']:.2f}, Needs refinement: {reflection['needs_refinement']}"
        )

        # If needs refinement and has search queries, flag for targeted retrieval
        if reflection.get("needs_refinement") and reflection.get("search_queries"):
            logger.info(
                f"[Reflection] Suggesting {len(reflection['search_queries'])} targeted searches"
            )

        return {
            "reflection_result": reflection,
            "reflection_confidence": reflection.get("confidence", 0.5),
            "reflection_needed": reflection.get("needs_refinement", False),
            "targeted_queries": reflection.get("search_queries", []),
            "weak_claims": reflection.get("weak_claims", []),
        }

    except Exception as e:
        logger.error(f"[Reflection] Failed: {e}")
        return {
            "reflection_needed": False,
            "reflection_confidence": 0.5,
            "reflection_error": str(e),
        }


def _summarize_context(tool_results: List[Dict]) -> str:
    """Summarize tool results for reflection"""
    if not tool_results:
        return "No context available"

    summary_parts = []
    for i, result in enumerate(tool_results[:5], 1):  # Limit to first 5
        content = result.get("result", result.get("output", ""))
        if isinstance(content, str):
            summary_parts.append(f"{i}. {content[:200]}...")
        elif isinstance(content, dict):
            summary_parts.append(f"{i}. {str(content)[:200]}...")

    return "\n".join(summary_parts)


# =============================================================================
# TARGETED REFINEMENT
# =============================================================================


def targeted_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute targeted searches based on reflection

    Args:
        state: AmaniQState with targeted_queries from reflection

    Returns:
        Updated state with refinement results
    """
    targeted_queries = state.get("targeted_queries", [])

    if not targeted_queries:
        logger.info("[Targeted Search] No queries needed")
        return {"refinement_results": []}

    logger.info(
        f"[Targeted Search] Executing {len(targeted_queries)} refinement searches"
    )

    # Execute each targeted query
    from Module4_NiruAPI.agents.tools.agentic_tools import get_agentic_tools

    tool_registry = get_agentic_tools()

    results = []
    for query in targeted_queries[:3]:  # Limit to 3 refinement searches
        try:
            result = tool_registry.execute(
                "search_knowledge_base", query=query, top_k=3
            )
            results.append(
                {
                    "query": query,
                    "result": result,
                    "success": result.get("success", False),
                }
            )
        except Exception as e:
            logger.error(f"[Targeted Search] Failed for '{query}': {e}")
            results.append({"query": query, "error": str(e), "success": False})

    logger.info(f"[Targeted Search] Completed {len(results)} searches")

    return {"refinement_results": results, "refinement_completed": True}


# =============================================================================
# ROUTING HELPER
# =============================================================================


def should_refine(state: Dict[str, Any]) -> bool:
    """Check if refinement is needed based on reflection"""
    return (
        state.get("reflection_needed", False)
        and len(state.get("targeted_queries", [])) > 0
    )
