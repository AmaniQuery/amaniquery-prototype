"""
Reflection Engine - Self-reflection and quality assessment
"""

from typing import Dict, Any, List, Optional
from loguru import logger


class ReflectionEngine:
    """
    Implements self-reflection mechanisms for quality assessment
    """

    def reflect(
        self,
        query: str,
        plan: List[Dict[str, Any]],
        actions_taken: List[Dict[str, Any]],
        tools_used: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
    ) -> str:
        """
        Reflect on the research process and assess quality

        Args:
            query: Original query
            plan: Research plan
            actions_taken: Actions performed
            tools_used: Tools used
            tool_results: Results from tools

        Returns:
            Reflection text
        """
        reflection_parts = []

        # Assess plan completion
        plan_steps = len(plan)
        completed_steps = len(actions_taken) + len(tools_used)
        completion_rate = completed_steps / plan_steps if plan_steps > 0 else 0

        reflection_parts.append(
            f"Plan Completion: {completion_rate:.0%} ({completed_steps}/{plan_steps} steps)"
        )

        # Assess tool usage
        tools_count = len(tools_used)
        results_count = len(tool_results)

        reflection_parts.append(
            f"Tools Used: {tools_count} tools executed, {results_count} results obtained"
        )

        # Assess result quality
        if tool_results:
            successful_results = sum(
                1
                for r in tool_results
                if r.get("result") and not r.get("result", {}).get("error")
            )
            success_rate = successful_results / len(tool_results)
            reflection_parts.append(f"Tool Success Rate: {success_rate:.0%}")

        # Assess information gathering
        total_sources = sum(
            len(r.get("result", {}).get("sources", []))
            for r in tool_results
            if isinstance(r.get("result"), dict)
        )
        reflection_parts.append(
            f"Information Gathered: {total_sources} sources collected"
        )

        # Overall assessment
        if completion_rate >= 0.8 and total_sources >= 5:
            reflection_parts.append(
                "Overall Assessment: Research process is sufficient and comprehensive"
            )
        elif completion_rate >= 0.5:
            reflection_parts.append(
                "Overall Assessment: Research process is partially complete, may need more information"
            )
        else:
            reflection_parts.append(
                "Overall Assessment: Research process is incomplete, requires more steps"
            )

        reflection = "\n".join(reflection_parts)

        logger.info(
            f"Reflection completed: {completion_rate:.0%} completion, {total_sources} sources"
        )

        return reflection

    def self_critique(
        self, answer: str, query: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Self-critique of the final answer

        Args:
            answer: Generated answer
            query: Original query
            sources: Sources used

        Returns:
            Critique with scores and suggestions
        """
        critique = {
            "answer_length": len(answer),
            "sources_count": len(sources),
            "scores": {},
            "suggestions": [],
        }

        # Score answer completeness
        if len(answer) < 100:
            critique["scores"]["completeness"] = 0.3
            critique["suggestions"].append("Answer is too short, may need more detail")
        elif len(answer) < 500:
            critique["scores"]["completeness"] = 0.7
        else:
            critique["scores"]["completeness"] = 1.0

        # Score source support
        if len(sources) == 0:
            critique["scores"]["source_support"] = 0.0
            critique["suggestions"].append(
                "No sources provided, answer lacks credibility"
            )
        elif len(sources) < 3:
            critique["scores"]["source_support"] = 0.5
            critique["suggestions"].append("More sources would strengthen the answer")
        else:
            critique["scores"]["source_support"] = 1.0

        # Score relevance
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(query_words.intersection(answer_words))
        relevance = min(overlap / len(query_words) if query_words else 0, 1.0)
        critique["scores"]["relevance"] = relevance

        if relevance < 0.5:
            critique["suggestions"].append(
                "Answer may not be fully relevant to the query"
            )

        # Overall score
        critique["scores"]["overall"] = sum(critique["scores"].values()) / len(
            critique["scores"]
        )

        return critique
