"""
Planner - Creates research plans using Chain-of-Thought and Tree-of-Thoughts
"""

from typing import List, Dict, Any, Optional
from loguru import logger

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Module4_NiruAPI.rag_pipeline import RAGPipeline


class Planner:
    """
    Creates structured research plans using reasoning techniques
    """

    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None):
        """
        Initialize planner

        Args:
            rag_pipeline: RAG pipeline for context-aware planning
        """
        self.rag_pipeline = rag_pipeline

    def create_plan(self, query: str) -> List[Dict[str, Any]]:
        """
        Create a research plan for a query

        Args:
            query: Research query

        Returns:
            List of plan steps
        """
        # Analyze query to determine plan structure
        plan = []

        # Step 1: Initial information gathering
        plan.append(
            {
                "step": 1,
                "action": "gather_initial_info",
                "action_type": "reasoning",
                "description": f"Gather initial information about: {query}",
                "requires_tools": True,
                "tool": "kb_search",
                "tool_args": {"query": query, "top_k": 5},
            }
        )

        # Step 2: Web search for current information
        plan.append(
            {
                "step": 2,
                "action": "web_search",
                "action_type": "tool_execution",
                "description": f"Search the web for current information about: {query}",
                "requires_tools": True,
                "tool": "web_search",
                "tool_args": {"query": query, "max_results": 10},
            }
        )

        # Step 3: News search for recent developments
        plan.append(
            {
                "step": 3,
                "action": "news_search",
                "action_type": "tool_execution",
                "description": f"Search news for recent developments related to: {query}",
                "requires_tools": True,
                "tool": "news_search",
                "tool_args": {"query": query, "max_results": 10},
            }
        )

        # Step 4: Social media intelligence (Twitter)
        plan.append(
            {
                "step": 4,
                "action": "twitter_search",
                "action_type": "tool_execution",
                "description": f"Search Twitter for public sentiment and discussions about: {query}",
                "requires_tools": True,
                "tool": "twitter_search",
                "tool_args": {"query": query, "max_results": 20},
            }
        )

        # Step 5: Analysis and synthesis
        plan.append(
            {
                "step": 5,
                "action": "analyze_and_synthesize",
                "action_type": "reasoning",
                "description": "Analyze gathered information and synthesize findings",
                "requires_tools": False,
            }
        )

        # Step 6: Validation and fact-checking
        plan.append(
            {
                "step": 6,
                "action": "validate_findings",
                "action_type": "reasoning",
                "description": "Validate findings and cross-reference sources",
                "requires_tools": False,
            }
        )

        logger.info(f"Created plan with {len(plan)} steps for query: {query[:50]}...")

        return plan

    def refine_plan(
        self, plan: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Refine plan based on context and intermediate results

        Args:
            plan: Original plan
            context: Context from previous steps

        Returns:
            Refined plan
        """
        # Check if we need to add additional steps based on context
        refined_plan = plan.copy()

        # If initial search found specific topics, add targeted searches
        if "initial_results" in context:
            results = context["initial_results"]
            if results and len(results) > 0:
                # Add targeted search steps
                for i, result in enumerate(results[:3], start=len(refined_plan) + 1):
                    refined_plan.append(
                        {
                            "step": i,
                            "action": f"targeted_search_{i}",
                            "action_type": "tool_execution",
                            "description": f'Targeted search based on finding: {result.get("title", "")[:50]}',
                            "requires_tools": True,
                            "tool": "web_search",
                            "tool_args": {
                                "query": result.get("title", ""),
                                "max_results": 5,
                            },
                        }
                    )

        return refined_plan
