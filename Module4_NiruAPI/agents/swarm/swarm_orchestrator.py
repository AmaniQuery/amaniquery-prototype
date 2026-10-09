"""
Swarm Intelligence Orchestrator
Coordinates multiple LLMs for consensus building and division of labor
"""

from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
from loguru import logger

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from Module4_NiruAPI.rag_pipeline import RAGPipeline


class SwarmOrchestrator:
    """
    Orchestrates multiple LLMs to work together on queries
    Supports both consensus (all LLMs on same query) and division of labor (split tasks)
    """

    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        providers: Optional[List[str]] = None,
    ):
        """
        Initialize swarm orchestrator

        Args:
            rag_pipeline: RAG pipeline instance
            providers: List of LLM providers to use (default: all available)
        """
        self.rag_pipeline = rag_pipeline
        self.providers = providers or ["openai", "anthropic", "gemini", "moonshot"]

        # Available providers mapping
        self.provider_configs = {
            "openai": {"model": "gpt-4", "requires_key": "OPENAI_API_KEY"},
            "anthropic": {
                "model": "claude-3-opus-20240229",
                "requires_key": "ANTHROPIC_API_KEY",
            },
            "gemini": {
                "model": "gemini-2.5-flash",
                "requires_key": "GEMINI_API_KEY",
            },  # Updated to 2.5-flash
            "moonshot": {"model": "moonshot-v1-8k", "requires_key": "MOONSHOT_API_KEY"},
        }

        # Filter providers based on available API keys
        self.available_providers = self._filter_available_providers()

        logger.info(
            f"Swarm orchestrator initialized with {len(self.available_providers)} providers: {self.available_providers}"
        )

    def _filter_available_providers(self) -> List[str]:
        """Filter providers based on available API keys"""
        import os

        available = []

        for provider in self.providers:
            config = self.provider_configs.get(provider)
            if config:
                key_name = config["requires_key"]
                if os.getenv(key_name):
                    available.append(provider)
                else:
                    logger.warning(
                        f"Provider {provider} not available: {key_name} not set"
                    )

        return available

    async def query_parallel(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query all available LLMs in parallel (consensus mode)

        Args:
            query: Query to process
            context: Optional context

        Returns:
            List of responses from each LLM
        """
        if not self.available_providers:
            logger.warning("No LLM providers available")
            return []

        tasks = []
        for provider in self.available_providers:
            task = self._query_provider(provider, query, context)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error querying {self.available_providers[i]}: {result}")
            else:
                valid_results.append(result)

        return valid_results

    async def _query_provider(
        self, provider: str, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query a single provider"""
        try:
            if self.rag_pipeline:
                # Use RAG pipeline with specific provider
                import asyncio

                # Run synchronous query in executor to make it async-compatible
                loop = asyncio.get_event_loop()
                original_provider = self.rag_pipeline.llm_provider
                self.rag_pipeline.llm_provider = provider

                # Execute query in thread pool to avoid blocking
                response = await loop.run_in_executor(
                    None, lambda: self.rag_pipeline.query(query=query, top_k=5)
                )

                self.rag_pipeline.llm_provider = original_provider

                return {
                    "provider": provider,
                    "response": response.get("answer", ""),
                    "sources": response.get("sources", []),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            else:
                # Direct LLM query (simplified)
                return {
                    "provider": provider,
                    "response": f"Response from {provider}",
                    "sources": [],
                    "timestamp": datetime.utcnow().isoformat(),
                }
        except Exception as e:
            logger.error(f"Error querying {provider}: {e}")
            return {
                "provider": provider,
                "response": "",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    def synthesize_responses(self, responses: List[Dict[str, Any]]) -> str:
        """
        Synthesize multiple LLM responses into a single answer
        Removes redundancy and builds consensus

        Args:
            responses: List of responses from different LLMs

        Returns:
            Synthesized response
        """
        if not responses:
            return "No responses available"

        if len(responses) == 1:
            return responses[0].get("response", "")

        # Extract responses
        response_texts = [r.get("response", "") for r in responses if r.get("response")]

        if not response_texts:
            return "No valid responses"

        # Simple synthesis: combine and deduplicate
        # In a production system, this would use an LLM to synthesize
        combined = []
        seen_points = set()

        for response in response_texts:
            # Split into sentences/points
            points = response.split(". ")
            for point in points:
                point = point.strip()
                if point and len(point) > 20:  # Filter very short points
                    # Simple deduplication
                    point_lower = point.lower()
                    if point_lower not in seen_points:
                        seen_points.add(point_lower)
                        combined.append(point)

        return (
            ". ".join(combined) + "." if combined else "Unable to synthesize responses"
        )

    async def query_with_context(self, query: str, context: Dict[str, Any]) -> str:
        """
        Query with additional context (for reasoning steps)

        Args:
            query: Query to process
            context: Additional context (plan, previous results, etc.)

        Returns:
            Response string
        """
        # Build enhanced query with context
        enhanced_query = f"{query}\n\nContext:\n"

        if "plan_step" in context:
            enhanced_query += f"Current step: {context['plan_step']}\n"

        if "previous_results" in context:
            enhanced_query += (
                f"Previous results: {len(context['previous_results'])} items\n"
            )

        # Query in parallel
        responses = await self.query_parallel(enhanced_query, context)

        # Synthesize
        return self.synthesize_responses(responses)

    async def synthesize_final_answer(self, query: str, context: Dict[str, Any]) -> str:
        """
        Synthesize final answer from all available information

        Args:
            query: Original query
            context: Full context including plan, actions, tool results, reflection

        Returns:
            Final synthesized answer
        """
        # Build comprehensive context
        synthesis_query = f"""
        Based on the following research process, provide a comprehensive final answer to the query.
        
        Original Query: {query}
        
        Research Plan: {context.get('plan', [])}
        
        Actions Taken: {len(context.get('actions_taken', []))} actions
        {self._format_actions(context.get('actions_taken', []))}
        
        Tool Results: {len(context.get('tool_results', []))} results
        {self._format_tool_results(context.get('tool_results', []))}
        
        Reflection: {context.get('reflection', 'N/A')}
        
        Please synthesize a comprehensive, well-structured answer that addresses the original query.
        """

        # Query all providers
        responses = await self.query_parallel(synthesis_query, context)

        # Synthesize responses
        return self.synthesize_responses(responses)

    def _format_actions(self, actions: List[Dict[str, Any]]) -> str:
        """Format actions for context"""
        if not actions:
            return "No actions taken"

        formatted = []
        for i, action in enumerate(actions, 1):
            action_type = action.get("action_type", "unknown")
            result = str(action.get("result", ""))[:200]  # Truncate long results
            formatted.append(f"{i}. {action_type}: {result}")

        return "\n".join(formatted)

    def _format_tool_results(self, tool_results: List[Dict[str, Any]]) -> str:
        """Format tool results for context"""
        if not tool_results:
            return "No tool results"

        formatted = []
        for i, result in enumerate(tool_results, 1):
            tool = result.get("tool", "unknown")
            result_data = str(result.get("result", ""))[:200]  # Truncate
            formatted.append(f"{i}. {tool}: {result_data}")

        return "\n".join(formatted)

    def divide_labor(self, query: str, subtasks: List[str]) -> Dict[str, str]:
        """
        Divide labor across multiple LLMs (each handles a subtask)

        Args:
            query: Main query
            subtasks: List of subtasks to assign

        Returns:
            Dictionary mapping subtask to response
        """
        if not subtasks:
            return {}

        # Assign subtasks to available providers
        assignments = {}
        for i, subtask in enumerate(subtasks):
            provider = self.available_providers[i % len(self.available_providers)]
            assignments[subtask] = provider

        # Execute in parallel
        async def execute():
            tasks = []
            for subtask, provider in assignments.items():
                task = self._query_provider(provider, f"{query}\n\nSubtask: {subtask}")
                tasks.append((subtask, task))

            results = {}
            for subtask, task in tasks:
                result = await task
                results[subtask] = result.get("response", "")

            return results

        # Run synchronously (in production, this would be async)
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(execute())
