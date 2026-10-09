"""
Hybrid Retrieval for Hybrid Module
BM25 + embeddings, reranking, summary chunking with agentic integration
"""

import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Module4_NiruAPI.agents.discovery.hybrid_search import HybridSearch
from Module4_NiruAPI.agents.discovery.rag_retriever import RAGRetriever
from Module4_NiruAPI.agents.discovery.query_expansion import QueryExpansion
from Module7_NiruHybrid.agents.multi_agent_composition import MultiAgentComposition
from Module4_NiruAPI.agents.composition.agent_orchestrator import AgentOrchestrator
from Module7_NiruHybrid.agents.structured_outputs import StructuredOutputs
from Module3_NiruDB.vector_store import VectorStore


class HybridRetrieval:
    """
    Enhanced hybrid retrieval for hybrid module

    Combines BM25 + embeddings hybrid search, advanced reranking,
    summary chunking, multi-agent composition, and structured outputs.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        enable_agents: bool = True,
        enable_structured_outputs: bool = True,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """
        Initialize hybrid retrieval

        Args:
            vector_store: Vector store instance
            enable_agents: Whether to enable multi-agent composition
            enable_structured_outputs: Whether to enable structured output validation
            max_retries: Maximum retry attempts
            timeout: Operation timeout in seconds
        """
        self.vector_store = vector_store or VectorStore()
        self.hybrid_search = HybridSearch(vector_store=vector_store)
        self.rag_retriever = RAGRetriever(vector_store=vector_store)
        self.query_expansion = QueryExpansion()

        # Agentic components
        self.enable_agents = enable_agents
        self.enable_structured_outputs = enable_structured_outputs

        if enable_agents:
            try:
                self.multi_agent = MultiAgentComposition()
                self.agent_orchestrator = AgentOrchestrator()
                logger.info("Multi-agent composition enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize multi-agent composition: {e}")
                self.multi_agent = None
                self.agent_orchestrator = None
                self.enable_agents = False
        else:
            self.multi_agent = None
            self.agent_orchestrator = None

        if enable_structured_outputs:
            try:
                self.structured_outputs = StructuredOutputs()
                logger.info("Structured outputs enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize structured outputs: {e}")
                self.structured_outputs = None
                self.enable_structured_outputs = False
        else:
            self.structured_outputs = None

        # Configuration
        self.max_retries = max_retries
        self.timeout = timeout

        # Statistics
        self.retrieval_count = 0
        self.success_count = 0
        self.error_count = 0
        self.agent_processing_count = 0
        self.structured_output_count = 0

        logger.info(
            f"Hybrid retrieval initialized (agents: {self.enable_agents}, structured: {self.enable_structured_outputs})"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_reranking: bool = True,
        use_summary_chunking: bool = True,
        use_query_expansion: bool = True,
        use_agents: Optional[bool] = None,
        reranking_strategy: str = "combined",
    ) -> List[Dict[str, Any]]:
        """
        Document retrieval with hybrid search and enhancements

        Args:
            query: Search query
            top_k: Number of results (validated and clamped)
            use_reranking: Whether to rerank results
            use_summary_chunking: Whether to use summary chunking
            use_query_expansion: Whether to expand query
            use_agents: Whether to use multi-agent processing (overrides default)
            reranking_strategy: Reranking strategy ('combined', 'relevance', 'diversity')

        Returns:
            Retrieved documents with enhanced metadata
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to hybrid retrieval")
            return []

        # Validate and clamp top_k
        top_k = max(1, min(top_k, 100))

        start_time = time.time()
        self.retrieval_count += 1

        use_agents = use_agents if use_agents is not None else self.enable_agents

        try:
            # Expand query if requested
            if use_query_expansion:
                expanded_queries = self.query_expansion.expand(query)
                # Use original + top 2 expansions
                queries_to_search = [query] + expanded_queries[:2]
            else:
                queries_to_search = [query]

            # Retrieve using RAG retriever with hybrid search
            all_results = []
            for q in queries_to_search:
                try:
                    results = self.rag_retriever.retrieve(
                        query=q,
                        top_k=top_k * 2,  # Get more for reranking
                        use_hybrid=True,
                        use_expansion=False,  # Already expanded
                        rerank=use_reranking,
                    )
                    all_results.extend(results)
                except Exception as e:
                    logger.warning(f"Error retrieving for query '{q}': {e}")
                    continue

            # Deduplicate results
            seen_content = set()
            unique_results = []
            for result in all_results:
                content = result.get("content", "")
                if content and content not in seen_content:
                    seen_content.add(content)
                    unique_results.append(result)

            # Apply advanced reranking if requested
            if use_reranking:
                unique_results = self._advanced_rerank(
                    unique_results, query, strategy=reranking_strategy
                )

            # Apply summary chunking if requested
            if use_summary_chunking:
                unique_results = self._apply_summary_chunking(unique_results)

            # Process with agents if enabled
            if use_agents and self.multi_agent:
                unique_results = self._process_with_agents(unique_results, query)

            # Limit to top_k
            final_results = unique_results[:top_k]

            # Add metadata
            for result in final_results:
                result["retrieval_metadata"] = {
                    "query": query,
                    "retrieval_time": time.time() - start_time,
                    "reranking_used": use_reranking,
                    "chunking_used": use_summary_chunking,
                    "agents_used": use_agents,
                    "timestamp": datetime.utcnow().isoformat(),
                }

            elapsed = time.time() - start_time
            self.success_count += 1
            logger.debug(
                f"Hybrid retrieval completed in {elapsed:.3f}s ({len(final_results)} results)"
            )

            return final_results

        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in hybrid retrieval: {e}", exc_info=True)
            # Graceful degradation - try simple retrieval
            try:
                simple_results = self.hybrid_search.search(query, top_k=top_k)
                return simple_results
            except Exception as fallback_error:
                logger.error(f"Fallback retrieval also failed: {fallback_error}")
                return []

    def _advanced_rerank(
        self, results: List[Dict[str, Any]], query: str, strategy: str = "combined"
    ) -> List[Dict[str, Any]]:
        """
        Advanced reranking with multiple strategies

        Args:
            results: Search results
            query: Original query
            strategy: Reranking strategy ('combined', 'relevance', 'diversity')

        Returns:
            Reranked results
        """
        if not results:
            return results

        query_terms = set(query.lower().split())

        for result in results:
            content = result.get("content", "").lower()
            content_terms = set(content.split())

            # Calculate relevance metrics
            term_overlap = len(query_terms.intersection(content_terms))
            overlap_ratio = term_overlap / len(query_terms) if query_terms else 0

            # Get existing scores
            combined_score = result.get("combined_score", result.get("score", 0.0))
            vector_score = result.get("vector_score", 0.0)
            bm25_score = result.get("bm25_score", 0.0)

            # Calculate rerank score based on strategy
            if strategy == "relevance":
                # Boost based on term overlap
                rerank_score = combined_score + (overlap_ratio * 0.3)
            elif strategy == "diversity":
                # Penalize very similar results (simple diversity)
                rerank_score = combined_score * (1.0 - min(overlap_ratio * 0.2, 0.1))
            else:  # combined
                # Combine relevance and diversity
                relevance_boost = overlap_ratio * 0.2
                diversity_penalty = min(overlap_ratio * 0.1, 0.05)
                rerank_score = combined_score + relevance_boost - diversity_penalty

            result["rerank_score"] = rerank_score
            result["reranking_metadata"] = {
                "strategy": strategy,
                "term_overlap": term_overlap,
                "overlap_ratio": overlap_ratio,
            }

        # Sort by rerank score
        return sorted(results, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

    def _apply_summary_chunking(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply intelligent summary chunking to results

        Args:
            results: Search results

        Returns:
            Results with summary chunks
        """
        for result in results:
            content = result.get("content", "")

            if not content:
                result["summary_chunk"] = ""
                continue

            content_length = len(content)

            if content_length > 2000:
                # For very long content: first 600 + middle 400 + last 200
                first_part = content[:600]
                middle_start = content_length // 2 - 200
                middle_part = content[middle_start : middle_start + 400]
                last_part = content[-200:]
                result["summary_chunk"] = f"{first_part}...{middle_part}...{last_part}"
            elif content_length > 1000:
                # For long content: first 500 + last 200
                result["summary_chunk"] = content[:500] + "..." + content[-200:]
            else:
                # Short content: use as-is
                result["summary_chunk"] = content

            # Add chunk metadata
            result["chunk_metadata"] = {
                "original_length": content_length,
                "chunk_length": len(result["summary_chunk"]),
                "compression_ratio": (
                    len(result["summary_chunk"]) / content_length
                    if content_length > 0
                    else 1.0
                ),
            }

        return results

    def _process_with_agents(
        self, results: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """
        Process results through multi-agent composition

        Args:
            results: Search results
            query: Original query

        Returns:
            Enhanced results with agent processing
        """
        if not self.multi_agent:
            return results

        try:
            self.agent_processing_count += 1

            # Process each result through agents
            enhanced_results = []
            for result in results:
                content = result.get("content", "")
                sources = result.get("sources", [])

                if not content:
                    enhanced_results.append(result)
                    continue

                # Use multi-agent to enhance result
                agent_result = self.multi_agent.process_with_agents(
                    content=content,
                    sources=sources,
                    use_citer=True,
                    use_editor=True,
                    use_validator=True,
                )

                # Update result with agent enhancements
                if agent_result and "final_result" in agent_result:
                    final = agent_result["final_result"]
                    result["agent_processed_content"] = final.get("content", content)
                    result["agent_validation"] = agent_result.get("agent_results", [])
                    result["agent_metadata"] = {
                        "agents_used": agent_result.get("agent_chain", []),
                        "processing_timestamp": datetime.utcnow().isoformat(),
                    }

                enhanced_results.append(result)

            return enhanced_results
        except Exception as e:
            logger.error(f"Error in agent processing: {e}")
            return results  # Return original results on error

    def retrieve_with_filters(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int = 5,
        filter_mode: str = "strict",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve with metadata filters

        Args:
            query: Search query
            filters: Metadata filters (supports nested keys with dot notation)
            top_k: Number of results
            filter_mode: Filter mode ('strict' = all must match, 'any' = any can match)

        Returns:
            Filtered results
        """
        # Get more results to account for filtering
        results = self.retrieve(query, top_k=top_k * 3)

        # Apply filters
        filtered = []
        for result in results:
            metadata = result.get("metadata", {})
            match = self._match_filters(metadata, filters, filter_mode)

            if match:
                filtered.append(result)
                if len(filtered) >= top_k:
                    break

        return filtered[:top_k]

    def _match_filters(
        self, metadata: Dict[str, Any], filters: Dict[str, Any], mode: str = "strict"
    ) -> bool:
        """
        Match metadata against filters

        Args:
            metadata: Document metadata
            filters: Filter criteria
            mode: 'strict' (all must match) or 'any' (any can match)

        Returns:
            Whether metadata matches filters
        """
        matches = []

        for key, value in filters.items():
            # Support nested keys with dot notation
            if "." in key:
                keys = key.split(".")
                current = metadata
                try:
                    for k in keys[:-1]:
                        current = current.get(k, {})
                    actual_value = current.get(keys[-1])
                except (AttributeError, KeyError, TypeError):
                    actual_value = None
            else:
                actual_value = metadata.get(key)

            # Match check
            if isinstance(value, list):
                match = actual_value in value
            elif isinstance(value, dict) and "in" in value:
                # Support range queries: {'in': [min, max]}
                range_vals = value["in"]
                match = range_vals[0] <= actual_value <= range_vals[1]
            else:
                match = actual_value == value

            matches.append(match)

        if mode == "any":
            return any(matches)
        else:  # strict
            return all(matches)

    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        stats = {
            "total_retrievals": self.retrieval_count,
            "successful_retrievals": self.success_count,
            "failed_retrievals": self.error_count,
            "agent_processing_count": self.agent_processing_count,
            "structured_output_count": self.structured_output_count,
            "success_rate": (
                self.success_count / self.retrieval_count
                if self.retrieval_count > 0
                else 0.0
            ),
            "agents_enabled": self.enable_agents,
            "structured_outputs_enabled": self.enable_structured_outputs,
        }

        # Add hybrid search stats
        if hasattr(self.hybrid_search, "get_stats"):
            stats["hybrid_search"] = self.hybrid_search.get_stats()

        return stats
