"""
RAG Retriever - Enhanced retrieval with reranking and context awareness
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from .hybrid_search import HybridSearch
from .query_expansion import QueryExpansion

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from Module3_NiruDB.vector_store import VectorStore


class RAGRetriever:
    """
    RAG retriever with hybrid search, reranking, and query expansion
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize RAG retriever

        Args:
            vector_store: Vector store instance
        """
        self.vector_store = vector_store or VectorStore()
        self.hybrid_search = HybridSearch(vector_store=vector_store)
        self.query_expansion = QueryExpansion()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        use_expansion: bool = True,
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents

        Args:
            query: Search query
            top_k: Number of results
            use_hybrid: Whether to use hybrid search
            use_expansion: Whether to expand query
            rerank: Whether to rerank results

        Returns:
            List of retrieved documents
        """
        # Expand query if requested
        queries = [query]
        if use_expansion:
            expanded = self.query_expansion.expand(query)
            queries.extend(expanded[:2])  # Use top 2 expansions

        all_results = []

        # Search with each query variation
        for q in queries:
            if use_hybrid:
                results = self.hybrid_search.search(q, top_k=top_k * 2)
            else:
                # Fallback to vector search only
                results = self.vector_store.search(q, top_k=top_k * 2)

            all_results.extend(results)

        # Deduplicate
        seen = set()
        unique_results = []
        for result in all_results:
            content = result.get("content", "")
            if content and content not in seen:
                seen.add(content)
                unique_results.append(result)

        # Rerank if requested
        if rerank:
            unique_results = self._rerank(unique_results, query)

        return unique_results[:top_k]

    def _rerank(
        self, results: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """
        Rerank results based on relevance

        Args:
            results: Search results
            query: Original query

        Returns:
            Reranked results
        """
        # Simple reranking based on query term overlap
        query_terms = set(query.lower().split())

        for result in results:
            content = result.get("content", "").lower()
            content_terms = set(content.split())

            # Calculate overlap
            overlap = len(query_terms.intersection(content_terms))
            overlap_ratio = overlap / len(query_terms) if query_terms else 0

            # Boost score based on overlap
            original_score = result.get("combined_score", result.get("score", 0.0))
            result["rerank_score"] = original_score + (overlap_ratio * 0.2)

        # Sort by rerank score
        return sorted(results, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
