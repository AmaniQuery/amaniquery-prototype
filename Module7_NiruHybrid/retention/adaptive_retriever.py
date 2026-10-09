"""
Adaptive Retriever for Context-Aware Retrieval

Implements multi-stage retrieval (coarse + fine) with query-dependent
adjustment and context-aware similarity thresholds.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass
from loguru import logger

from ..config import RetentionConfig, default_config
from ..hybrid_encoder import HybridEncoder


@dataclass
class QueryContext:
    """Represents query context for adaptive retrieval"""

    query_text: str
    query_embeddings: torch.Tensor
    recent_queries: List[str] = None  # Recent query history
    query_type: Optional[str] = None  # e.g., "legal", "parliament", "news"
    timestamp: float = None

    def __post_init__(self):
        if self.recent_queries is None:
            self.recent_queries = []
        if self.timestamp is None:
            import time

            self.timestamp = time.time()


class AdaptiveRetriever:
    """Adaptive retriever with context-aware retrieval adjustment"""

    def __init__(
        self,
        hybrid_encoder: Optional[HybridEncoder] = None,
        vector_store=None,  # Existing vector store
        coarse_top_k: int = 50,
        fine_top_k: int = 5,
        similarity_threshold: float = 0.5,
        context_window_size: int = 5,
        config: Optional[RetentionConfig] = None,
    ):
        if config is not None:
            coarse_top_k = config.coarse_top_k
            fine_top_k = config.fine_top_k
            similarity_threshold = config.similarity_threshold
            context_window_size = config.context_window_size

        self.hybrid_encoder = hybrid_encoder
        self.vector_store = vector_store
        self.coarse_top_k = coarse_top_k
        self.fine_top_k = fine_top_k
        self.similarity_threshold = similarity_threshold
        self.context_window_size = context_window_size

        # Query history for context
        self.query_history = deque(maxlen=context_window_size)

        # Adaptive thresholds (learned from context)
        self.adaptive_thresholds = {}
        self.query_type_stats = {}  # Statistics per query type

    def encode_query(self, query_text: str, use_hybrid: bool = True) -> torch.Tensor:
        """
        Encode query to embeddings

        Args:
            query_text: Query text
            use_hybrid: Whether to use hybrid encoder

        Returns:
            embeddings: Query embeddings
        """
        if use_hybrid and self.hybrid_encoder is not None:
            # Use hybrid encoder
            with torch.no_grad():
                # Tokenize if needed (simplified - would use proper tokenizer)
                # For now, assume embeddings are provided or use simple encoding
                embeddings = self.hybrid_encoder.encode(
                    text=query_text, return_pooled=True
                )
                return embeddings
        else:
            # Fallback to existing vector store embedding
            if self.vector_store is not None:
                embeddings = self.vector_store.embedding_model.encode(query_text)
                return torch.tensor(embeddings)
            else:
                raise ValueError("No encoder available")

    def get_adaptive_threshold(self, query_context: QueryContext) -> float:
        """
        Get adaptive similarity threshold based on context

        Args:
            query_context: Query context

        Returns:
            threshold: Adaptive threshold
        """
        base_threshold = self.similarity_threshold

        # Adjust based on query type
        if query_context.query_type is not None:
            if query_context.query_type in self.adaptive_thresholds:
                # Use learned threshold for this query type
                base_threshold = self.adaptive_thresholds[query_context.query_type]

        # Adjust based on recent query history
        if len(self.query_history) > 0:
            # If recent queries are similar, lower threshold (more permissive)
            recent_similarities = []
            for recent_query in self.query_history:
                if hasattr(recent_query, "query_embeddings"):
                    similarity = torch.nn.functional.cosine_similarity(
                        query_context.query_embeddings.unsqueeze(0),
                        recent_query.query_embeddings.unsqueeze(0),
                    ).item()
                    recent_similarities.append(similarity)

            if recent_similarities:
                avg_similarity = np.mean(recent_similarities)
                # Lower threshold if queries are similar (more permissive)
                if avg_similarity > 0.7:
                    base_threshold *= 0.9
                # Higher threshold if queries are diverse (more strict)
                elif avg_similarity < 0.3:
                    base_threshold *= 1.1

        return min(max(base_threshold, 0.1), 0.9)  # Clamp between 0.1 and 0.9

    def coarse_retrieval(
        self, query_embeddings: torch.Tensor, top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Coarse retrieval stage - fast, approximate retrieval

        Args:
            query_embeddings: Query embeddings
            top_k: Number of results (default: coarse_top_k)

        Returns:
            results: List of retrieved documents
        """
        top_k = top_k or self.coarse_top_k

        if self.vector_store is not None:
            # Use existing vector store
            query_text = ""  # Not needed for embedding-based query
            results = self.vector_store.query(
                query_text="",  # Will use embeddings directly if supported
                n_results=top_k,
            )
            return results
        else:
            # Fallback: return empty results
            logger.warning("No vector store available for retrieval")
            return []

    def fine_retrieval(
        self,
        query_embeddings: torch.Tensor,
        coarse_results: List[Dict],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Fine retrieval stage - precise, similarity-based filtering

        Args:
            query_embeddings: Query embeddings
            coarse_results: Results from coarse retrieval
            top_k: Number of results (default: fine_top_k)
            threshold: Similarity threshold

        Returns:
            results: Filtered and ranked results
        """
        top_k = top_k or self.fine_top_k
        threshold = threshold or self.similarity_threshold

        if not coarse_results:
            return []

        # Compute similarities
        scored_results = []
        for result in coarse_results:
            # Get document embeddings
            if "embedding" in result:
                doc_emb = torch.tensor(result["embedding"])
            elif "metadata" in result and "embedding" in result["metadata"]:
                doc_emb = torch.tensor(result["metadata"]["embedding"])
            else:
                # Re-encode if needed
                if "text" in result:
                    doc_emb = self.encode_query(result["text"], use_hybrid=False)
                else:
                    continue

            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                query_embeddings.unsqueeze(0), doc_emb.unsqueeze(0)
            ).item()

            # Filter by threshold
            if similarity >= threshold:
                result["similarity"] = similarity
                scored_results.append(result)

        # Sort by similarity
        scored_results.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)

        # Return top k
        return scored_results[:top_k]

    def retrieve(
        self,
        query_text: str,
        query_context: Optional[QueryContext] = None,
        use_hybrid: bool = True,
        adaptive: bool = True,
    ) -> List[Dict]:
        """
        Adaptive retrieval with multi-stage filtering

        Args:
            query_text: Query text
            query_context: Optional query context
            use_hybrid: Whether to use hybrid encoder
            adaptive: Whether to use adaptive thresholds

        Returns:
            results: Retrieved documents
        """
        # Encode query
        query_embeddings = self.encode_query(query_text, use_hybrid=use_hybrid)

        # Create query context if not provided
        if query_context is None:
            query_context = QueryContext(
                query_text=query_text, query_embeddings=query_embeddings
            )

        # Get adaptive threshold
        if adaptive:
            threshold = self.get_adaptive_threshold(query_context)
        else:
            threshold = self.similarity_threshold

        # Coarse retrieval
        coarse_results = self.coarse_retrieval(
            query_embeddings, top_k=self.coarse_top_k
        )

        # Fine retrieval
        fine_results = self.fine_retrieval(
            query_embeddings, coarse_results, top_k=self.fine_top_k, threshold=threshold
        )

        # Update query history
        self.query_history.append(query_context)

        # Update statistics
        self._update_stats(query_context, len(fine_results))

        return fine_results

    def _update_stats(self, query_context: QueryContext, num_results: int):
        """Update statistics for adaptive learning"""
        query_type = query_context.query_type or "unknown"

        if query_type not in self.query_type_stats:
            self.query_type_stats[query_type] = {
                "count": 0,
                "avg_results": 0.0,
                "total_results": 0,
            }

        stats = self.query_type_stats[query_type]
        stats["count"] += 1
        stats["total_results"] += num_results
        stats["avg_results"] = stats["total_results"] / stats["count"]

        # Learn adaptive threshold (simple heuristic)
        if stats["count"] > 10:
            # If consistently getting few results, lower threshold
            if stats["avg_results"] < 3:
                self.adaptive_thresholds[query_type] = self.similarity_threshold * 0.9
            # If consistently getting many results, raise threshold
            elif stats["avg_results"] > 10:
                self.adaptive_thresholds[query_type] = self.similarity_threshold * 1.1

    def get_context_embeddings(self) -> Optional[torch.Tensor]:
        """
        Get aggregated context embeddings from recent queries

        Returns:
            context_embeddings: Aggregated context embeddings
        """
        if len(self.query_history) == 0:
            return None

        # Aggregate recent query embeddings
        embeddings_list = [
            qc.query_embeddings
            for qc in self.query_history
            if hasattr(qc, "query_embeddings")
        ]

        if not embeddings_list:
            return None

        # Mean pooling
        context_emb = torch.stack(embeddings_list).mean(dim=0)
        return context_emb

    def clear_history(self):
        """Clear query history"""
        self.query_history.clear()
        logger.info("Query history cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            "query_history_size": len(self.query_history),
            "adaptive_thresholds": self.adaptive_thresholds.copy(),
            "query_type_stats": {k: v.copy() for k, v in self.query_type_stats.items()},
            "coarse_top_k": self.coarse_top_k,
            "fine_top_k": self.fine_top_k,
            "base_similarity_threshold": self.similarity_threshold,
        }
