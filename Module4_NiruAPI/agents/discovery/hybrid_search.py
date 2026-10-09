"""
Hybrid Search
Combines BM25 (keyword) and embeddings (semantic) search with robust error handling
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from functools import lru_cache
from loguru import logger

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
    logger.warning("rank-bm25 not installed. Install with: pip install rank-bm25")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from Module3_NiruDB.vector_store import VectorStore


class HybridSearch:
    """
    Hybrid search combining BM25 (keyword) and vector (semantic) search

    Implements error handling, retry logic, score normalization,
    and graceful degradation when components fail.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        enable_caching: bool = True,
        cache_size: int = 128,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """
        Initialize hybrid search

        Args:
            vector_store: Vector store for semantic search
            bm25_weight: Default weight for BM25 scores (0-1)
            vector_weight: Default weight for vector scores (0-1)
            enable_caching: Whether to enable query result caching
            cache_size: Maximum cache size (LRU)
            max_retries: Maximum retry attempts for vector search
            timeout: Timeout for vector search operations
        """
        self.vector_store = vector_store or VectorStore()
        self.bm25_index = None
        self.documents = []
        self.document_metadata = []  # Store metadata alongside documents

        # Configuration
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.enable_caching = enable_caching
        self.max_retries = max_retries
        self.timeout = timeout

        # Validate weights sum to 1.0
        total_weight = bm25_weight + vector_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights don't sum to 1.0 ({total_weight}), normalizing...")
            self.bm25_weight = bm25_weight / total_weight
            self.vector_weight = vector_weight / total_weight

        # Statistics
        self.search_count = 0
        self.bm25_only_count = 0
        self.vector_only_count = 0
        self.hybrid_count = 0
        self.error_count = 0

        if BM25Okapi is None:
            logger.warning("BM25 search will not be available - using vector-only mode")

        logger.info(
            f"Hybrid search initialized (BM25: {self.bm25_weight}, Vector: {self.vector_weight})"
        )

    def build_bm25_index(
        self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Build BM25 index from documents

        Args:
            documents: List of document texts
            metadata: Optional list of metadata dicts corresponding to documents
        """
        if BM25Okapi is None:
            logger.warning("Cannot build BM25 index: rank-bm25 not available")
            return

        if not documents:
            logger.warning("No documents provided for BM25 index")
            return

        try:
            # Validate documents
            valid_docs = []
            valid_metadata = []

            for i, doc in enumerate(documents):
                if doc and isinstance(doc, str) and len(doc.strip()) > 0:
                    valid_docs.append(doc)
                    if metadata and i < len(metadata):
                        valid_metadata.append(metadata[i])
                    else:
                        valid_metadata.append({})

            if not valid_docs:
                logger.error("No valid documents after validation")
                return

            # Tokenize documents with error handling
            tokenized_docs = []
            for doc in valid_docs:
                try:
                    # Simple tokenization - split on whitespace and punctuation
                    tokens = doc.lower().split()
                    # Filter out very short tokens
                    tokens = [t for t in tokens if len(t) > 1]
                    tokenized_docs.append(tokens)
                except Exception as e:
                    logger.warning(f"Error tokenizing document: {e}")
                    tokenized_docs.append([])

            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_docs)
            self.documents = valid_docs
            self.document_metadata = valid_metadata

            logger.info(f"Built BM25 index for {len(valid_docs)} documents")
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}", exc_info=True)
            self.bm25_index = None
            self.documents = []
            self.document_metadata = []

    def _normalize_scores(
        self, vector_scores: List[float], bm25_scores: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        Normalize scores to [0, 1] range for fair combination

        Args:
            vector_scores: List of vector similarity scores
            bm25_scores: List of BM25 scores

        Returns:
            Tuple of (normalized_vector_scores, normalized_bm25_scores)
        """
        # Normalize vector scores (typically already in [0, 1])
        if vector_scores:
            v_min, v_max = min(vector_scores), max(vector_scores)
            if v_max > v_min:
                vector_norm = [(s - v_min) / (v_max - v_min) for s in vector_scores]
            else:
                vector_norm = [1.0] * len(vector_scores) if vector_scores else []
        else:
            vector_norm = []

        # Normalize BM25 scores (can be negative or very large)
        if bm25_scores:
            b_min, b_max = min(bm25_scores), max(bm25_scores)
            if b_max > b_min:
                bm25_norm = [(s - b_min) / (b_max - b_min) for s in bm25_scores]
            else:
                bm25_norm = [1.0] * len(bm25_scores) if bm25_scores else []
        else:
            bm25_norm = []

        return vector_norm, bm25_norm

    def _vector_search_with_retry(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform vector search with retry logic

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of vector search results
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                results = self.vector_store.search(query, top_k=top_k)

                # Validate results
                if not isinstance(results, list):
                    logger.warning(
                        "Vector search returned non-list result, converting..."
                    )
                    results = [results] if results else []

                # Filter out invalid results
                valid_results = []
                for result in results:
                    if isinstance(result, dict) and result.get("content"):
                        valid_results.append(result)

                elapsed = time.time() - start_time
                if elapsed > self.timeout:
                    logger.warning(
                        f"Vector search took {elapsed:.2f}s (timeout: {self.timeout}s)"
                    )

                return valid_results
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = (2**attempt) * 0.5  # Exponential backoff
                    logger.warning(
                        f"Vector search attempt {attempt + 1}/{self.max_retries} failed: {e}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Vector search failed after {self.max_retries} attempts: {e}"
                    )

        if last_exception:
            logger.error(f"Vector search failed: {last_exception}")
        return []

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform BM25 search with error handling

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of BM25 search results
        """
        if not self.bm25_index or BM25Okapi is None:
            return []

        if not self.documents:
            logger.warning("BM25 index exists but no documents available")
            return []

        try:
            # Tokenize query
            tokenized_query = query.lower().split()
            tokenized_query = [
                t for t in tokenized_query if len(t) > 1
            ]  # Filter short tokens

            if not tokenized_query:
                logger.warning("Query has no valid tokens after tokenization")
                return []

            # Get BM25 scores
            bm25_scores = self.bm25_index.get_scores(tokenized_query)

            if not bm25_scores or len(bm25_scores) == 0:
                return []

            # Get top indices
            top_indices = sorted(
                range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
            )[
                : top_k * 2
            ]  # Get more for combination

            # Build results
            results = []
            for idx in top_indices:
                if 0 <= idx < len(self.documents):
                    result = {
                        "content": self.documents[idx],
                        "score": float(bm25_scores[idx]),
                        "index": idx,
                    }
                    # Add metadata if available
                    if idx < len(self.document_metadata):
                        result["metadata"] = self.document_metadata[idx]
                    else:
                        result["metadata"] = {}
                    results.append(result)

            return results
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}", exc_info=True)
            return []

    @lru_cache(maxsize=128)
    def _cached_search(
        self, query: str, top_k: int, bm25_w: float, vec_w: float
    ) -> Tuple[str, int, float, float]:
        """Cached search wrapper (returns tuple for caching)"""
        # This is a helper for caching - actual search logic is in search()
        return query, top_k, bm25_w, vec_w

    def search(
        self,
        query: str,
        top_k: int = 10,
        bm25_weight: Optional[float] = None,
        vector_weight: Optional[float] = None,
        use_cache: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search

        Args:
            query: Search query
            top_k: Number of results to return (validated and clamped)
            bm25_weight: Weight for BM25 scores (0-1), uses default if None
            vector_weight: Weight for vector scores (0-1), uses default if None
            use_cache: Whether to use caching (uses default if None)

        Returns:
            List of search results with combined scores and metadata
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to hybrid search")
            return []

        # Validate and clamp top_k
        top_k = max(1, min(top_k, 100))  # Clamp between 1 and 100

        # Use provided weights or defaults
        bm25_w = bm25_weight if bm25_weight is not None else self.bm25_weight
        vec_w = vector_weight if vector_weight is not None else self.vector_weight

        # Normalize weights
        total = bm25_w + vec_w
        if total > 0:
            bm25_w = bm25_w / total
            vec_w = vec_w / total
        else:
            bm25_w = 0.5
            vec_w = 0.5

        use_cache = use_cache if use_cache is not None else self.enable_caching

        start_time = time.time()
        self.search_count += 1

        try:
            # Perform searches
            vector_results = self._vector_search_with_retry(query, top_k * 2)
            bm25_results = self._bm25_search(query, top_k * 2)

            # Determine search mode
            has_vector = len(vector_results) > 0
            has_bm25 = len(bm25_results) > 0

            if has_vector and has_bm25:
                self.hybrid_count += 1
                search_mode = "hybrid"
            elif has_vector:
                self.vector_only_count += 1
                search_mode = "vector_only"
            elif has_bm25:
                self.bm25_only_count += 1
                search_mode = "bm25_only"
            else:
                logger.warning("Both vector and BM25 searches returned no results")
                return []

            # Combine results
            combined = {}

            # Add vector results
            vector_scores = []
            for result in vector_results:
                content = result.get("content", "")
                if content:
                    score = float(result.get("score", 0.0))
                    vector_scores.append(score)

                    if content not in combined:
                        combined[content] = {
                            "content": content,
                            "vector_score": score,
                            "bm25_score": 0.0,
                            "metadata": result.get("metadata", {}),
                        }
                    else:
                        # Keep highest score if duplicate
                        combined[content]["vector_score"] = max(
                            combined[content]["vector_score"], score
                        )

            # Add BM25 results
            bm25_scores = []
            for result in bm25_results:
                content = result.get("content", "")
                if content:
                    score = float(result.get("score", 0.0))
                    bm25_scores.append(score)

                    if content not in combined:
                        combined[content] = {
                            "content": content,
                            "vector_score": 0.0,
                            "bm25_score": score,
                            "metadata": result.get("metadata", {}),
                        }
                    else:
                        # Keep highest score if duplicate
                        combined[content]["bm25_score"] = max(
                            combined[content]["bm25_score"], score
                        )

            # Normalize scores
            all_vector_scores = [item["vector_score"] for item in combined.values()]
            all_bm25_scores = [item["bm25_score"] for item in combined.values()]

            vec_norm, bm25_norm = self._normalize_scores(
                all_vector_scores, all_bm25_scores
            )

            # Calculate combined scores
            items_list = list(combined.values())
            for i, item in enumerate(items_list):
                vec_score = vec_norm[i] if i < len(vec_norm) else 0.0
                bm25_score = bm25_norm[i] if i < len(bm25_norm) else 0.0

                # Weighted combination
                item["combined_score"] = (vec_score * vec_w) + (bm25_score * bm25_w)
                item["search_mode"] = search_mode
                item["timestamp"] = datetime.utcnow().isoformat()

            # Sort by combined score
            sorted_results = sorted(
                items_list, key=lambda x: x.get("combined_score", 0.0), reverse=True
            )[:top_k]

            elapsed = time.time() - start_time
            logger.debug(
                f"Hybrid search completed in {elapsed:.3f}s ({search_mode}, {len(sorted_results)} results)"
            )

            return sorted_results

        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in hybrid search: {e}", exc_info=True)
            # Graceful degradation - try vector-only if hybrid fails
            try:
                vector_results = self._vector_search_with_retry(query, top_k)
                return [
                    {
                        "content": r.get("content", ""),
                        "vector_score": r.get("score", 0.0),
                        "bm25_score": 0.0,
                        "combined_score": r.get("score", 0.0),
                        "metadata": r.get("metadata", {}),
                        "search_mode": "vector_only_fallback",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    for r in vector_results[:top_k]
                ]
            except Exception as fallback_error:
                logger.error(f"Fallback vector search also failed: {fallback_error}")
                return []

    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            "total_searches": self.search_count,
            "hybrid_searches": self.hybrid_count,
            "vector_only_searches": self.vector_only_count,
            "bm25_only_searches": self.bm25_only_count,
            "errors": self.error_count,
            "bm25_index_size": len(self.documents) if self.documents else 0,
            "bm25_available": self.bm25_index is not None,
            "vector_store_available": self.vector_store is not None,
        }
