import time
import re
import json
import hashlib
import numpy as np
from typing import Dict, Optional, List, Any
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

_STOP = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
}


def _normalize(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\s+", " ", q)
    return " ".join(w for w in q.split() if w not in _STOP and len(w) > 1)


class SemanticCache:
    """
    Semantic Cache for RAG Pipeline.
    Uses cosine similarity to find cached responses for similar queries.
    """

    def __init__(self, threshold: float = 0.85, max_size: int = 1000):
        self.cache: List[Dict[str, Any]] = []
        self.threshold = threshold
        self.max_size = max_size
        self.embedding_model = None

    def set_embedding_model(self, model):
        """Set the embedding model to be used for query encoding"""
        self.embedding_model = model

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text using the vector store's model"""
        if not self.embedding_model:
            return None
        try:
            # Assume model has encode method (SentenceTransformer)
            embedding = self.embedding_model.encode(text)
            if isinstance(embedding, list):
                return np.array(embedding)
            return embedding
        except Exception as e:
            logger.warning(f"Failed to generate embedding for cache: {e}")
            return None

    def get(
        self, query: str, top_k: int = 5, category: Optional[str] = None
    ) -> Optional[Dict]:
        """Retrieve cached result if semantically similar query exists"""
        if not self.embedding_model:
            return None

        # Try normalized exact match first
        q_norm = _normalize(query)
        for entry in self.cache:
            if entry.get("_norm") == q_norm:
                if entry.get("top_k") != top_k:
                    continue
                if category and entry.get("category") != category:
                    continue
                logger.info(f"Semantic cache normalized hit for: '{query}'")
                return entry["result"]

        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            return None

        best_score = -1
        best_entry = None

        for entry in self.cache:
            if entry.get("top_k") != top_k:
                continue
            if category and entry.get("category") != category:
                continue

            score = cosine_similarity(
                query_embedding.reshape(1, -1), entry["embedding"].reshape(1, -1)
            )[0][0]

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.threshold:
            logger.info(
                f"Semantic cache hit! Score: {best_score:.4f} for query: '{query}'"
            )
            return best_entry["result"]

        return None

    def set(
        self, query: str, result: Dict, top_k: int = 5, category: Optional[str] = None
    ):
        """Cache the result for a query"""
        if not self.embedding_model:
            return

        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            return

        # Evict if full (Simple LRU - remove first)
        if len(self.cache) >= self.max_size:
            self.cache.pop(0)

        entry = {
            "query": query,
            "_norm": _normalize(query),
            "embedding": query_embedding,
            "result": result,
            "top_k": top_k,
            "category": category,
            "timestamp": time.time(),
        }
        self.cache.append(entry)
