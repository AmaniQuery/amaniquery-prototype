"""
Aggressive Semantic Cache — maximizes cache hit rate with query normalization,
lowered similarity thresholds, embedding integration, and auto-write on response.
"""

import os
import re
import time
import json
import hashlib
import threading
import numpy as np
from typing import Optional, Dict, List, Any, Callable
from collections import OrderedDict
from loguru import logger
from dataclasses import dataclass, field

STOP_WORDS = {
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
    "need",
    "dare",
    "ought",
    "used",
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
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "out",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "because",
    "but",
    "and",
    "or",
    "if",
    "while",
    "although",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "he",
    "she",
    "they",
    "them",
    "his",
    "her",
    "their",
    "please",
    "tell",
    "about",
}


def normalize_query(query: str) -> str:
    """Aggressively normalize a query for cache key matching."""
    q = query.lower().strip()
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\s+", " ", q)
    words = q.split()
    words = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    return " ".join(words)


def query_hash(query: str) -> str:
    """Deterministic hash for exact cache lookup."""
    return hashlib.md5(query.strip().lower().encode()).hexdigest()


# Common Kenyan legal queries for cache pre-warming
PREDICTIVE_QUERIES = [
    "what does article 40 say about property rights",
    "what is the finance bill 2025",
    "what does the constitution say about freedom of speech",
    "what are my rights as a tenant in kenya",
    "how do i register a business in kenya",
    "what is the housing levy",
    "what is shif",
    "what are the requirements for presidential elections in kenya",
    "what does article 50 say about fair hearing",
    "what is the digital services tax",
    "how does the finance bill housing levy align with the constitution",
    "what are the grounds for divorce in kenya",
    "what is the minimum wage in kenya",
    "how to get a passport in kenya",
    "what are the laws on abortion in kenya",
    "what is the retirement age in kenya",
    "what are the traffic laws in kenya",
    "what is the land control act",
    "what are the employment rights in kenya",
    "what is the marriage act 2014",
    "what is the penalty for tax evasion in kenya",
    "what does article 27 say about equality",
    "how to register a marriage in kenya",
    "what are the nhif rates",
    "what is the children act",
    "what are the nssf rates in kenya",
    "what is the data protection act 2019",
    "what is the electoral code of conduct",
    "what are the laws on defamation in kenya",
    "what is the budget 2025 allocation for education",
    "what are the legal requirements for a will in kenya",
    "what is the companies act 2015",
    "what is the competition act",
    "what are the laws on hate speech in kenya",
    "what is the public procurement act",
    "what is the moveable property security rights act",
]


@dataclass
class CacheEntry:
    """Single cache entry with metadata for frequency-based promotion."""

    query: str
    normalized: str
    result: Any
    embedding: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    access_count: int = 1
    ttl: int = 86400
    category: str = "default"


class AggressiveSemanticCache:
    """
    Multi-strategy cache that chains exact → normalized → semantic lookup.
    Writes every response to all three tiers automatically.
    """

    def __init__(
        self,
        exact_capacity: int = 2000,
        semantic_capacity: int = 2000,
        semantic_threshold: float = 0.85,
        legal_threshold: float = 0.82,
        default_ttl: int = 86400,
    ):
        self.exact: Dict[str, CacheEntry] = OrderedDict()
        self.exact_capacity = exact_capacity

        self.semantic: List[CacheEntry] = []
        self.semantic_capacity = semantic_capacity
        self.semantic_threshold = semantic_threshold
        self.legal_threshold = legal_threshold

        self._embedding_model = None
        self._embedding_lock = threading.Lock()
        self._lock = threading.RLock()

        self.stats = {
            "exact_hits": 0,
            "normalized_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "total": 0,
        }

        logger.info(
            f"AggressiveSemanticCache: exact={exact_capacity}, "
            f"semantic={semantic_capacity}, threshold={semantic_threshold}, "
            f"legal_threshold={legal_threshold}"
        )

    def _get_embedder(self):
        """Lazy-load the SentenceTransformer embedder."""
        if self._embedding_model is not None:
            return self._embedding_model
        with self._embedding_lock:
            if self._embedding_model is not None:
                return self._embedding_model
            try:
                from sentence_transformers import SentenceTransformer

                model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
                logger.info(f"Loading embedding model for cache: {model_name}")
                self._embedding_model = SentenceTransformer(model_name)
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self._embedding_model = False
        return self._embedding_model if self._embedding_model is not False else None

    def _embed(self, text: str) -> Optional[np.ndarray]:
        """Get embedding vector for text."""
        model = self._get_embedder()
        if model is None:
            return None
        try:
            return model.encode(text, normalize_embeddings=True)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _is_legal_query(self, query: str) -> bool:
        q = query.lower()
        kw = {
            "constitution",
            "article",
            "act",
            "bill",
            "law",
            "legal",
            "court",
            "rights",
            "section",
            "clause",
            "statute",
            "judgment",
            "penalty",
            "offence",
            "offense",
            "marriage",
            "divorce",
            "tenancy",
            "lease",
            "contract",
            "tort",
            "litigation",
        }
        return any(k in q for k in kw)

    def _get_ttl(self, query: str) -> int:
        if self._is_legal_query(query):
            return 86400 * 90  # 90 days for legal
        if any(w in query.lower() for w in ["news", "today", "latest", "breaking"]):
            return 7200  # 2 hours for news
        return 86400  # 24 hours default

    def get(self, query: str) -> Optional[dict]:
        """
        Three-tier lookup:
          1. Exact match (hash key)
          2. Normalized match
          3. Semantic similarity
        """
        with self._lock:
            self.stats["total"] += 1

        q_lower = query.strip().lower()
        q_hash = query_hash(query)
        q_norm = normalize_query(query)

        # Tier 1: Exact match
        with self._lock:
            entry = self.exact.get(q_hash)
            if entry and self._valid(entry):
                entry.access_count += 1
                self.exact.move_to_end(q_hash)
                self.stats["exact_hits"] += 1
                logger.debug(f"Exact cache hit: {query[:50]}...")
                return entry.result
            elif entry and not self._valid(entry):
                del self.exact[q_hash]

        # Tier 2: Normalized match across all cached queries
        with self._lock:
            for key, entry in list(self.exact.items()):
                if not self._valid(entry):
                    continue
                if entry.normalized == q_norm:
                    entry.access_count += 1
                    self.exact.move_to_end(key)
                    self.stats["normalized_hits"] += 1
                    logger.debug(f"Normalized cache hit: {query[:50]}...")
                    return entry.result

        # Tier 3: Semantic similarity
        query_emb = self._embed(query)
        if query_emb is not None:
            threshold = (
                self.legal_threshold
                if self._is_legal_query(query)
                else self.semantic_threshold
            )
            with self._lock:
                best_score = -1.0
                best_entry = None
                for entry in self.semantic:
                    if not self._valid(entry) or entry.embedding is None:
                        continue
                    score = self._cosine_sim(query_emb, entry.embedding)
                    if score > best_score:
                        best_score = score
                        best_entry = entry

                if best_score >= threshold and best_entry is not None:
                    best_entry.access_count += 1
                    self.stats["semantic_hits"] += 1
                    logger.info(
                        f"Semantic cache hit: {query[:50]}... "
                        f"(score={best_score:.3f}, threshold={threshold})"
                    )
                    return best_entry.result

        self.stats["misses"] += 1
        return None

    def set(self, query: str, result: Any):
        """Write to all three tiers automatically."""
        q_lower = query.strip().lower()
        q_hash = query_hash(query)
        q_norm = normalize_query(query)
        ttl = self._get_ttl(query)
        query_emb = self._embed(query)

        entry = CacheEntry(
            query=q_lower,
            normalized=q_norm,
            result=result,
            embedding=query_emb,
            ttl=ttl,
            category="legal" if self._is_legal_query(query) else "default",
        )

        with self._lock:
            # Exact cache (LRU)
            self.exact[q_hash] = entry
            self.exact.move_to_end(q_hash)
            while len(self.exact) > self.exact_capacity:
                self.exact.popitem(last=False)

            # Semantic cache
            if query_emb is not None:
                self.semantic.append(entry)
                while len(self.semantic) > self.semantic_capacity:
                    self.semantic.pop(0)

        logger.debug(f"Cached: {query[:50]}... (ttl={ttl}s)")

    def _valid(self, entry: CacheEntry) -> bool:
        return (time.time() - entry.timestamp) < entry.ttl

    def get_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """Return top-k similar cached queries (for analytics/warmup)."""
        query_emb = self._embed(query)
        if query_emb is None:
            return []

        results = []
        with self._lock:
            for entry in self.semantic:
                if entry.embedding is None:
                    continue
                score = self._cosine_sim(query_emb, entry.embedding)
                results.append(
                    {
                        "query": entry.query,
                        "score": score,
                        "access_count": entry.access_count,
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def warmup(self, queries: Optional[List[str]] = None):
        """
        Pre-populate cache with common Kenyan legal queries.
        Stores placeholder entries so first real request with similar wording gets a semantic hit.
        """
        if queries is None:
            queries = PREDICTIVE_QUERIES

        logger.info(f"Warming cache with {len(queries)} common queries...")
        warmed = 0
        for q in queries:
            q_hash = query_hash(q)
            if q_hash not in self.exact:
                self.set(q, {"answer": None, "sources": [], "_warmup": True})
                warmed += 1
        logger.info(f"Cache warmup complete: {warmed} entries added")

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.stats["total"] or 1
            hits = (
                self.stats["exact_hits"]
                + self.stats["normalized_hits"]
                + self.stats["semantic_hits"]
            )
            return {
                **self.stats,
                "hit_rate": hits / total,
                "exact_size": len(self.exact),
                "semantic_size": len(self.semantic),
            }

    def clear_expired(self):
        now = time.time()
        with self._lock:
            expired_exact = [k for k, v in self.exact.items() if not self._valid(v)]
            for k in expired_exact:
                del self.exact[k]
            self.semantic = [e for e in self.semantic if self._valid(e)]
        logger.info(f"Cleared {len(expired_exact)} exact + semantic expired entries")


# Global singleton
_instance: Optional[AggressiveSemanticCache] = None
_instance_lock = threading.Lock()


def get_aggressive_cache() -> AggressiveSemanticCache:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = AggressiveSemanticCache()
    return _instance
