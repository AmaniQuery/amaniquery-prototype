"""
Intelligent Re-ranker Service - 2025 RAG Best Practices

Features:
- Cross-encoder re-ranking for improved relevance
- LLM-based relevance scoring
- Score normalization and filtering
- Async processing for performance
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import os

# Optional: sentence-transformers for cross-encoder
try:
    from sentence_transformers import CrossEncoder

    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers not installed, using LLM-based reranking")


@dataclass
class RankedDocument:
    """Document with relevance score."""

    document: Dict[str, Any]
    original_score: float
    rerank_score: float
    combined_score: float


class IntelligentReranker:
    """
    Intelligent document re-ranker using cross-encoder or LLM.

    Improves retrieval quality by re-scoring documents based on:
    - Query-document relevance (cross-encoder)
    - Contextual fit (LLM-based)
    - Freshness and source quality
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_llm_fallback: bool = True,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize re-ranker.

        Args:
            model_name: Cross-encoder model name
            use_llm_fallback: Use LLM if cross-encoder unavailable
            llm_client: Optional LLM client for LLM-based reranking
        """
        self.model_name = model_name
        self.use_llm_fallback = use_llm_fallback
        self.llm_client = llm_client

        # Initialize cross-encoder if available
        self.cross_encoder = None
        if CROSS_ENCODER_AVAILABLE:
            try:
                self.cross_encoder = CrossEncoder(model_name)
                logger.info(f"Initialized cross-encoder: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}")

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents based on relevance to query.

        Args:
            query: User query
            documents: Retrieved documents
            top_k: Number of documents to return
            min_score: Minimum score threshold

        Returns:
            Re-ranked list of documents
        """
        if not documents:
            return []

        if len(documents) <= top_k:
            return documents

        # Choose reranking method
        if self.cross_encoder:
            ranked_docs = await self._rerank_with_cross_encoder(query, documents)
        elif self.use_llm_fallback and self.llm_client:
            ranked_docs = await self._rerank_with_llm(query, documents)
        else:
            # Fallback: use original scores with simple heuristics
            ranked_docs = self._rerank_with_heuristics(query, documents)

        # Filter by minimum score and return top_k
        filtered = [d for d in ranked_docs if d.combined_score >= min_score]

        # Sort by combined score
        filtered.sort(key=lambda x: x.combined_score, reverse=True)

        # Return only the document dicts (not RankedDocument)
        return [d.document for d in filtered[:top_k]]

    async def _rerank_with_cross_encoder(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[RankedDocument]:
        """Re-rank using cross-encoder model."""
        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            text = doc.get("text", doc.get("content", ""))
            pairs.append([query, text[:512]])  # Limit text length

        # Run cross-encoder in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None, lambda: self.cross_encoder.predict(pairs)
        )

        # Create ranked documents
        ranked = []
        for i, doc in enumerate(documents):
            original_score = doc.get("score", 0.5)
            rerank_score = float(scores[i])

            # Normalize rerank score to 0-1
            rerank_score = (rerank_score + 3) / 6  # Typical range is -3 to 3
            rerank_score = max(0, min(1, rerank_score))

            # Combine scores (favor rerank score)
            combined = 0.3 * original_score + 0.7 * rerank_score

            ranked.append(
                RankedDocument(
                    document=doc,
                    original_score=original_score,
                    rerank_score=rerank_score,
                    combined_score=combined,
                )
            )

        return ranked

    async def _rerank_with_llm(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[RankedDocument]:
        """Re-rank using LLM relevance scoring."""
        prompt = f"""Rate the relevance of each document to the query on a scale of 0-10.

Query: {query}

Documents:
"""
        for i, doc in enumerate(documents[:10]):  # Limit to 10 docs for LLM
            text = doc.get("text", doc.get("content", ""))[:300]
            prompt += f"\n[{i+1}] {text}\n"

        prompt += '\nProvide scores as JSON: {"scores": [score1, score2, ...]}'

        try:
            # Call LLM
            response = await asyncio.to_thread(self._call_llm, prompt)

            # Parse scores
            import json

            scores_data = json.loads(response)
            scores = scores_data.get("scores", [5] * len(documents))

        except Exception as e:
            logger.warning(f"LLM reranking failed: {e}")
            scores = [5] * len(documents)

        # Create ranked documents
        ranked = []
        for i, doc in enumerate(documents):
            original_score = doc.get("score", 0.5)
            rerank_score = scores[i] / 10 if i < len(scores) else 0.5
            combined = 0.4 * original_score + 0.6 * rerank_score

            ranked.append(
                RankedDocument(
                    document=doc,
                    original_score=original_score,
                    rerank_score=rerank_score,
                    combined_score=combined,
                )
            )

        return ranked

    def _call_llm(self, prompt: str) -> str:
        """Call LLM for reranking."""
        if hasattr(self.llm_client, "chat"):
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            return response.choices[0].message.content
        return '{"scores": []}'

    def _rerank_with_heuristics(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[RankedDocument]:
        """Fallback: re-rank using simple heuristics."""
        query_terms = set(query.lower().split())

        ranked = []
        for doc in documents:
            text = doc.get("text", doc.get("content", "")).lower()

            original_score = doc.get("score", 0.5)

            # Heuristic: keyword overlap
            text_terms = set(text.split())
            overlap = len(query_terms & text_terms) / max(len(query_terms), 1)

            # Heuristic: query terms in title
            title = doc.get("title", "").lower()
            title_match = sum(1 for t in query_terms if t in title) / max(
                len(query_terms), 1
            )

            # Heuristic: document length (prefer moderate length)
            length_score = min(len(text) / 500, 1.0) * 0.5

            # Combined heuristic score
            rerank_score = overlap * 0.5 + title_match * 0.3 + length_score * 0.2

            combined = 0.5 * original_score + 0.5 * rerank_score

            ranked.append(
                RankedDocument(
                    document=doc,
                    original_score=original_score,
                    rerank_score=rerank_score,
                    combined_score=combined,
                )
            )

        return ranked


class QueryOptimizer:
    """
    Query optimization for improved retrieval.

    Techniques:
    - Query expansion with synonyms
    - HyDE (Hypothetical Document Embeddings)
    - Multi-query generation
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client

        # Kenya-specific synonyms
        self.synonyms = {
            "constitution": ["constitutional", "constitution of kenya", "katiba"],
            "parliament": ["bunge", "national assembly", "senate"],
            "court": ["judiciary", "high court", "supreme court"],
            "mp": ["member of parliament", "legislator", "mheshimiwa"],
            "president": ["head of state", "rais"],
            "county": ["devolution", "county government"],
            "bill": ["legislation", "act", "law"],
            "tax": ["taxation", "levy", "revenue"],
        }

    def expand_query(self, query: str) -> str:
        """Expand query with synonyms."""
        expanded = query
        query_lower = query.lower()

        for term, synonyms in self.synonyms.items():
            if term in query_lower:
                # Add first synonym
                expanded += f" {synonyms[0]}"

        return expanded

    async def generate_multi_queries(self, query: str, n: int = 3) -> List[str]:
        """Generate multiple query variants."""
        if not self.llm_client:
            # Simple variant generation
            return [query, self.expand_query(query), f"{query} Kenya"][:n]

        prompt = f"""Generate {n} different search queries for:
"{query}"

Make each query focus on different aspects or use different terms.
Return as JSON: {{"queries": ["q1", "q2", "q3"]}}"""

        try:
            response = await asyncio.to_thread(self._call_llm, prompt)
            import json

            data = json.loads(response)
            return data.get("queries", [query])[:n]
        except Exception:
            return [query, self.expand_query(query)]

    async def hyde_transform(self, query: str) -> str:
        """
        HyDE: Generate hypothetical document for query.

        The embedding of the hypothetical document often matches
        better with actual relevant documents.
        """
        if not self.llm_client:
            return query

        prompt = f"""Write a short paragraph that would be a perfect answer to:
"{query}"

Write as if it's from an authoritative Kenyan legal document.
Keep it under 150 words."""

        try:
            response = await asyncio.to_thread(self._call_llm, prompt)
            return response[:500]  # Limit length
        except Exception as e:
            logger.warning(f"HyDE transform failed: {e}")
            return query

    def _call_llm(self, prompt: str) -> str:
        """Call LLM."""
        if hasattr(self.llm_client, "chat"):
            response = self.llm_client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
            )
            return response.choices[0].message.content
        return prompt


# Export
__all__ = ["IntelligentReranker", "QueryOptimizer", "RankedDocument"]
