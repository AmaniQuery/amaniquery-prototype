"""
Qdrant search interface for NiruSense-analyzed documents.
Provides filtered vector search using stored analysis metadata.
"""

from typing import List, Optional, Dict, Any
from .config import settings
from .embedding import embedding_generator
from .monitoring import logger


async def search_analyzed_documents(
    query: str,
    top_k: int = 10,
    topics: Optional[List[str]] = None,
    sentiment: Optional[str] = None,
    source: Optional[str] = None,
    min_quality_score: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Search NiruSense-analyzed documents with optional metadata filters.

    Args:
        query: Search query text
        top_k: Maximum number of results
        topics: Filter by topic labels
        sentiment: Filter by sentiment (positive/negative/neutral)
        source: Filter by document source
        min_quality_score: Minimum quality score filter

    Returns:
        List of result dicts with document text, analysis metadata, and score
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels
    except ImportError:
        logger.error("qdrant_client not available")
        return []

    if not embedding_generator.model:
        logger.warning("Embedding model not loaded, cannot search")
        return []

    try:
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=10,
        )

        query_vector = embedding_generator.embed(query)

        # Build metadata filter
        must_conditions = []
        if topics:
            must_conditions.append(
                qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="topics", match=qmodels.MatchAny(any=topics)
                        )
                    ]
                )
            )
        if sentiment:
            must_conditions.append(
                qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="sentiment", match=qmodels.MatchValue(value=sentiment)
                        )
                    ]
                )
            )
        if source:
            must_conditions.append(
                qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="source", match=qmodels.MatchValue(value=source)
                        )
                    ]
                )
            )
        if min_quality_score is not None:
            must_conditions.append(
                qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="quality_score",
                            range=qmodels.Range(gte=min_quality_score),
                        )
                    ]
                )
            )

        filter_obj = qmodels.Filter(must=must_conditions) if must_conditions else None

        results = client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filter_obj,
            with_payload=True,
        )

        output = []
        for point in results:
            payload = point.payload or {}
            output.append(
                {
                    "id": point.id,
                    "score": point.score,
                    "text": payload.get("text", ""),
                    "summary": payload.get("summary", ""),
                    "topics": payload.get("topics", []),
                    "sentiment": payload.get("sentiment", "unknown"),
                    "entities": payload.get("entities", []),
                    "quality_score": payload.get("quality_score", 0),
                    "language": payload.get("language", "unknown"),
                    "source": payload.get("source", "unknown"),
                    "timestamp": payload.get("timestamp"),
                }
            )

        return output

    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        return []
