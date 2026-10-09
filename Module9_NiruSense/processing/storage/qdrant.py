from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from ..config import settings


class QdrantStorage:
    """Qdrant vector storage client"""

    def __init__(self):
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Qdrant client with retry logic"""
        try:
            self.client = QdrantClient(
                url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=30
            )
            self._ensure_collection()
            print(f"Qdrant connected: {settings.QDRANT_COLLECTION}")
        except Exception as e:
            print(f"Warning: Qdrant initialization failed: {e}")
            self.client = None

    def _ensure_collection(self):
        """Ensure collection exists with proper configuration"""
        if not self.client:
            return

        try:
            self.client.get_collection(settings.QDRANT_COLLECTION)
            print(f"Qdrant collection '{settings.QDRANT_COLLECTION}' exists")
        except Exception:
            # Collection doesn't exist, create it
            self.client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=768,  # nomic-embed-text-v1.5 dimension
                    distance=models.Distance.COSINE,
                ),
            )
            print(f"Created Qdrant collection: {settings.QDRANT_COLLECTION}")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def upsert(self, doc_id: str, vector: List[float], payload: Dict[str, Any]):
        """Upsert single document vector"""
        if not self.client:
            raise ConnectionError("Qdrant client not initialized")

        self.client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=[models.PointStruct(id=doc_id, vector=vector, payload=payload)],
        )

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def batch_upsert(self, points: List[Dict[str, Any]]):
        """Batch upsert multiple document vectors"""
        if not self.client:
            raise ConnectionError("Qdrant client not initialized")

        qdrant_points = [
            models.PointStruct(
                id=point["id"], vector=point["vector"], payload=point.get("payload", {})
            )
            for point in points
        ]

        # Process in batches
        batch_size = settings.QDRANT_BATCH_SIZE
        for i in range(0, len(qdrant_points), batch_size):
            batch = qdrant_points[i : i + batch_size]
            self.client.upsert(collection_name=settings.QDRANT_COLLECTION, points=batch)

    def health_check(self) -> Dict[str, Any]:
        """Check Qdrant health"""
        try:
            if not self.client:
                return {"status": "unhealthy", "error": "Client not initialized"}

            collection_info = self.client.get_collection(settings.QDRANT_COLLECTION)
            return {
                "status": "healthy",
                "collection": settings.QDRANT_COLLECTION,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


qdrant_storage = QdrantStorage()
