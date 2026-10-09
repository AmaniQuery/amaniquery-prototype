from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")  # Cloud-based, no localhost fallback
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = "kenyan_sentiments_2025"


class DriftDetector:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    def get_uncertain_samples(self, limit=200):
        """
        Retrieves samples where the model was uncertain.
        Uncertainty proxy: Sentiment score close to 0 (e.g., -0.4 to 0.4).
        """
        print(f"Scanning {COLLECTION_NAME} for uncertain samples...")

        # We want points where sentiment_score is between -0.4 and 0.4
        # Qdrant Range filter
        uncertainty_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="sentiment_score", range=models.Range(gte=-0.4, lte=0.4)
                )
            ]
        )

        # We can scroll or search. Scroll is better for retrieving existing points without a query vector.
        # But we want "top" uncertain? Maybe just random or recent?
        # Let's get recent ones.

        results, _ = self.client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=uncertainty_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        samples = []
        for point in results:
            payload = point.payload
            samples.append(
                {
                    "text": payload.get("text"),
                    "score": payload.get("sentiment_score"),
                    "id": point.id,
                    "created_at": payload.get("created_at"),
                }
            )

        print(f"Found {len(samples)} uncertain samples.")
        return samples


if __name__ == "__main__":
    detector = DriftDetector()
    samples = detector.get_uncertain_samples()
    for s in samples[:5]:
        print(s)
