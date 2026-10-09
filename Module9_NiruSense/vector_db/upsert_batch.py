from qdrant_client import QdrantClient
from qdrant_client.http import models
from transformers import AutoTokenizer, AutoModel
import torch
import os
from dotenv import load_dotenv
from typing import List, Dict, Iterator
import uuid
from tqdm import tqdm

load_dotenv()

# Configuration - Qdrant Cloud
QDRANT_URL = os.getenv("QDRANT_URL")  # Cloud-based, no localhost fallback
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "kenyan_sentiments_2025"
MODEL_NAME = "Davlan/afriberta-large"
BATCH_SIZE = 10000


class BatchProcessor:
    def __init__(self):
        print(f"Loading model {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings.tolist()

    def process_and_upsert(self, data_iterator: Iterator[Dict]):
        """
        Reads from data_iterator, encodes text, and upserts to Qdrant in batches.
        Expected data format:
        {
            "text": str,
            "sentiment_score": float,
            "label": str,
            "platform": str,
            "created_at": str (ISO format),
            "topic": str,
            "intensity": float,
            "is_sarcasm": bool,
            "id": str (optional, will generate UUID if missing)
        }
        """
        batch_points = []
        batch_texts = []

        print("Starting batch processing...")
        for item in tqdm(data_iterator):
            text = item.get("text", "")
            if not text:
                continue

            batch_texts.append(text)

            # Prepare payload
            payload = {
                "text": text,
                "sentiment_score": item.get("sentiment_score"),
                "label": item.get("label"),
                "platform": item.get("platform"),
                "created_at": item.get("created_at"),
                "topic": item.get("topic"),
                "intensity": item.get("intensity"),
                "is_sarcasm": item.get("is_sarcasm"),
            }

            point_id = item.get("id", str(uuid.uuid4()))

            # We store the point temporarily without vector, will add vector after batch encoding
            batch_points.append((point_id, payload))

            if len(batch_points) >= BATCH_SIZE:
                self._flush_batch(batch_points, batch_texts)
                batch_points = []
                batch_texts = []

        # Flush remaining
        if batch_points:
            self._flush_batch(batch_points, batch_texts)

    def _flush_batch(self, points_data, texts):
        try:
            vectors = self.encode_batch(texts)

            points = [
                models.PointStruct(id=pid, vector=vec, payload=payload)
                for (pid, payload), vec in zip(points_data, vectors)
            ]

            self.client.upsert(
                collection_name=COLLECTION_NAME, points=points, wait=True
            )
        except Exception as e:
            print(f"Error upserting batch: {e}")


if __name__ == "__main__":
    # Example usage with dummy data
    processor = BatchProcessor()

    def dummy_data_gen():
        for i in range(20):
            yield {
                "text": f"Hii ni test tweet number {i}",
                "sentiment_score": 0.5,
                "label": "Positive",
                "platform": "twitter",
                "created_at": "2025-11-27T10:00:00Z",
                "topic": "Politics",
                "intensity": 0.8,
                "is_sarcasm": False,
            }

    processor.process_and_upsert(dummy_data_gen())
