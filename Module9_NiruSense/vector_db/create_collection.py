from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration - Qdrant Cloud
QDRANT_URL = os.getenv("QDRANT_URL")  # Cloud-based, no localhost fallback
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "kenyan_sentiments_2025"
VECTOR_SIZE = 1024  # Davlan/afriberta-large


def create_collection():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    print(f"Checking if collection '{COLLECTION_NAME}' exists...")
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return

    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
            on_disk=True,  # Store vectors on disk to save RAM for 5M+ points
        ),
        hnsw_config=models.HnswConfigDiff(
            m=32,
            ef_construct=256,  # Higher build time accuracy
            full_scan_threshold=10000,
            max_indexing_threads=0,
            on_disk=True,  # HNSW graph on disk
            payload_m=16,
        ),
        optimizers_config=models.OptimizersConfigDiff(
            default_segment_number=2, memmap_threshold=20000
        ),
    )

    # Create Payload Indices for fast filtering
    print("Creating payload indices...")
    indices = [
        ("created_at", models.PayloadSchemaType.DATETIME),
        ("sentiment_score", models.PayloadSchemaType.FLOAT),
        ("platform", models.PayloadSchemaType.KEYWORD),
        ("topic", models.PayloadSchemaType.KEYWORD),
        ("is_sarcasm", models.PayloadSchemaType.BOOL),
    ]

    for field_name, field_type in indices:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=field_type,
        )
        print(f"Index created for '{field_name}'")

    print(f"Collection '{COLLECTION_NAME}' created successfully!")


if __name__ == "__main__":
    create_collection()
