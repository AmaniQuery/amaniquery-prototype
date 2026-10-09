"""
Embedding module for NiruSense using nomic-embed-text-v1.5.
Generates 768-dimensional vectors for document search.
"""

from typing import List, Union
from sentence_transformers import SentenceTransformer
from .config import settings
from .monitoring import logger


class EmbeddingGenerator:
    """Generate embeddings using nomic-embed-text-v1.5"""

    def __init__(self):
        self.model = None
        self.model_name = settings.MODEL_EMBEDDING
        self._load_model()

    def _load_model(self):
        """Load embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
            logger.info(f"Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for single text

        Args:
            text: Text to embed

        Returns:
            768-dimensional vector
        """
        if not self.model:
            logger.warning("Embedding model not loaded, returning zero vector")
            return [0.0] * 768

        try:
            # Truncate to reasonable length
            truncated_text = text[:8192]  # nomic-embed supports 8192 tokens

            # Generate embedding
            embedding = self.model.encode(
                truncated_text, convert_to_numpy=True, show_progress_bar=False
            )

            return embedding.tolist()

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return [0.0] * 768

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            List of 768-dimensional vectors
        """
        if not self.model:
            logger.warning("Embedding model not loaded, returning zero vectors")
            return [[0.0] * 768] * len(texts)

        try:
            # Truncate all texts
            truncated_texts = [text[:8192] for text in texts]

            # Generate embeddings in batches
            embeddings = self.model.encode(
                truncated_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            return [emb.tolist() for emb in embeddings]

        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            return [[0.0] * 768] * len(texts)


# Global embedding generator instance
embedding_generator = EmbeddingGenerator()
