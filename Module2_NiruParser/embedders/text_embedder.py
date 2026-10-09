"""
Text Embedder using Sentence Transformers
"""

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
from tqdm import tqdm


def _auto_detect_device() -> str:
    """Auto-detect best device: CUDA > MPS > CPU"""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TextEmbedder:
    """Generate vector embeddings for text"""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize: bool = True,
        device: str = None,
    ):
        """
        Initialize embedder

        Args:
            model_name: Name of the Sentence Transformer model
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
            device: Device to run on ('cpu', 'cuda', 'mps', or None for auto-detect)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        device = device or _auto_detect_device()
        logger.info(f"Loading embedding model: {model_name} on {device}")

        import torch
        import os

        # Temporarily disable device_map to prevent meta tensor loading
        old_device_map = os.environ.get("HF_DEVICE_MAP", None)
        old_accelerate_device_map = os.environ.get("ACCELERATE_DEVICE_MAP", None)

        try:
            if old_device_map:
                del os.environ["HF_DEVICE_MAP"]
            if old_accelerate_device_map:
                del os.environ["ACCELERATE_DEVICE_MAP"]

            self.model = SentenceTransformer(model_name, device=device)

            try:
                test_embedding = self.model.encode(
                    "test", convert_to_numpy=True, show_progress_bar=False
                )
                logger.debug("Model loaded successfully and tested")
            except Exception as test_error:
                if "meta" in str(test_error).lower():
                    logger.warning(
                        f"Model on meta device, reinitializing: {test_error}"
                    )
                    import gc

                    del self.model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.model = SentenceTransformer(model_name, device=device)
                else:
                    raise

        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            try:
                logger.info("Attempting fallback model loading...")
                self.model = SentenceTransformer(model_name)
                if hasattr(self.model, "to"):
                    try:
                        self.model = self.model.to(device)
                    except Exception as to_error:
                        if "meta" not in str(to_error).lower():
                            raise
                        logger.warning(
                            f"Could not move model (meta tensor): {to_error}"
                        )
            except Exception as fallback_error:
                logger.error(f"Fallback model loading also failed: {fallback_error}")
                raise RuntimeError(f"Failed to load embedding model {model_name}: {e}")
        finally:
            if old_device_map:
                os.environ["HF_DEVICE_MAP"] = old_device_map
            if old_accelerate_device_map:
                os.environ["ACCELERATE_DEVICE_MAP"] = old_accelerate_device_map

        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.dimension)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for batch of texts

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=True,
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return np.zeros((len(texts), self.dimension))

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Add embeddings to chunk dictionaries

        Args:
            chunks: List of chunk dictionaries with 'text' field

        Returns:
            Chunks with added 'embedding' field
        """
        if not chunks:
            return []

        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        # Extract texts
        texts = [chunk.get("text", "") for chunk in chunks]

        # Generate embeddings
        embeddings = self.embed_batch(texts)

        # Add to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = (
                embedding.tolist()
            )  # Convert to list for JSON serialization

        logger.info("Embeddings generated successfully")
        return chunks

    def get_model_info(self) -> Dict:
        """Get information about the embedding model"""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_seq_length": self.model.max_seq_length,
            "normalize": self.normalize,
        }
