"""
Vector Store Adapter for Hybrid Embeddings

Adapts existing vector store to work with hybrid encoder embeddings
and supports indexing diffusion-generated documents.
"""

import torch
import numpy as np
import time
from typing import List, Dict, Optional, Any
from loguru import logger
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Module3_NiruDB.vector_store import VectorStore
from ..hybrid_encoder import HybridEncoder
from ..config import HybridEncoderConfig


class HybridVectorStoreAdapter:
    """Adapter for vector store with hybrid encoder support"""

    def __init__(
        self,
        vector_store: VectorStore,
        hybrid_encoder: Optional[HybridEncoder] = None,
        use_hybrid: bool = True,
        fallback_to_original: bool = True,
    ):
        """
        Initialize adapter

        Args:
            vector_store: Existing vector store instance
            hybrid_encoder: Hybrid encoder for enhanced embeddings
            use_hybrid: Whether to use hybrid encoder by default
            fallback_to_original: Fallback to original encoder if hybrid fails
        """
        self.vector_store = vector_store
        self.hybrid_encoder = hybrid_encoder
        self.use_hybrid = use_hybrid
        self.fallback_to_original = fallback_to_original

        # Statistics
        self.hybrid_encodings = 0
        self.fallback_encodings = 0

    def encode(self, text: str, use_hybrid: Optional[bool] = None) -> np.ndarray:
        """
        Encode text to embeddings

        Args:
            text: Input text
            use_hybrid: Whether to use hybrid encoder (overrides default)

        Returns:
            embeddings: Text embeddings as numpy array
        """
        use_hybrid = use_hybrid if use_hybrid is not None else self.use_hybrid

        if use_hybrid and self.hybrid_encoder is not None:
            try:
                # Use hybrid encoder
                # First, get base embeddings from the vector store's embedding model
                if hasattr(self.vector_store, "embedding_model"):
                    base_embeddings = self.vector_store.embedding_model.encode(text)
                    base_embeddings_tensor = torch.tensor(
                        base_embeddings, dtype=torch.float32
                    )

                    # Reshape if needed: [embed_dim] -> [1, embed_dim] or [1, seq_len, embed_dim]
                    if base_embeddings_tensor.ndim == 1:
                        # If it's a pooled embedding, expand to [1, embed_dim]
                        base_embeddings_tensor = base_embeddings_tensor.unsqueeze(0)

                    # Ensure it's [1, seq_len, embed_dim] format
                    if base_embeddings_tensor.ndim == 2:
                        # [1, embed_dim] -> [1, 1, embed_dim] for sequence processing
                        base_embeddings_tensor = base_embeddings_tensor.unsqueeze(1)

                    # Check if sequence length is too short for convolutional layers
                    # Convolutional layers need seq_len > 1, so if we have [1, 1, dim], expand it
                    if base_embeddings_tensor.shape[1] == 1:
                        # Repeat the embedding to create a minimal sequence (seq_len=2)
                        # This avoids the "Expected more than 1 value per channel" error
                        base_embeddings_tensor = base_embeddings_tensor.repeat(1, 2, 1)

                    # Pass base embeddings to hybrid encoder
                    with torch.no_grad():
                        hybrid_output = self.hybrid_encoder.forward(
                            embeddings=base_embeddings_tensor
                        )

                        # Pool if needed (mean pooling)
                        if hybrid_output.ndim == 3:
                            embeddings = hybrid_output.mean(
                                dim=1
                            )  # [1, seq_len, dim] -> [1, dim]
                        else:
                            embeddings = hybrid_output

                        embeddings_np = embeddings.cpu().numpy()
                else:
                    # Fallback if no embedding model
                    raise ValueError(
                        "Vector store has no embedding model for base embeddings"
                    )

                # Ensure correct shape and dimension
                if embeddings_np.ndim > 1:
                    embeddings_np = embeddings_np.flatten()

                # Ensure compatibility with vector store dimension
                if hasattr(self.vector_store, "embedding_model"):
                    expected_dim = (
                        self.vector_store.embedding_model.get_sentence_embedding_dimension()
                    )
                    if embeddings_np.shape[-1] != expected_dim:
                        logger.warning(
                            f"Embedding dimension mismatch: "
                            f"got {embeddings_np.shape[-1]}, expected {expected_dim}. "
                            f"Using original encoder."
                        )
                        return self._encode_fallback(text)

                self.hybrid_encodings += 1
                return embeddings_np  # Return 1D array

            except Exception as e:
                logger.warning(f"Hybrid encoding failed: {e}, using fallback")
                if self.fallback_to_original:
                    return self._encode_fallback(text)
                else:
                    raise
        else:
            return self._encode_fallback(text)

    def _encode_fallback(self, text: str) -> np.ndarray:
        """Fallback to original encoder"""
        if hasattr(self.vector_store, "embedding_model"):
            embeddings = self.vector_store.embedding_model.encode(text)
            self.fallback_encodings += 1
            return np.array(embeddings)
        else:
            raise ValueError("No encoder available")

    def add_documents(
        self,
        chunks: List[Dict],
        use_hybrid: Optional[bool] = None,
        batch_size: int = 100,
    ):
        """
        Add documents with hybrid embeddings

        Args:
            chunks: List of chunk dictionaries
            use_hybrid: Whether to use hybrid encoder
            batch_size: Batch size for processing
        """
        use_hybrid = use_hybrid if use_hybrid is not None else self.use_hybrid

        # Process chunks and add embeddings
        processed_chunks = []

        for chunk in chunks:
            text = chunk.get("text", "")
            if not text:
                continue

            # Encode with hybrid encoder
            embedding = self.encode(text, use_hybrid=use_hybrid)

            # Add embedding to chunk
            chunk["embedding"] = embedding.tolist()
            processed_chunks.append(chunk)

        # Add to vector store
        self.vector_store.add_documents(processed_chunks, batch_size=batch_size)

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter: Optional[Dict] = None,
        use_hybrid: Optional[bool] = None,
    ) -> List[Dict]:
        """
        Query with hybrid embeddings

        Args:
            query_text: Query text
            n_results: Number of results
            filter: Optional metadata filter
            use_hybrid: Whether to use hybrid encoder

        Returns:
            results: List of retrieved documents
        """
        use_hybrid = use_hybrid if use_hybrid is not None else self.use_hybrid

        # Encode query
        query_embedding = self.encode(query_text, use_hybrid=use_hybrid)

        # Query vector store
        # Note: This assumes vector_store.query can accept embeddings directly
        # If not, we may need to modify the query method
        if hasattr(self.vector_store, "query"):
            # Try to use embedding directly if supported
            try:
                results = self.vector_store.query(
                    query_text=query_text,  # Some stores need text for metadata
                    n_results=n_results,
                    filter=filter,
                )
                return results
            except Exception as e:
                logger.warning(f"Query with embedding failed: {e}, using text query")
                return self.vector_store.query(
                    query_text=query_text, n_results=n_results, filter=filter
                )
        else:
            raise ValueError("Vector store does not support query method")

    def add_diffusion_generated_documents(
        self,
        generated_texts: List[str],
        metadata: Optional[List[Dict]] = None,
        use_hybrid: bool = True,
    ):
        """
        Add diffusion-generated documents to vector store

        Args:
            generated_texts: List of generated text documents
            metadata: Optional metadata for each document
            use_hybrid: Whether to use hybrid encoder
        """
        chunks = []

        for i, text in enumerate(generated_texts):
            doc_metadata = metadata[i] if metadata and i < len(metadata) else {}

            # Create chunk from generated text
            chunk = {
                "text": text,
                "chunk_id": f"diffusion_gen_{i}_{int(time.time())}",
                "title": doc_metadata.get("title", f"Generated Document {i}"),
                "category": doc_metadata.get("category", "Generated"),
                "source_name": "diffusion_model",
                "source_url": "",
                "metadata": {
                    **doc_metadata,
                    "generated": True,
                    "generation_method": "diffusion",
                },
            }
            chunks.append(chunk)

        # Add with hybrid embeddings
        self.add_documents(chunks, use_hybrid=use_hybrid)
        logger.info(
            f"Added {len(chunks)} diffusion-generated documents to vector store"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics"""
        stats = {
            "hybrid_encodings": self.hybrid_encodings,
            "fallback_encodings": self.fallback_encodings,
            "total_encodings": self.hybrid_encodings + self.fallback_encodings,
            "hybrid_ratio": (
                self.hybrid_encodings
                / (self.hybrid_encodings + self.fallback_encodings)
                if (self.hybrid_encodings + self.fallback_encodings) > 0
                else 0.0
            ),
        }

        # Add vector store stats if available
        if hasattr(self.vector_store, "get_stats"):
            stats["vector_store"] = self.vector_store.get_stats()

        return stats
