"""
Stream Processor for Real-time Query and Data Processing

Processes streaming queries and generated data in real-time with
async support and batch processing.
"""

import asyncio
import torch
from typing import List, Dict, Optional, Any, Callable, AsyncIterator
from datetime import datetime
import time
from loguru import logger

from .stream_buffer import StreamBuffer, AsyncStreamBuffer, StreamItem
from ..config import StreamingConfig, default_config
from ..hybrid_encoder import HybridEncoder
from ..diffusion.text_diffusion import TextDiffusionModel
from ..diffusion.embedding_diffusion import EmbeddingDiffusionModel


class StreamProcessor:
    """Processes streaming queries and generated data"""

    def __init__(
        self,
        hybrid_encoder: Optional[HybridEncoder] = None,
        text_diffusion: Optional[TextDiffusionModel] = None,
        embedding_diffusion: Optional[EmbeddingDiffusionModel] = None,
        batch_size: int = 32,
        max_concurrent_streams: int = 10,
        processing_timeout: float = 30.0,
        config: Optional[StreamingConfig] = None,
    ):
        if config is not None:
            batch_size = config.batch_size
            max_concurrent_streams = config.max_concurrent_streams
            processing_timeout = config.processing_timeout

        self.hybrid_encoder = hybrid_encoder
        self.text_diffusion = text_diffusion
        self.embedding_diffusion = embedding_diffusion

        self.batch_size = batch_size
        self.max_concurrent_streams = max_concurrent_streams
        self.processing_timeout = processing_timeout

        # Buffers
        self.query_buffer = StreamBuffer(batch_size=batch_size)
        self.data_buffer = StreamBuffer(batch_size=batch_size)

        # Processing state
        self.active_streams = 0
        self.processed_queries = 0
        self.processed_data = 0

    def process_query(
        self, query_id: str, query_text: str, metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process a single query

        Args:
            query_id: Unique query identifier
            query_text: Query text
            metadata: Optional metadata

        Returns:
            result: Processing result
        """
        start_time = time.time()

        try:
            # Encode query
            if self.hybrid_encoder is not None:
                with torch.no_grad():
                    query_emb = self.hybrid_encoder.encode(
                        text=query_text, return_pooled=True
                    )
            else:
                # Fallback: return text as-is
                query_emb = None

            result = {
                "query_id": query_id,
                "query_text": query_text,
                "embeddings": (
                    query_emb.cpu().tolist() if query_emb is not None else None
                ),
                "metadata": metadata or {},
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
            }

            self.processed_queries += 1
            return result

        except Exception as e:
            logger.error(f"Error processing query {query_id}: {e}")
            return {
                "query_id": query_id,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def process_query_batch(
        self, queries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process batch of queries

        Args:
            queries: List of query dictionaries with 'id' and 'text'

        Returns:
            results: List of processing results
        """
        results = []

        for query in queries:
            result = self.process_query(
                query_id=query.get("id", f"query_{len(results)}"),
                query_text=query.get("text", ""),
                metadata=query.get("metadata"),
            )
            results.append(result)

        return results

    def process_generated_data(
        self,
        data_id: str,
        data: Any,
        data_type: str = "text",  # "text" or "embedding"
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Process generated data

        Args:
            data_id: Unique data identifier
            data: Generated data (text or embeddings)
            data_type: Type of data
            metadata: Optional metadata

        Returns:
            result: Processing result
        """
        start_time = time.time()

        try:
            result = {
                "data_id": data_id,
                "data_type": data_type,
                "data": data,
                "metadata": metadata or {},
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
            }

            # Encode if text
            if data_type == "text" and self.hybrid_encoder is not None:
                with torch.no_grad():
                    embeddings = self.hybrid_encoder.encode(
                        text=data, return_pooled=True
                    )
                    result["embeddings"] = embeddings.cpu().tolist()

            self.processed_data += 1
            return result

        except Exception as e:
            logger.error(f"Error processing data {data_id}: {e}")
            return {
                "data_id": data_id,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def add_query_to_buffer(
        self, query_id: str, query_text: str, metadata: Optional[Dict] = None
    ):
        """Add query to buffer for batch processing"""
        self.query_buffer.add(
            item_id=query_id,
            data={"text": query_text, "metadata": metadata},
            metadata={"type": "query"},
        )

    def add_data_to_buffer(
        self,
        data_id: str,
        data: Any,
        data_type: str = "text",
        metadata: Optional[Dict] = None,
    ):
        """Add generated data to buffer"""
        self.data_buffer.add(
            item_id=data_id,
            data={"data": data, "type": data_type, "metadata": metadata},
            metadata={"type": "generated_data"},
        )

    def process_buffered_queries(self) -> List[Dict[str, Any]]:
        """Process all buffered queries"""
        batch = self.query_buffer.get_batch()
        if not batch:
            return []

        queries = [
            {
                "id": item.id,
                "text": item.data["text"],
                "metadata": item.data.get("metadata"),
            }
            for item in batch
        ]

        results = self.process_query_batch(queries)
        self.query_buffer.clear_processed()

        return results

    def process_buffered_data(self) -> List[Dict[str, Any]]:
        """Process all buffered generated data"""
        batch = self.data_buffer.get_batch()
        if not batch:
            return []

        results = []
        for item in batch:
            result = self.process_generated_data(
                data_id=item.id,
                data=item.data["data"],
                data_type=item.data["type"],
                metadata=item.data.get("metadata"),
            )
            results.append(result)

        self.data_buffer.clear_processed()
        return results

    async def process_query_stream(
        self, query_stream: AsyncIterator[str]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process streaming queries asynchronously

        Args:
            query_stream: Async iterator of query texts

        Yields:
            result: Processing result for each query
        """
        async for query_text in query_stream:
            query_id = f"query_{int(time.time() * 1000)}"
            result = self.process_query(query_id, query_text)
            yield result

    async def process_data_stream(
        self, data_stream: AsyncIterator[Dict[str, Any]]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process streaming generated data asynchronously

        Args:
            data_stream: Async iterator of data dictionaries

        Yields:
            result: Processing result for each data item
        """
        async for data_item in data_stream:
            data_id = data_item.get("id", f"data_{int(time.time() * 1000)}")
            data = data_item.get("data")
            data_type = data_item.get("type", "text")
            metadata = data_item.get("metadata")

            result = self.process_generated_data(data_id, data, data_type, metadata)
            yield result

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "processed_queries": self.processed_queries,
            "processed_data": self.processed_data,
            "active_streams": self.active_streams,
            "query_buffer": self.query_buffer.get_stats(),
            "data_buffer": self.data_buffer.get_stats(),
        }


class AsyncStreamProcessor:
    """Async version of stream processor"""

    def __init__(
        self,
        hybrid_encoder: Optional[HybridEncoder] = None,
        text_diffusion: Optional[TextDiffusionModel] = None,
        embedding_diffusion: Optional[EmbeddingDiffusionModel] = None,
        batch_size: int = 32,
        max_concurrent_streams: int = 10,
        processing_timeout: float = 30.0,
        config: Optional[StreamingConfig] = None,
    ):
        if config is not None:
            batch_size = config.batch_size
            max_concurrent_streams = config.max_concurrent_streams
            processing_timeout = config.processing_timeout

        self.hybrid_encoder = hybrid_encoder
        self.text_diffusion = text_diffusion
        self.embedding_diffusion = embedding_diffusion

        self.batch_size = batch_size
        self.max_concurrent_streams = max_concurrent_streams
        self.processing_timeout = processing_timeout

        # Async buffers
        self.query_buffer = AsyncStreamBuffer(batch_size=batch_size)
        self.data_buffer = AsyncStreamBuffer(batch_size=batch_size)

        # Processing state
        self.active_streams = 0
        self.processed_queries = 0
        self.processed_data = 0

    async def process_query(
        self, query_id: str, query_text: str, metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Process query asynchronously"""
        start_time = time.time()

        try:
            if self.hybrid_encoder is not None:
                with torch.no_grad():
                    query_emb = self.hybrid_encoder.encode(
                        text=query_text, return_pooled=True
                    )
            else:
                query_emb = None

            result = {
                "query_id": query_id,
                "query_text": query_text,
                "embeddings": (
                    query_emb.cpu().tolist() if query_emb is not None else None
                ),
                "metadata": metadata or {},
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
            }

            self.processed_queries += 1
            return result

        except Exception as e:
            logger.error(f"Error processing query {query_id}: {e}")
            return {
                "query_id": query_id,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    async def process_query_stream(
        self, query_stream: AsyncIterator[str]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Process streaming queries"""
        async for query_text in query_stream:
            query_id = f"query_{int(time.time() * 1000)}"
            result = await self.process_query(query_id, query_text)
            yield result

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "processed_queries": self.processed_queries,
            "processed_data": self.processed_data,
            "active_streams": self.active_streams,
            "query_buffer": self.query_buffer.get_stats(),
            "data_buffer": self.data_buffer.get_stats(),
        }
