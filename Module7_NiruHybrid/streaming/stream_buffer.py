"""
Stream Buffer for Real-time Data Processing

Manages buffers for streaming queries and generated data with timeout
and batch processing capabilities.
"""

import asyncio
import time
from typing import List, Dict, Optional, Any, Callable
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import threading
from loguru import logger

from ..config import StreamingConfig, default_config


@dataclass
class StreamItem:
    """Represents an item in the stream buffer"""

    id: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False


class StreamBuffer:
    """Buffer for streaming data with timeout and batch processing"""

    def __init__(
        self,
        buffer_size: int = 1000,
        timeout: float = 0.1,
        batch_size: int = 32,
        config: Optional[StreamingConfig] = None,
    ):
        if config is not None:
            buffer_size = config.buffer_size
            timeout = config.buffer_timeout
            batch_size = config.batch_size

        self.buffer_size = buffer_size
        self.timeout = timeout
        self.batch_size = batch_size

        # Buffer storage
        self.buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()

        # Statistics
        self.total_items = 0
        self.total_processed = 0
        self.total_dropped = 0
        self.last_process_time = None

    def add(self, item_id: str, data: Any, metadata: Optional[Dict] = None) -> bool:
        """
        Add item to buffer

        Args:
            item_id: Unique item identifier
            data: Item data
            metadata: Optional metadata

        Returns:
            success: Whether item was added
        """
        with self.lock:
            # Check if buffer is full
            if len(self.buffer) >= self.buffer_size:
                # Drop oldest item
                self.buffer.popleft()
                self.total_dropped += 1
                logger.warning(f"Buffer full, dropped item")

            # Create stream item
            item = StreamItem(id=item_id, data=data, metadata=metadata or {})

            # Add to buffer
            self.buffer.append(item)
            self.total_items += 1

            return True

    def get_batch(
        self, max_items: Optional[int] = None, timeout: Optional[float] = None
    ) -> List[StreamItem]:
        """
        Get batch of items from buffer

        Args:
            max_items: Maximum number of items (default: batch_size)
            timeout: Timeout in seconds (default: self.timeout)

        Returns:
            items: List of stream items
        """
        timeout = timeout or self.timeout
        max_items = max_items or self.batch_size

        with self.lock:
            # Get unprocessed items
            unprocessed = [item for item in self.buffer if not item.processed]

            # Take up to max_items
            batch = unprocessed[:max_items]

            # Mark as processed
            for item in batch:
                item.processed = True

            self.total_processed += len(batch)
            self.last_process_time = time.time()

            return batch

    def get_all_unprocessed(self) -> List[StreamItem]:
        """Get all unprocessed items"""
        with self.lock:
            unprocessed = [item for item in self.buffer if not item.processed]
            return unprocessed

    def clear_processed(self):
        """Remove processed items from buffer"""
        with self.lock:
            self.buffer = deque(
                [item for item in self.buffer if not item.processed],
                maxlen=self.buffer_size,
            )

    def clear(self):
        """Clear all items from buffer"""
        with self.lock:
            self.buffer.clear()
            logger.info("Buffer cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            return {
                "buffer_size": len(self.buffer),
                "max_buffer_size": self.buffer_size,
                "unprocessed": sum(1 for item in self.buffer if not item.processed),
                "processed": sum(1 for item in self.buffer if item.processed),
                "total_items": self.total_items,
                "total_processed": self.total_processed,
                "total_dropped": self.total_dropped,
                "last_process_time": self.last_process_time,
            }


class AsyncStreamBuffer:
    """Async version of stream buffer for async processing"""

    def __init__(
        self,
        buffer_size: int = 1000,
        timeout: float = 0.1,
        batch_size: int = 32,
        config: Optional[StreamingConfig] = None,
    ):
        if config is not None:
            buffer_size = config.buffer_size
            timeout = config.buffer_timeout
            batch_size = config.batch_size

        self.buffer_size = buffer_size
        self.timeout = timeout
        self.batch_size = batch_size

        # Async buffer
        self.buffer = asyncio.Queue(maxsize=buffer_size)
        self.lock = asyncio.Lock()

        # Statistics
        self.total_items = 0
        self.total_processed = 0
        self.total_dropped = 0

    async def add(
        self, item_id: str, data: Any, metadata: Optional[Dict] = None
    ) -> bool:
        """Add item to async buffer"""
        async with self.lock:
            try:
                item = StreamItem(id=item_id, data=data, metadata=metadata or {})

                # Try to add without blocking
                try:
                    self.buffer.put_nowait(item)
                    self.total_items += 1
                    return True
                except asyncio.QueueFull:
                    # Drop oldest item
                    try:
                        self.buffer.get_nowait()
                        self.total_dropped += 1
                        self.buffer.put_nowait(item)
                        self.total_items += 1
                        return True
                    except asyncio.QueueEmpty:
                        return False
            except Exception as e:
                logger.error(f"Error adding item to async buffer: {e}")
                return False

    async def get_batch(
        self, max_items: Optional[int] = None, timeout: Optional[float] = None
    ) -> List[StreamItem]:
        """Get batch from async buffer"""
        timeout = timeout or self.timeout
        max_items = max_items or self.batch_size

        items = []
        end_time = time.time() + timeout

        while len(items) < max_items and time.time() < end_time:
            try:
                # Wait for item with timeout
                remaining_time = end_time - time.time()
                if remaining_time <= 0:
                    break

                item = await asyncio.wait_for(
                    self.buffer.get(), timeout=min(remaining_time, 0.1)
                )
                items.append(item)
                self.total_processed += 1
            except asyncio.TimeoutError:
                break
            except Exception as e:
                logger.error(f"Error getting item from async buffer: {e}")
                break

        return items

    async def get_all(self) -> List[StreamItem]:
        """Get all items from buffer"""
        items = []
        while not self.buffer.empty():
            try:
                item = self.buffer.get_nowait()
                items.append(item)
            except asyncio.QueueEmpty:
                break
        return items

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            "buffer_size": self.buffer.qsize(),
            "max_buffer_size": self.buffer_size,
            "total_items": self.total_items,
            "total_processed": self.total_processed,
            "total_dropped": self.total_dropped,
        }
