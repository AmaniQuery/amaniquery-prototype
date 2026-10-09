"""
Memory Manager for Selective Pattern Retention

Manages memory buffer with importance scoring and eviction policies
for retaining important patterns while discarding less relevant ones.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from loguru import logger

from ..config import RetentionConfig, default_config


@dataclass
class MemoryPattern:
    """Represents a memory pattern"""

    id: str
    embeddings: torch.Tensor  # Pattern embeddings
    metadata: Dict[str, Any]  # Additional metadata
    importance_score: float = 0.0
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "embeddings": (
                self.embeddings.cpu().tolist()
                if isinstance(self.embeddings, torch.Tensor)
                else self.embeddings
            ),
            "metadata": self.metadata,
            "importance_score": self.importance_score,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryPattern":
        """Create from dictionary"""
        pattern = cls(
            id=data["id"],
            embeddings=torch.tensor(data["embeddings"]),
            metadata=data["metadata"],
            importance_score=data.get("importance_score", 0.0),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(
                data.get("last_accessed", datetime.now().isoformat())
            ),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.now().isoformat())
            ),
        )
        return pattern


class MemoryManager:
    """Manages memory buffer with selective retention"""

    def __init__(
        self,
        buffer_size: int = 10000,
        importance_threshold: float = 0.7,
        eviction_policy: str = "lru",  # lru, lfu, importance
        retention_ratio: float = 0.1,
        config: Optional[RetentionConfig] = None,
    ):
        if config is not None:
            buffer_size = config.memory_buffer_size
            importance_threshold = config.importance_threshold
            eviction_policy = config.eviction_policy
            retention_ratio = config.retention_ratio

        self.buffer_size = buffer_size
        self.importance_threshold = importance_threshold
        self.eviction_policy = eviction_policy
        self.retention_ratio = retention_ratio

        # Memory buffer: OrderedDict for LRU, regular dict for others
        if eviction_policy == "lru":
            self.memory_buffer = OrderedDict()
        else:
            self.memory_buffer = {}

        # Statistics
        self.total_patterns_added = 0
        self.total_patterns_evicted = 0
        self.total_accesses = 0

    def compute_importance_score(
        self,
        embeddings: torch.Tensor,
        metadata: Optional[Dict] = None,
        query_similarity: Optional[float] = None,
    ) -> float:
        """
        Compute importance score for a pattern

        Args:
            embeddings: Pattern embeddings
            metadata: Optional metadata
            query_similarity: Similarity to recent queries

        Returns:
            importance_score: Score between 0 and 1
        """
        score = 0.0

        # Base score from embedding norm (more informative patterns)
        if isinstance(embeddings, torch.Tensor):
            norm = torch.norm(embeddings, p="fro").item()
            score += min(norm / 10.0, 1.0) * 0.3  # Normalize and weight

        # Query similarity (if provided)
        if query_similarity is not None:
            score += query_similarity * 0.4

        # Metadata-based scoring
        if metadata is not None:
            # Higher score for frequently accessed patterns
            access_count = metadata.get("access_count", 0)
            score += min(access_count / 100.0, 1.0) * 0.2

            # Higher score for recent patterns
            if "created_at" in metadata:
                age_days = (
                    datetime.now() - datetime.fromisoformat(metadata["created_at"])
                ).days
                recency_score = max(0, 1.0 - age_days / 30.0)  # Decay over 30 days
                score += recency_score * 0.1

        return min(score, 1.0)

    def add_pattern(
        self,
        pattern_id: str,
        embeddings: torch.Tensor,
        metadata: Optional[Dict] = None,
        query_similarity: Optional[float] = None,
    ) -> bool:
        """
        Add pattern to memory buffer

        Args:
            pattern_id: Unique pattern identifier
            embeddings: Pattern embeddings
            metadata: Optional metadata
            query_similarity: Similarity to recent queries

        Returns:
            success: Whether pattern was added
        """
        # Compute importance score
        importance_score = self.compute_importance_score(
            embeddings, metadata, query_similarity
        )

        # Create pattern
        pattern = MemoryPattern(
            id=pattern_id,
            embeddings=embeddings,
            metadata=metadata or {},
            importance_score=importance_score,
        )

        # Check if buffer is full
        if len(self.memory_buffer) >= self.buffer_size:
            # Evict least important pattern
            self._evict_pattern()

        # Add to buffer
        if self.eviction_policy == "lru":
            self.memory_buffer[pattern_id] = pattern
            # Move to end (most recently used)
            self.memory_buffer.move_to_end(pattern_id)
        else:
            self.memory_buffer[pattern_id] = pattern

        self.total_patterns_added += 1
        return True

    def get_pattern(self, pattern_id: str) -> Optional[MemoryPattern]:
        """Retrieve pattern from memory"""
        if pattern_id not in self.memory_buffer:
            return None

        pattern = self.memory_buffer[pattern_id]

        # Update access statistics
        pattern.access_count += 1
        pattern.last_accessed = datetime.now()
        pattern.metadata["access_count"] = pattern.access_count

        # Update importance based on access
        pattern.importance_score = self.compute_importance_score(
            pattern.embeddings, pattern.metadata
        )

        # Move to end for LRU
        if self.eviction_policy == "lru" and isinstance(
            self.memory_buffer, OrderedDict
        ):
            self.memory_buffer.move_to_end(pattern_id)

        self.total_accesses += 1
        return pattern

    def _evict_pattern(self):
        """Evict least important pattern based on policy"""
        if len(self.memory_buffer) == 0:
            return

        if self.eviction_policy == "lru":
            # Remove least recently used (first item)
            if isinstance(self.memory_buffer, OrderedDict):
                pattern_id = next(iter(self.memory_buffer))
                del self.memory_buffer[pattern_id]
                self.total_patterns_evicted += 1

        elif self.eviction_policy == "lfu":
            # Remove least frequently used
            min_access_count = min(p.access_count for p in self.memory_buffer.values())
            for pattern_id, pattern in self.memory_buffer.items():
                if pattern.access_count == min_access_count:
                    del self.memory_buffer[pattern_id]
                    self.total_patterns_evicted += 1
                    break

        elif self.eviction_policy == "importance":
            # Remove least important
            min_importance = min(
                p.importance_score for p in self.memory_buffer.values()
            )
            for pattern_id, pattern in self.memory_buffer.items():
                if pattern.importance_score == min_importance:
                    del self.memory_buffer[pattern_id]
                    self.total_patterns_evicted += 1
                    break

    def get_top_patterns(
        self, top_k: Optional[int] = None, min_importance: Optional[float] = None
    ) -> List[MemoryPattern]:
        """
        Get top patterns by importance

        Args:
            top_k: Number of top patterns to return
            min_importance: Minimum importance threshold

        Returns:
            patterns: List of top patterns
        """
        patterns = list(self.memory_buffer.values())

        # Filter by minimum importance
        if min_importance is not None:
            patterns = [p for p in patterns if p.importance_score >= min_importance]

        # Sort by importance
        patterns.sort(key=lambda p: p.importance_score, reverse=True)

        # Return top k
        if top_k is not None:
            patterns = patterns[:top_k]

        return patterns

    def retain_important_patterns(self) -> List[str]:
        """
        Retain only important patterns (top retention_ratio)

        Returns:
            retained_ids: List of retained pattern IDs
        """
        num_retain = int(len(self.memory_buffer) * self.retention_ratio)
        top_patterns = self.get_top_patterns(top_k=num_retain)

        retained_ids = [p.id for p in top_patterns]

        # Remove patterns not in retained list
        patterns_to_remove = [
            pid for pid in self.memory_buffer.keys() if pid not in retained_ids
        ]

        for pid in patterns_to_remove:
            del self.memory_buffer[pid]
            self.total_patterns_evicted += 1

        logger.info(
            f"Retained {len(retained_ids)} patterns "
            f"({self.retention_ratio * 100:.1f}% of buffer)"
        )

        return retained_ids

    def search_similar(
        self,
        query_embeddings: torch.Tensor,
        top_k: int = 10,
        threshold: Optional[float] = None,
    ) -> List[Tuple[MemoryPattern, float]]:
        """
        Search for similar patterns

        Args:
            query_embeddings: Query embeddings
            top_k: Number of results
            threshold: Minimum similarity threshold

        Returns:
            results: List of (pattern, similarity) tuples
        """
        results = []

        for pattern in self.memory_buffer.values():
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                query_embeddings.unsqueeze(0), pattern.embeddings.unsqueeze(0)
            ).item()

            if threshold is None or similarity >= threshold:
                results.append((pattern, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        return results[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "buffer_size": len(self.memory_buffer),
            "max_buffer_size": self.buffer_size,
            "total_patterns_added": self.total_patterns_added,
            "total_patterns_evicted": self.total_patterns_evicted,
            "total_accesses": self.total_accesses,
            "eviction_policy": self.eviction_policy,
            "avg_importance": (
                np.mean([p.importance_score for p in self.memory_buffer.values()])
                if self.memory_buffer
                else 0.0
            ),
            "avg_access_count": (
                np.mean([p.access_count for p in self.memory_buffer.values()])
                if self.memory_buffer
                else 0.0
            ),
        }

    def save(self, filepath: Path):
        """Save memory buffer to file"""
        data = {
            "patterns": [p.to_dict() for p in self.memory_buffer.values()],
            "stats": self.get_stats(),
            "config": {
                "buffer_size": self.buffer_size,
                "importance_threshold": self.importance_threshold,
                "eviction_policy": self.eviction_policy,
                "retention_ratio": self.retention_ratio,
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved memory buffer to {filepath}")

    def load(self, filepath: Path):
        """Load memory buffer from file"""
        with open(filepath, "r") as f:
            data = json.load(f)

        # Clear existing buffer
        self.memory_buffer.clear()

        # Load patterns
        for pattern_dict in data["patterns"]:
            pattern = MemoryPattern.from_dict(pattern_dict)
            if self.eviction_policy == "lru":
                self.memory_buffer[pattern.id] = pattern
            else:
                self.memory_buffer[pattern.id] = pattern

        # Update statistics
        self.total_patterns_added = data["stats"].get("total_patterns_added", 0)
        self.total_patterns_evicted = data["stats"].get("total_patterns_evicted", 0)
        self.total_accesses = data["stats"].get("total_accesses", 0)

        logger.info(f"Loaded {len(self.memory_buffer)} patterns from {filepath}")
