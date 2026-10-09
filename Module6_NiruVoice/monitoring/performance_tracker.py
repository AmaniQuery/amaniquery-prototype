"""
Performance tracking for voice agent operations
"""

import time
from contextlib import asynccontextmanager, contextmanager
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from loguru import logger
from threading import Lock


@dataclass
class PerformanceMetrics:
    """Performance metrics for an operation"""

    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def finish(self, success: bool = True, error: Optional[str] = None):
        """Mark operation as finished"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error = error


class PerformanceTracker:
    """
    Tracks performance metrics for voice agent operations
    """

    def __init__(self, max_samples: int = 1000):
        """
        Initialize performance tracker

        Args:
            max_samples: Maximum number of samples to keep per operation type
        """
        self.max_samples = max_samples
        self.metrics: Dict[str, List[PerformanceMetrics]] = {}
        self._lock = Lock()
        logger.info("Performance tracker initialized")

    @contextmanager
    def track(self, operation_name: str, **metadata):
        """
        Context manager for tracking sync operations

        Usage:
            with tracker.track("stt_transcribe", provider="openai"):
                result = stt.transcribe(audio)
        """
        metric = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            metadata=metadata,
        )

        try:
            yield metric
            metric.finish(success=True)
        except Exception as e:
            metric.finish(success=False, error=str(e))
            raise
        finally:
            self._record_metric(metric)

    @asynccontextmanager
    async def track_async(self, operation_name: str, **metadata):
        """
        Context manager for tracking async operations

        Usage:
            async with tracker.track_async("rag_query", query="..."):
                result = await rag.query(text)
        """
        metric = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            metadata=metadata,
        )

        try:
            yield metric
            metric.finish(success=True)
        except Exception as e:
            metric.finish(success=False, error=str(e))
            raise
        finally:
            self._record_metric(metric)

    def _record_metric(self, metric: PerformanceMetrics):
        """Record a performance metric"""
        with self._lock:
            if metric.operation_name not in self.metrics:
                self.metrics[metric.operation_name] = []

            self.metrics[metric.operation_name].append(metric)

            # Keep only last N samples
            if len(self.metrics[metric.operation_name]) > self.max_samples:
                self.metrics[metric.operation_name].pop(0)

    def get_stats(self, operation_name: str) -> Optional[Dict]:
        """
        Get statistics for an operation

        Args:
            operation_name: Name of the operation

        Returns:
            Dictionary with statistics or None if no data
        """
        with self._lock:
            if operation_name not in self.metrics or not self.metrics[operation_name]:
                return None

            samples = self.metrics[operation_name]
            durations = [m.duration for m in samples if m.duration is not None]

            if not durations:
                return None

            successful = [m for m in samples if m.success]
            failed = [m for m in samples if not m.success]

            sorted_durations = sorted(durations)

            return {
                "operation": operation_name,
                "total_count": len(samples),
                "success_count": len(successful),
                "failure_count": len(failed),
                "success_rate": len(successful) / len(samples) if samples else 0.0,
                "duration": {
                    "min": min(durations),
                    "max": max(durations),
                    "avg": sum(durations) / len(durations),
                    "p50": sorted_durations[len(sorted_durations) // 2],
                    "p95": sorted_durations[int(len(sorted_durations) * 0.95)],
                    "p99": sorted_durations[int(len(sorted_durations) * 0.99)],
                },
                "error_types": self._get_error_types(failed),
            }

    def _get_error_types(
        self, failed_samples: List[PerformanceMetrics]
    ) -> Dict[str, int]:
        """Get error type counts"""
        error_types = {}
        for sample in failed_samples:
            if sample.error:
                error_type = sample.error.split(":")[0]  # Get error class name
                error_types[error_type] = error_types.get(error_type, 0) + 1
        return error_types

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all operations"""
        with self._lock:
            return {name: self.get_stats(name) for name in self.metrics.keys()}

    def clear_stats(self, operation_name: Optional[str] = None):
        """
        Clear statistics

        Args:
            operation_name: Clear specific operation (None = all)
        """
        with self._lock:
            if operation_name:
                if operation_name in self.metrics:
                    del self.metrics[operation_name]
                    logger.info(f"Cleared stats for operation: {operation_name}")
            else:
                self.metrics.clear()
                logger.info("Cleared all performance stats")
