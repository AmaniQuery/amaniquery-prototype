"""
Structured logging and monitoring for NiruSense processing pipeline.
"""

import logging
import time
from functools import wraps
from typing import Dict, Any
from pythonjsonlogger import jsonlogger
from .config import settings


# Configure JSON logging
def setup_logging():
    """Setup structured JSON logging"""
    logger = logging.getLogger("nirusense")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    # JSON formatter
    log_handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        "%(timestamp)s %(level)s %(name)s %(message)s",
        rename_fields={"levelname": "level", "asctime": "timestamp"},
    )
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)

    return logger


logger = setup_logging()


class MetricsCollector:
    """Collect and track processing metrics"""

    def __init__(self):
        self.metrics = {
            "documents_processed": 0,
            "documents_failed": 0,
            "total_processing_time": 0.0,
            "agent_metrics": {},
        }

    def record_document_processed(self, processing_time: float):
        """Record successful document processing"""
        self.metrics["documents_processed"] += 1
        self.metrics["total_processing_time"] += processing_time

    def record_document_failed(self):
        """Record failed document processing"""
        self.metrics["documents_failed"] += 1

    def record_agent_execution(
        self, agent_name: str, execution_time: float, success: bool
    ):
        """Record agent execution metrics"""
        if agent_name not in self.metrics["agent_metrics"]:
            self.metrics["agent_metrics"][agent_name] = {
                "executions": 0,
                "failures": 0,
                "total_time": 0.0,
            }

        self.metrics["agent_metrics"][agent_name]["executions"] += 1
        self.metrics["agent_metrics"][agent_name]["total_time"] += execution_time
        if not success:
            self.metrics["agent_metrics"][agent_name]["failures"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        avg_processing_time = (
            self.metrics["total_processing_time"] / self.metrics["documents_processed"]
            if self.metrics["documents_processed"] > 0
            else 0
        )

        return {
            **self.metrics,
            "average_processing_time": avg_processing_time,
            "success_rate": (
                self.metrics["documents_processed"]
                / (
                    self.metrics["documents_processed"]
                    + self.metrics["documents_failed"]
                )
                if (
                    self.metrics["documents_processed"]
                    + self.metrics["documents_failed"]
                )
                > 0
                else 0
            ),
        }


# Global metrics collector
metrics = MetricsCollector()


def track_execution(agent_name: str):
    """Decorator to track agent execution time and success"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                logger.error(f"Agent {agent_name} failed", extra={"error": str(e)})
                raise
            finally:
                execution_time = time.time() - start_time
                metrics.record_agent_execution(agent_name, execution_time, success)

        return wrapper

    return decorator
