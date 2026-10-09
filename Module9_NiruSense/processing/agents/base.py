from abc import ABC, abstractmethod
import time
from typing import Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from ..config import settings
from ..monitoring import logger

"""Base agent class for all NiruSense processing agents"""


class BaseAgent(ABC):
    """
    Base class for all NiruSense agents.
    Provides common functionality for timing, error handling, and retry logic.
    """

    def __init__(self, agent_id: str, model_version: str = "v1"):
        self.agent_id = agent_id
        self.model_version = model_version
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.error_count = 0

        logger.info(
            f"Initialized agent: {agent_id}",
            extra={"agent_id": agent_id, "model_version": model_version},
        )

    @abstractmethod
    def process(self, text: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process text and return results.
        Must be implemented by subclasses.
        """
        pass

    def execute(self, text: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute agent with timing and error tracking.
        """
        start_time = time.time()
        self.execution_count += 1

        try:
            result = self._execute_with_retry(text, metadata)
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time

            # Add execution metadata
            result["_meta"] = {
                "agent_id": self.agent_id,
                "model_version": self.model_version,
                "execution_time_ms": int(execution_time * 1000),
            }

            logger.debug(
                f"Agent {self.agent_id} completed",
                extra={
                    "agent_id": self.agent_id,
                    "execution_time_ms": int(execution_time * 1000),
                },
            )

            return result

        except Exception as e:
            self.error_count += 1
            logger.error(
                f"Agent {self.agent_id} failed",
                extra={
                    "agent_id": self.agent_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

            # Return error result
            return {
                "error": str(e),
                "agent_id": self.agent_id,
                "_meta": {
                    "agent_id": self.agent_id,
                    "model_version": self.model_version,
                    "execution_time_ms": int((time.time() - start_time) * 1000),
                    "status": "error",
                },
            }

    @retry(
        stop=stop_after_attempt(settings.AGENT_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _execute_with_retry(
        self, text: str, metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute with automatic retry on failure"""
        return self.process(text, metadata)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent execution statistics"""
        avg_time = (
            self.total_execution_time / self.execution_count
            if self.execution_count > 0
            else 0
        )

        return {
            "agent_id": self.agent_id,
            "executions": self.execution_count,
            "errors": self.error_count,
            "average_execution_time": avg_time,
            "total_execution_time": self.total_execution_time,
            "success_rate": (
                (self.execution_count - self.error_count) / self.execution_count
                if self.execution_count > 0
                else 0
            ),
        }
