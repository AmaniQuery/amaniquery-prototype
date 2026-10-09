"""
Retry handler with exponential backoff for resilient operations
"""

import asyncio
import time
from typing import Callable, TypeVar, Optional, List, Type
from dataclasses import dataclass
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
    before_sleep_log,
    after_log,
)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""

    max_attempts: int = 3
    initial_wait: float = 1.0  # seconds
    max_wait: float = 60.0  # seconds
    exponential_base: float = 2.0
    retryable_exceptions: tuple = (
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
        OSError,
    )
    jitter: bool = True  # Add random jitter to prevent thundering herd

    def __post_init__(self):
        """Validate configuration"""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_wait < 0:
            raise ValueError("initial_wait must be non-negative")
        if self.max_wait < self.initial_wait:
            raise ValueError("max_wait must be >= initial_wait")


class RetryHandler:
    """
    Handles retries with exponential backoff for async and sync operations
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry handler

        Args:
            config: Retry configuration (uses defaults if None)
        """
        self.config = config or RetryConfig()
        self._stats = {
            "total_attempts": 0,
            "successful_retries": 0,
            "failed_retries": 0,
        }

    async def execute_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute async function with retry logic

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func execution

        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        attempt = 0

        while attempt < self.config.max_attempts:
            try:
                self._stats["total_attempts"] += 1
                result = await func(*args, **kwargs)

                if attempt > 0:
                    self._stats["successful_retries"] += 1
                    logger.info(f"Operation succeeded after {attempt} retries")

                return result

            except self.config.retryable_exceptions as e:
                last_exception = e
                attempt += 1

                if attempt >= self.config.max_attempts:
                    self._stats["failed_retries"] += 1
                    logger.error(
                        f"Operation failed after {attempt} attempts: {e}", exc_info=True
                    )
                    raise

                # Calculate wait time with exponential backoff
                wait_time = min(
                    self.config.initial_wait
                    * (self.config.exponential_base ** (attempt - 1)),
                    self.config.max_wait,
                )

                # Add jitter if enabled
                if self.config.jitter:
                    import random

                    jitter_amount = wait_time * 0.1  # 10% jitter
                    wait_time += random.uniform(-jitter_amount, jitter_amount)
                    wait_time = max(0, wait_time)  # Ensure non-negative

                logger.warning(
                    f"Operation failed (attempt {attempt}/{self.config.max_attempts}): {e}. "
                    f"Retrying in {wait_time:.2f}s..."
                )

                await asyncio.sleep(wait_time)

            except Exception as e:
                # Non-retryable exception - raise immediately
                logger.error(f"Non-retryable exception: {e}", exc_info=True)
                raise

        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry handler reached unexpected state")

    def execute_sync(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute sync function with retry logic

        Args:
            func: Sync function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func execution

        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        attempt = 0

        while attempt < self.config.max_attempts:
            try:
                self._stats["total_attempts"] += 1
                result = func(*args, **kwargs)

                if attempt > 0:
                    self._stats["successful_retries"] += 1
                    logger.info(f"Operation succeeded after {attempt} retries")

                return result

            except self.config.retryable_exceptions as e:
                last_exception = e
                attempt += 1

                if attempt >= self.config.max_attempts:
                    self._stats["failed_retries"] += 1
                    logger.error(
                        f"Operation failed after {attempt} attempts: {e}", exc_info=True
                    )
                    raise

                # Calculate wait time with exponential backoff
                wait_time = min(
                    self.config.initial_wait
                    * (self.config.exponential_base ** (attempt - 1)),
                    self.config.max_wait,
                )

                # Add jitter if enabled
                if self.config.jitter:
                    import random

                    jitter_amount = wait_time * 0.1  # 10% jitter
                    wait_time += random.uniform(-jitter_amount, jitter_amount)
                    wait_time = max(0, wait_time)  # Ensure non-negative

                logger.warning(
                    f"Operation failed (attempt {attempt}/{self.config.max_attempts}): {e}. "
                    f"Retrying in {wait_time:.2f}s..."
                )

                time.sleep(wait_time)

            except Exception as e:
                # Non-retryable exception - raise immediately
                logger.error(f"Non-retryable exception: {e}", exc_info=True)
                raise

        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry handler reached unexpected state")

    def get_stats(self) -> dict:
        """Get retry statistics"""
        return self._stats.copy()

    def reset_stats(self):
        """Reset retry statistics"""
        self._stats = {
            "total_attempts": 0,
            "successful_retries": 0,
            "failed_retries": 0,
        }


def retry_with_config(config: RetryConfig):
    """
    Decorator for retrying functions with custom configuration

    Usage:
        @retry_with_config(RetryConfig(max_attempts=5))
        async def my_function():
            ...
    """

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                handler = RetryHandler(config)
                return await handler.execute_async(func, *args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                handler = RetryHandler(config)
                return handler.execute_sync(func, *args, **kwargs)

            return sync_wrapper

    return decorator
