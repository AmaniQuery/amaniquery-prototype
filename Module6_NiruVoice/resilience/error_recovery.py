"""
Error recovery strategies for graceful degradation
"""

from enum import Enum
from typing import Callable, TypeVar, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger

T = TypeVar("T")


class RecoveryStrategy(Enum):
    """Recovery strategies"""

    RETRY = "retry"  # Retry the operation
    FALLBACK = "fallback"  # Use fallback function
    CACHE = "cache"  # Return cached result
    DEFAULT = "default"  # Return default value
    FAIL = "fail"  # Fail immediately


@dataclass
class RecoveryConfig:
    """Configuration for error recovery"""

    strategy: RecoveryStrategy = RecoveryStrategy.FALLBACK
    fallback_func: Optional[Callable] = None
    default_value: Any = None
    cache_key: Optional[str] = None
    log_error: bool = True

    def __post_init__(self):
        """Validate configuration"""
        if self.strategy == RecoveryStrategy.FALLBACK and self.fallback_func is None:
            raise ValueError("fallback_func required for FALLBACK strategy")
        if self.strategy == RecoveryStrategy.DEFAULT and self.default_value is None:
            raise ValueError("default_value required for DEFAULT strategy")
        if self.strategy == RecoveryStrategy.CACHE and self.cache_key is None:
            raise ValueError("cache_key required for CACHE strategy")


class ErrorRecovery:
    """
    Handles error recovery with various strategies
    """

    def __init__(self, config: Optional[RecoveryConfig] = None):
        """
        Initialize error recovery

        Args:
            config: Recovery configuration
        """
        self.config = config or RecoveryConfig()
        self._cache: Dict[str, Any] = {}
        self._stats = {
            "total_recoveries": 0,
            "retry_recoveries": 0,
            "fallback_recoveries": 0,
            "cache_recoveries": 0,
            "default_recoveries": 0,
        }

    async def recover_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute async function with error recovery

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of function or recovery strategy result
        """
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if self.config.log_error:
                logger.error(f"Error in operation: {e}", exc_info=True)

            return await self._apply_recovery_strategy(e, func, *args, **kwargs)

    def recover_sync(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute sync function with error recovery

        Args:
            func: Sync function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of function or recovery strategy result
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if self.config.log_error:
                logger.error(f"Error in operation: {e}", exc_info=True)

            return self._apply_recovery_strategy_sync(e, func, *args, **kwargs)

    async def _apply_recovery_strategy(
        self, error: Exception, func: Callable, *args, **kwargs
    ) -> T:
        """Apply recovery strategy for async function"""
        self._stats["total_recoveries"] += 1

        if self.config.strategy == RecoveryStrategy.RETRY:
            self._stats["retry_recoveries"] += 1
            logger.info("Retrying operation after error")
            # Simple retry - in practice, use RetryHandler
            try:
                return await func(*args, **kwargs)
            except Exception:
                # If retry fails, continue to next strategy or fail
                pass

        if self.config.strategy == RecoveryStrategy.FALLBACK:
            self._stats["fallback_recoveries"] += 1
            logger.info("Using fallback function")
            if asyncio.iscoroutinefunction(self.config.fallback_func):
                return await self.config.fallback_func(*args, **kwargs)
            else:
                return self.config.fallback_func(*args, **kwargs)

        if self.config.strategy == RecoveryStrategy.CACHE:
            self._stats["cache_recoveries"] += 1
            logger.info(f"Returning cached result for key: {self.config.cache_key}")
            if self.config.cache_key in self._cache:
                return self._cache[self.config.cache_key]
            else:
                logger.warning(f"Cache miss for key: {self.config.cache_key}")
                raise error  # No cache available, re-raise

        if self.config.strategy == RecoveryStrategy.DEFAULT:
            self._stats["default_recoveries"] += 1
            logger.info("Returning default value")
            return self.config.default_value

        # RecoveryStrategy.FAIL
        logger.error("Recovery strategy is FAIL, re-raising error")
        raise error

    def _apply_recovery_strategy_sync(
        self, error: Exception, func: Callable, *args, **kwargs
    ) -> T:
        """Apply recovery strategy for sync function"""
        self._stats["total_recoveries"] += 1

        if self.config.strategy == RecoveryStrategy.RETRY:
            self._stats["retry_recoveries"] += 1
            logger.info("Retrying operation after error")
            try:
                return func(*args, **kwargs)
            except Exception:
                pass

        if self.config.strategy == RecoveryStrategy.FALLBACK:
            self._stats["fallback_recoveries"] += 1
            logger.info("Using fallback function")
            return self.config.fallback_func(*args, **kwargs)

        if self.config.strategy == RecoveryStrategy.CACHE:
            self._stats["cache_recoveries"] += 1
            logger.info(f"Returning cached result for key: {self.config.cache_key}")
            if self.config.cache_key in self._cache:
                return self._cache[self.config.cache_key]
            else:
                logger.warning(f"Cache miss for key: {self.config.cache_key}")
                raise error

        if self.config.strategy == RecoveryStrategy.DEFAULT:
            self._stats["default_recoveries"] += 1
            logger.info("Returning default value")
            return self.config.default_value

        # RecoveryStrategy.FAIL
        logger.error("Recovery strategy is FAIL, re-raising error")
        raise error

    def set_cache(self, key: str, value: Any):
        """Set cache value"""
        self._cache[key] = value

    def get_cache(self, key: str) -> Optional[Any]:
        """Get cache value"""
        return self._cache.get(key)

    def clear_cache(self):
        """Clear all cache"""
        self._cache.clear()

    def get_stats(self) -> Dict:
        """Get recovery statistics"""
        return self._stats.copy()

    def reset_stats(self):
        """Reset statistics"""
        self._stats = {
            "total_recoveries": 0,
            "retry_recoveries": 0,
            "fallback_recoveries": 0,
            "cache_recoveries": 0,
            "default_recoveries": 0,
        }


# Import asyncio for async check
import asyncio
