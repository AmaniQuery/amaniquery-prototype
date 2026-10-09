"""
Media Processing Resilience - Circuit breaker and retry patterns for media operations

Extends patterns from Module6_NiruVoice/resilience for media processing with:
- Circuit breaker for external services (Cohere, Gemini, Whisper)
- Retry with exponential backoff
- Graceful fallbacks (text-only RAG when vision fails)
- Health monitoring
"""

import asyncio
import time
from enum import Enum
from typing import Callable, TypeVar, Optional, Dict, Any, List
from dataclasses import dataclass, field
from loguru import logger
from threading import Lock

T = TypeVar("T")


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class MediaCircuitBreakerConfig:
    """Configuration for media processing circuit breaker"""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes in half-open to close
    timeout_seconds: float = 60.0  # Time before half-open
    half_open_max_calls: int = 3  # Max calls in half-open

    def __post_init__(self):
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be at least 1")


class MediaCircuitBreaker:
    """
    Circuit breaker for media processing operations

    Prevents cascading failures when external services (Cohere, Gemini, etc.) are down.
    """

    def __init__(
        self,
        name: str,
        config: Optional[MediaCircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or MediaCircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self._lock = Lock()

        # Counters
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0

        # Timing
        self.last_failure_time: Optional[float] = None
        self.last_state_change: float = time.time()

        # Stats
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "rejected_calls": 0,
            "state_transitions": 0,
        }

        logger.info(
            f"Circuit breaker '{name}' initialized (threshold={config.failure_threshold if config else 5})"
        )

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if timeout has elapsed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.timeout_seconds

    def _transition_to(self, new_state: CircuitState, reason: str = ""):
        """Transition to new state"""
        with self._lock:
            if self.state == new_state:
                return

            old_state = self.state
            self.state = new_state
            self.last_state_change = time.time()
            self.stats["state_transitions"] += 1

            # Reset counters on transition
            if new_state == CircuitState.HALF_OPEN:
                self.half_open_calls = 0
                self.success_count = 0
            elif new_state == CircuitState.CLOSED:
                self.failure_count = 0

            logger.info(
                f"Circuit '{self.name}': {old_state.value} -> {new_state.value} ({reason})"
            )

    def record_success(self):
        """Record successful call"""
        with self._lock:
            self.stats["successful_calls"] += 1

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED, "recovery confirmed")
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

    def record_failure(self, error: Optional[Exception] = None):
        """Record failed call"""
        with self._lock:
            self.stats["failed_calls"] += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN, "failed during recovery test")
            elif self.state == CircuitState.CLOSED:
                self.failure_count += 1
                if self.failure_count >= self.config.failure_threshold:
                    self._transition_to(
                        CircuitState.OPEN,
                        f"threshold exceeded ({self.failure_count} failures)",
                    )

    def can_execute(self) -> bool:
        """Check if request can be executed"""
        self.stats["total_calls"] += 1

        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN, "timeout elapsed")
                self.half_open_calls = 1
                return True
            else:
                self.stats["rejected_calls"] += 1
                return False

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls < self.config.half_open_max_calls:
                self.half_open_calls += 1
                return True
            else:
                self.stats["rejected_calls"] += 1
                return False

        return False

    async def execute_async(
        self,
        func: Callable[..., T],
        *args,
        fallback: Optional[Callable[..., T]] = None,
        **kwargs,
    ) -> T:
        """
        Execute async function with circuit breaker protection

        Args:
            func: Async function to execute
            fallback: Optional fallback function if circuit is open
            *args, **kwargs: Arguments for func

        Returns:
            Result of func or fallback

        Raises:
            CircuitOpenError if circuit is open and no fallback
        """
        if not self.can_execute():
            if fallback:
                logger.warning(f"Circuit '{self.name}' open, using fallback")
                return (
                    await fallback(*args, **kwargs)
                    if asyncio.iscoroutinefunction(fallback)
                    else fallback(*args, **kwargs)
                )
            raise CircuitOpenError(f"Circuit '{self.name}' is open")

        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e)
            raise

    def execute(
        self,
        func: Callable[..., T],
        *args,
        fallback: Optional[Callable[..., T]] = None,
        **kwargs,
    ) -> T:
        """Execute sync function with circuit breaker protection"""
        if not self.can_execute():
            if fallback:
                logger.warning(f"Circuit '{self.name}' open, using fallback")
                return fallback(*args, **kwargs)
            raise CircuitOpenError(f"Circuit '{self.name}' is open")

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e)
            raise

    def get_stats(self) -> Dict:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "time_in_state": time.time() - self.last_state_change,
            **self.stats,
        }

    def reset(self):
        """Manually reset circuit breaker"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.half_open_calls = 0
            logger.info(f"Circuit '{self.name}' manually reset")


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open"""

    pass


# =============================================================================
# RETRY HANDLER
# =============================================================================


@dataclass
class MediaRetryConfig:
    """Configuration for media processing retries"""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (
        ConnectionError,
        TimeoutError,
        OSError,
    )


class MediaRetryHandler:
    """
    Retry handler with exponential backoff for media operations
    """

    def __init__(self, config: Optional[MediaRetryConfig] = None):
        self.config = config or MediaRetryConfig()
        self.stats = {
            "total_attempts": 0,
            "successful_retries": 0,
            "failed_operations": 0,
        }

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next retry"""
        delay = self.config.base_delay * (self.config.exponential_base**attempt)
        delay = min(delay, self.config.max_delay)

        if self.config.jitter:
            import random

            delay *= 0.5 + random.random()

        return delay

    def _should_retry(self, exception: Exception) -> bool:
        """Check if exception is retryable"""
        return isinstance(exception, self.config.retryable_exceptions)

    async def execute_async(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute async function with retry logic"""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            self.stats["total_attempts"] += 1

            try:
                result = await func(*args, **kwargs)
                if attempt > 0:
                    self.stats["successful_retries"] += 1
                    logger.info(f"Operation succeeded after {attempt} retries")
                return result

            except Exception as e:
                last_exception = e

                if not self._should_retry(e) or attempt >= self.config.max_retries:
                    self.stats["failed_operations"] += 1
                    raise

                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Retry {attempt + 1}/{self.config.max_retries} after {delay:.1f}s: {e}"
                )
                await asyncio.sleep(delay)

        self.stats["failed_operations"] += 1
        raise last_exception

    def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute sync function with retry logic"""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            self.stats["total_attempts"] += 1

            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self.stats["successful_retries"] += 1
                return result

            except Exception as e:
                last_exception = e

                if not self._should_retry(e) or attempt >= self.config.max_retries:
                    self.stats["failed_operations"] += 1
                    raise

                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Retry {attempt + 1}/{self.config.max_retries} after {delay:.1f}s: {e}"
                )
                time.sleep(delay)

        self.stats["failed_operations"] += 1
        raise last_exception


# =============================================================================
# FALLBACK STRATEGIES
# =============================================================================


class FallbackMode(Enum):
    """Fallback strategy modes"""

    SEQUENTIAL = "sequential"  # Try providers in order
    HEALTH_BASED = "health_based"  # Prefer healthy providers
    ROUND_ROBIN = "round_robin"  # Rotate between providers


@dataclass
class FallbackConfig:
    """Configuration for fallback strategies"""

    mode: FallbackMode = FallbackMode.SEQUENTIAL
    max_fallbacks: int = 3


class MediaFallbackManager:
    """
    Manages fallback strategies for media processing

    When primary processing fails, falls back to:
    1. Alternative embedding model
    2. Text-only RAG (no vision)
    3. Cached results
    """

    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
        self.fallback_stats = {
            "fallbacks_triggered": 0,
            "fallbacks_succeeded": 0,
            "fallbacks_exhausted": 0,
        }

    async def execute_with_fallback_async(
        self,
        primary: Callable[..., T],
        fallbacks: List[Callable[..., T]],
        *args,
        **kwargs,
    ) -> T:
        """
        Execute with fallback chain

        Args:
            primary: Primary function to execute
            fallbacks: List of fallback functions
            *args, **kwargs: Arguments for functions

        Returns:
            Result from primary or first successful fallback
        """
        # Try primary
        try:
            return (
                await primary(*args, **kwargs)
                if asyncio.iscoroutinefunction(primary)
                else primary(*args, **kwargs)
            )
        except Exception as primary_error:
            logger.warning(f"Primary operation failed: {primary_error}")
            self.fallback_stats["fallbacks_triggered"] += 1

        # Try fallbacks
        for i, fallback in enumerate(fallbacks[: self.config.max_fallbacks]):
            try:
                logger.info(f"Trying fallback {i + 1}/{len(fallbacks)}")
                result = (
                    await fallback(*args, **kwargs)
                    if asyncio.iscoroutinefunction(fallback)
                    else fallback(*args, **kwargs)
                )
                self.fallback_stats["fallbacks_succeeded"] += 1
                return result
            except Exception as fallback_error:
                logger.warning(f"Fallback {i + 1} failed: {fallback_error}")
                continue

        self.fallback_stats["fallbacks_exhausted"] += 1
        raise RuntimeError("All fallbacks exhausted")


# =============================================================================
# RESILIENT MEDIA PROCESSOR
# =============================================================================


class ResilientMediaProcessor:
    """
    Wraps media processing with full resilience support

    Combines circuit breaker, retry, and fallback for robust media handling.
    """

    def __init__(
        self,
        name: str = "media_processor",
        circuit_config: Optional[MediaCircuitBreakerConfig] = None,
        retry_config: Optional[MediaRetryConfig] = None,
        fallback_config: Optional[FallbackConfig] = None,
    ):
        self.name = name
        self.circuit_breaker = MediaCircuitBreaker(name, circuit_config)
        self.retry_handler = MediaRetryHandler(retry_config)
        self.fallback_manager = MediaFallbackManager(fallback_config)

        logger.info(f"Resilient media processor '{name}' initialized")

    async def process_async(
        self,
        operation: Callable[..., T],
        *args,
        fallback: Optional[Callable[..., T]] = None,
        skip_retry: bool = False,
        **kwargs,
    ) -> T:
        """
        Process with full resilience chain

        Order: Circuit Breaker -> Retry -> Fallback
        """

        async def operation_with_retry():
            if skip_retry:
                return (
                    await operation(*args, **kwargs)
                    if asyncio.iscoroutinefunction(operation)
                    else operation(*args, **kwargs)
                )
            return await self.retry_handler.execute_async(operation, *args, **kwargs)

        return await self.circuit_breaker.execute_async(
            operation_with_retry,
            fallback=fallback,
        )

    def get_health_status(self) -> Dict:
        """Get health status of all components"""
        return {
            "name": self.name,
            "circuit_breaker": self.circuit_breaker.get_stats(),
            "retry_handler": self.retry_handler.stats,
            "fallback_manager": self.fallback_manager.fallback_stats,
        }

    def reset(self):
        """Reset all components"""
        self.circuit_breaker.reset()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

# Global instances for different service types
_processors: Dict[str, ResilientMediaProcessor] = {}


def get_resilient_processor(service_name: str) -> ResilientMediaProcessor:
    """Get or create a resilient processor for a service"""
    if service_name not in _processors:
        _processors[service_name] = ResilientMediaProcessor(name=service_name)
    return _processors[service_name]


def get_vision_processor() -> ResilientMediaProcessor:
    """Get processor for vision/embedding operations"""
    return get_resilient_processor("vision_embedding")


def get_transcription_processor() -> ResilientMediaProcessor:
    """Get processor for audio transcription"""
    return get_resilient_processor("audio_transcription")


def get_video_processor() -> ResilientMediaProcessor:
    """Get processor for video processing"""
    return get_resilient_processor("video_processing")


def get_all_processor_health() -> Dict[str, Dict]:
    """Get health status of all processors"""
    return {name: proc.get_health_status() for name, proc in _processors.items()}
