"""
Circuit breaker pattern implementation for protecting external services
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, TypeVar, Optional, Dict
from loguru import logger
from threading import Lock

T = TypeVar("T")


class CircuitBreakerState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing, requests are rejected immediately
    HALF_OPEN = "half_open"  # Testing if service recovered, limited requests allowed


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""

    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 2  # Number of successes in half-open to close
    timeout: float = 60.0  # Seconds to wait before transitioning to half-open
    expected_exception: type = Exception  # Exception type to count as failures

    def __post_init__(self):
        """Validate configuration"""
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be at least 1")
        if self.timeout < 0:
            raise ValueError("timeout must be non-negative")


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker"""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0  # Requests rejected when circuit is open
    state_transitions: int = 0
    current_failures: int = 0
    current_successes: int = 0  # For half-open state

    def reset_counters(self):
        """Reset current failure/success counters"""
        self.current_failures = 0
        self.current_successes = 0


class CircuitBreaker:
    """
    Circuit breaker to protect external services from cascading failures

    The circuit breaker has three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests are rejected immediately
    - HALF_OPEN: Testing recovery, limited requests allowed
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker

        Args:
            name: Name identifier for this circuit breaker
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.stats = CircuitBreakerStats()
        self.last_failure_time: Optional[float] = None
        self._lock = Lock()  # Thread-safe state management

        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"threshold={self.config.failure_threshold}, "
            f"timeout={self.config.timeout}s"
        )

    def _transition_to(self, new_state: CircuitBreakerState, reason: str = ""):
        """Transition to a new state (thread-safe)"""
        with self._lock:
            if self.state == new_state:
                return

            old_state = self.state
            self.state = new_state
            self.stats.state_transitions += 1

            logger.info(
                f"Circuit breaker '{self.name}': {old_state.value} -> {new_state.value}"
                + (f" ({reason})" if reason else "")
            )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True

        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.config.timeout

    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute async function with circuit breaker protection

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of function execution

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Original exception from func if it fails
        """
        self.stats.total_requests += 1

        # Check circuit state
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                # Transition to half-open to test recovery
                self._transition_to(CircuitBreakerState.HALF_OPEN, "timeout expired")
                self.stats.reset_counters()
            else:
                # Circuit is open, reject request
                self.stats.rejected_requests += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Service unavailable. Try again later."
                )

        # Execute function
        try:
            result = await func(*args, **kwargs)

            # Success
            self.stats.successful_requests += 1

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.stats.current_successes += 1
                if self.stats.current_successes >= self.config.success_threshold:
                    # Service recovered, close circuit
                    self._transition_to(
                        CircuitBreakerState.CLOSED, "recovery confirmed"
                    )
                    self.stats.reset_counters()
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self.stats.current_failures = 0

            return result

        except self.config.expected_exception as e:
            # Failure
            self.stats.failed_requests += 1
            self.last_failure_time = time.time()

            if self.state == CircuitBreakerState.HALF_OPEN:
                # Still failing, open circuit again
                self._transition_to(CircuitBreakerState.OPEN, "recovery failed")
                self.stats.reset_counters()
            elif self.state == CircuitBreakerState.CLOSED:
                self.stats.current_failures += 1
                if self.stats.current_failures >= self.config.failure_threshold:
                    # Too many failures, open circuit
                    self._transition_to(CircuitBreakerState.OPEN, "threshold exceeded")
                    self.stats.reset_counters()

            # Re-raise original exception
            raise

    def call_sync(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute sync function with circuit breaker protection

        Args:
            func: Sync function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of function execution

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Original exception from func if it fails
        """
        self.stats.total_requests += 1

        # Check circuit state
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                # Transition to half-open to test recovery
                self._transition_to(CircuitBreakerState.HALF_OPEN, "timeout expired")
                self.stats.reset_counters()
            else:
                # Circuit is open, reject request
                self.stats.rejected_requests += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Service unavailable. Try again later."
                )

        # Execute function
        try:
            result = func(*args, **kwargs)

            # Success
            self.stats.successful_requests += 1

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.stats.current_successes += 1
                if self.stats.current_successes >= self.config.success_threshold:
                    # Service recovered, close circuit
                    self._transition_to(
                        CircuitBreakerState.CLOSED, "recovery confirmed"
                    )
                    self.stats.reset_counters()
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self.stats.current_failures = 0

            return result

        except self.config.expected_exception as e:
            # Failure
            self.stats.failed_requests += 1
            self.last_failure_time = time.time()

            if self.state == CircuitBreakerState.HALF_OPEN:
                # Still failing, open circuit again
                self._transition_to(CircuitBreakerState.OPEN, "recovery failed")
                self.stats.reset_counters()
            elif self.state == CircuitBreakerState.CLOSED:
                self.stats.current_failures += 1
                if self.stats.current_failures >= self.config.failure_threshold:
                    # Too many failures, open circuit
                    self._transition_to(CircuitBreakerState.OPEN, "threshold exceeded")
                    self.stats.reset_counters()

            # Re-raise original exception
            raise

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state"""
        return self.state

    def get_stats(self) -> Dict:
        """Get circuit breaker statistics"""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "stats": {
                    "total_requests": self.stats.total_requests,
                    "successful_requests": self.stats.successful_requests,
                    "failed_requests": self.stats.failed_requests,
                    "rejected_requests": self.stats.rejected_requests,
                    "state_transitions": self.stats.state_transitions,
                    "current_failures": self.stats.current_failures,
                    "current_successes": self.stats.current_successes,
                },
                "last_failure_time": self.last_failure_time,
            }

    def reset(self):
        """Manually reset circuit breaker to closed state"""
        self._transition_to(CircuitBreakerState.CLOSED, "manual reset")
        self.stats.reset_counters()
        self.last_failure_time = None
        logger.info(f"Circuit breaker '{self.name}' manually reset")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""

    pass
