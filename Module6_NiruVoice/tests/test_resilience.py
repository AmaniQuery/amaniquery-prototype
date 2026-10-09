"""
Tests for resilience module
"""

import pytest
import asyncio
from Module6_NiruVoice.resilience.retry_handler import RetryHandler, RetryConfig
from Module6_NiruVoice.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
)


class TestRetryHandler:
    """Tests for RetryHandler"""

    def test_retry_config_validation(self):
        """Test retry configuration validation"""
        # Valid config
        config = RetryConfig(max_attempts=3, initial_wait=1.0)
        assert config.max_attempts == 3

        # Invalid config
        with pytest.raises(ValueError):
            RetryConfig(max_attempts=0)

    async def test_retry_success(self):
        """Test successful retry"""
        handler = RetryHandler(RetryConfig(max_attempts=3))

        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await handler.execute_async(success_func)
        assert result == "success"
        assert call_count == 1

    async def test_retry_failure(self):
        """Test retry on failure"""
        handler = RetryHandler(RetryConfig(max_attempts=3, initial_wait=0.1))

        call_count = 0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Connection failed")

        with pytest.raises(ConnectionError):
            await handler.execute_async(failing_func)

        assert call_count == 3  # Should retry 3 times


class TestCircuitBreaker:
    """Tests for CircuitBreaker"""

    def test_circuit_breaker_initial_state(self):
        """Test initial circuit breaker state"""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        assert cb.get_state() == CircuitBreakerState.CLOSED

    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures"""
        config = CircuitBreakerConfig(failure_threshold=3, timeout=1.0)
        cb = CircuitBreaker("test", config)

        # Cause failures
        for _ in range(3):
            try:
                cb.call_sync(lambda: (_ for _ in ()).throw(ConnectionError("Fail")))
            except ConnectionError:
                pass

        # Circuit should be open
        assert cb.get_state() == CircuitBreakerState.OPEN

    def test_circuit_breaker_rejects_when_open(self):
        """Test circuit breaker rejects requests when open"""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=10.0)
        cb = CircuitBreaker("test", config)

        # Cause failure to open circuit
        try:
            cb.call_sync(lambda: (_ for _ in ()).throw(ConnectionError("Fail")))
        except ConnectionError:
            pass

        # Should reject new requests
        from Module6_NiruVoice.resilience.circuit_breaker import CircuitBreakerOpenError

        with pytest.raises(CircuitBreakerOpenError):
            cb.call_sync(lambda: "success")
