"""
Provider manager for managing multiple STT/TTS providers with failover
"""

import time
from typing import Dict, List, Optional, Callable, TypeVar, Any
from dataclasses import dataclass
from loguru import logger

from Module6_NiruVoice.providers.health_monitor import HealthMonitor, ProviderHealth
from Module6_NiruVoice.providers.fallback_strategy import (
    FallbackStrategy,
    FallbackConfig,
    FallbackMode,
)
from Module6_NiruVoice.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
)

T = TypeVar("T")


@dataclass
class ProviderConfig:
    """Configuration for a provider"""

    name: str
    priority: int = 0  # Lower = higher priority
    enabled: bool = True
    timeout: float = 30.0  # Request timeout in seconds
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None

    def __post_init__(self):
        """Validate configuration"""
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")


class ProviderManager:
    """
    Manages multiple providers with health monitoring and failover
    """

    def __init__(
        self,
        provider_configs: List[ProviderConfig],
        fallback_config: Optional[FallbackConfig] = None,
        enable_health_monitoring: bool = True,
    ):
        """
        Initialize provider manager

        Args:
            provider_configs: List of provider configurations
            fallback_config: Fallback strategy configuration
            enable_health_monitoring: Enable health monitoring
        """
        if not provider_configs:
            raise ValueError("At least one provider configuration required")

        # Sort by priority
        self.provider_configs = sorted(provider_configs, key=lambda x: x.priority)
        self.providers = [cfg.name for cfg in self.provider_configs if cfg.enabled]

        if not self.providers:
            raise ValueError("At least one enabled provider required")

        # Initialize health monitor
        self.health_monitor = HealthMonitor() if enable_health_monitoring else None

        # Initialize fallback strategy
        self.fallback_strategy = FallbackStrategy(
            providers=self.providers,
            config=fallback_config or FallbackConfig(),
            health_monitor=self.health_monitor,
        )

        # Initialize circuit breakers for each provider
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        for config in self.provider_configs:
            if config.enabled:
                cb_config = config.circuit_breaker_config or CircuitBreakerConfig()
                self.circuit_breakers[config.name] = CircuitBreaker(
                    name=f"{config.name}_cb",
                    config=cb_config,
                )

        # Register providers with health monitor
        if self.health_monitor:
            for provider in self.providers:
                self.health_monitor.register_provider(provider)

        logger.info(
            f"Provider manager initialized with {len(self.providers)} providers: "
            f"{', '.join(self.providers)}"
        )

    async def execute_with_fallback_async(
        self, provider_funcs: Dict[str, Callable], *args, **kwargs
    ) -> T:
        """
        Execute operation with automatic failover

        Args:
            provider_funcs: Dictionary mapping provider names to async functions
            *args: Positional arguments for provider functions
            **kwargs: Keyword arguments for provider functions

        Returns:
            Result from first successful provider

        Raises:
            Exception: If all providers fail
        """
        selected_providers = self.fallback_strategy.select_providers()
        last_exception = None

        for i, provider_name in enumerate(selected_providers):
            if provider_name not in provider_funcs:
                logger.warning(f"Provider function not found: {provider_name}")
                continue

            provider_func = provider_funcs[provider_name]
            was_fallback = i > 0

            try:
                # Get circuit breaker for this provider
                circuit_breaker = self.circuit_breakers.get(provider_name)

                # Execute with circuit breaker protection
                start_time = time.time()

                if circuit_breaker:
                    result = await circuit_breaker.call_async(
                        provider_func, *args, **kwargs
                    )
                else:
                    result = await provider_func(*args, **kwargs)

                latency = time.time() - start_time

                # Record success
                if self.health_monitor:
                    self.health_monitor.record_success(provider_name, latency)

                self.fallback_strategy.record_success(provider_name, was_fallback)

                if was_fallback:
                    logger.info(f"Fallback to {provider_name} succeeded")

                return result

            except Exception as e:
                last_exception = e
                error_type = type(e).__name__

                # Record failure
                if self.health_monitor:
                    self.health_monitor.record_failure(provider_name, error_type)

                logger.warning(
                    f"Provider {provider_name} failed: {e}. "
                    f"{'Trying next provider...' if i < len(selected_providers) - 1 else 'All providers exhausted'}"
                )

                # Continue to next provider
                continue

        # All providers failed
        self.fallback_strategy.record_failure()
        logger.error("All providers failed")

        if last_exception:
            raise last_exception
        raise RuntimeError("All providers failed and no exception was captured")

    def execute_with_fallback_sync(
        self, provider_funcs: Dict[str, Callable], *args, **kwargs
    ) -> T:
        """
        Execute sync operation with automatic failover

        Args:
            provider_funcs: Dictionary mapping provider names to sync functions
            *args: Positional arguments for provider functions
            **kwargs: Keyword arguments for provider functions

        Returns:
            Result from first successful provider

        Raises:
            Exception: If all providers fail
        """
        selected_providers = self.fallback_strategy.select_providers()
        last_exception = None

        for i, provider_name in enumerate(selected_providers):
            if provider_name not in provider_funcs:
                logger.warning(f"Provider function not found: {provider_name}")
                continue

            provider_func = provider_funcs[provider_name]
            was_fallback = i > 0

            try:
                # Get circuit breaker for this provider
                circuit_breaker = self.circuit_breakers.get(provider_name)

                # Execute with circuit breaker protection
                start_time = time.time()

                if circuit_breaker:
                    result = circuit_breaker.call_sync(provider_func, *args, **kwargs)
                else:
                    result = provider_func(*args, **kwargs)

                latency = time.time() - start_time

                # Record success
                if self.health_monitor:
                    self.health_monitor.record_success(provider_name, latency)

                self.fallback_strategy.record_success(provider_name, was_fallback)

                if was_fallback:
                    logger.info(f"Fallback to {provider_name} succeeded")

                return result

            except Exception as e:
                last_exception = e
                error_type = type(e).__name__

                # Record failure
                if self.health_monitor:
                    self.health_monitor.record_failure(provider_name, error_type)

                logger.warning(
                    f"Provider {provider_name} failed: {e}. "
                    f"{'Trying next provider...' if i < len(selected_providers) - 1 else 'All providers exhausted'}"
                )

                # Continue to next provider
                continue

        # All providers failed
        self.fallback_strategy.record_failure()
        logger.error("All providers failed")

        if last_exception:
            raise last_exception
        raise RuntimeError("All providers failed and no exception was captured")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for all providers"""
        if self.health_monitor:
            return self.health_monitor.get_all_health_status()
        return {}

    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        if self.health_monitor:
            return self.health_monitor.get_healthy_providers()
        return self.providers

    def get_stats(self) -> Dict:
        """Get statistics"""
        return {
            "providers": self.providers,
            "fallback_stats": self.fallback_strategy.get_stats(),
            "health_status": self.get_health_status(),
        }
