"""
Fallback strategy for provider failover
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Callable, TypeVar, Any
from loguru import logger

T = TypeVar("T")


class FallbackMode(Enum):
    """Fallback modes"""

    SEQUENTIAL = "sequential"  # Try providers in order until one succeeds
    ROUND_ROBIN = "round_robin"  # Distribute requests across providers
    HEALTH_BASED = "health_based"  # Prefer healthier providers
    RANDOM = "random"  # Random selection from available providers


@dataclass
class FallbackConfig:
    """Configuration for fallback strategy"""

    mode: FallbackMode = FallbackMode.HEALTH_BASED
    enable_fallback: bool = True
    max_attempts: int = 3  # Max providers to try
    prefer_primary: bool = True  # Always try primary first if healthy

    def __post_init__(self):
        """Validate configuration"""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")


class FallbackStrategy:
    """
    Manages fallback between multiple providers
    """

    def __init__(
        self,
        providers: List[str],
        config: Optional[FallbackConfig] = None,
        health_monitor: Optional[Any] = None,  # HealthMonitor type
    ):
        """
        Initialize fallback strategy

        Args:
            providers: List of provider names in priority order
            config: Fallback configuration
            health_monitor: HealthMonitor instance for health-based selection
        """
        if not providers:
            raise ValueError("At least one provider must be specified")

        self.providers = providers
        self.config = config or FallbackConfig()
        self.health_monitor = health_monitor
        self._round_robin_index = 0
        self._stats = {
            "total_requests": 0,
            "primary_successes": 0,
            "fallback_successes": 0,
            "total_failures": 0,
        }

        logger.info(
            f"Fallback strategy initialized: mode={self.config.mode.value}, "
            f"providers={len(providers)}"
        )

    def select_providers(self) -> List[str]:
        """
        Select providers to try based on fallback mode

        Returns:
            List of provider names in order to try
        """
        self._stats["total_requests"] += 1

        if not self.config.enable_fallback or len(self.providers) == 1:
            return self.providers[:1]

        if self.config.mode == FallbackMode.SEQUENTIAL:
            return self._select_sequential()
        elif self.config.mode == FallbackMode.ROUND_ROBIN:
            return self._select_round_robin()
        elif self.config.mode == FallbackMode.HEALTH_BASED:
            return self._select_health_based()
        elif self.config.mode == FallbackMode.RANDOM:
            return self._select_random()
        else:
            # Default to sequential
            return self._select_sequential()

    def _select_sequential(self) -> List[str]:
        """Select providers in priority order"""
        if self.config.prefer_primary and self.health_monitor:
            # Check if primary is healthy
            primary = self.providers[0]
            health = self.health_monitor.get_provider_health(primary)
            if health and health.is_available():
                return [primary]

        # Return all providers up to max_attempts
        return self.providers[: self.config.max_attempts]

    def _select_round_robin(self) -> List[str]:
        """Select providers using round-robin"""
        available = self._get_available_providers()
        if not available:
            # Fallback to all providers if none are marked available
            available = self.providers

        # Start from current index
        selected = []
        start_idx = self._round_robin_index % len(available)

        for i in range(min(self.config.max_attempts, len(available))):
            idx = (start_idx + i) % len(available)
            selected.append(available[idx])

        # Update index for next time
        self._round_robin_index = (start_idx + 1) % len(available)

        return selected

    def _select_health_based(self) -> List[str]:
        """Select providers based on health status"""
        if not self.health_monitor:
            # No health monitor, fall back to sequential
            return self._select_sequential()

        # Get all providers sorted by health
        provider_health = []
        for provider in self.providers:
            health = self.health_monitor.get_provider_health(provider)
            if health:
                provider_health.append((provider, health))
            else:
                # Unknown health, add with low priority
                provider_health.append((provider, None))

        # Sort by health status (healthy > degraded > unhealthy > unknown)
        def health_priority(item):
            provider, health = item
            if health is None:
                return 3  # Unknown
            status = health.status.value
            if status == "healthy":
                return 0
            elif status == "degraded":
                return 1
            else:
                return 2

        provider_health.sort(key=health_priority)

        # Return top providers
        selected = [
            provider for provider, _ in provider_health[: self.config.max_attempts]
        ]

        # Always prefer primary if it's healthy
        if self.config.prefer_primary:
            primary = self.providers[0]
            if primary not in selected and self.health_monitor:
                health = self.health_monitor.get_provider_health(primary)
                if health and health.is_available():
                    selected.insert(0, primary)
                    selected = selected[: self.config.max_attempts]

        return selected

    def _select_random(self) -> List[str]:
        """Select providers randomly"""
        import random

        available = self._get_available_providers()
        if not available:
            available = self.providers

        # Randomly select up to max_attempts
        selected = random.sample(
            available, min(self.config.max_attempts, len(available))
        )

        # Always try primary first if prefer_primary
        if self.config.prefer_primary and self.providers[0] in available:
            if self.providers[0] not in selected:
                selected.insert(0, self.providers[0])
                selected = selected[: self.config.max_attempts]

        return selected

    def _get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        if not self.health_monitor:
            return self.providers

        return self.health_monitor.get_healthy_providers()

    def record_success(self, provider_name: str, was_fallback: bool = False):
        """Record successful request"""
        if was_fallback:
            self._stats["fallback_successes"] += 1
        else:
            self._stats["primary_successes"] += 1

    def record_failure(self):
        """Record failed request"""
        self._stats["total_failures"] += 1

    def get_stats(self) -> dict:
        """Get fallback statistics"""
        return self._stats.copy()

    def reset_stats(self):
        """Reset statistics"""
        self._stats = {
            "total_requests": 0,
            "primary_successes": 0,
            "fallback_successes": 0,
            "total_failures": 0,
        }
