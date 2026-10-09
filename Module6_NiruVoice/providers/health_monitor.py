"""
Provider health monitoring for STT/TTS providers
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from loguru import logger
from threading import Lock


class ProviderStatus(Enum):
    """Provider health status"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Working but slow/high error rate
    UNHEALTHY = "unhealthy"  # Failing frequently
    UNKNOWN = "unknown"  # No data yet


@dataclass
class ProviderHealth:
    """Health status for a provider"""

    provider_name: str
    status: ProviderStatus = ProviderStatus.UNKNOWN
    last_check: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None

    # Statistics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Performance metrics
    average_latency: float = 0.0  # seconds
    latency_samples: List[float] = field(default_factory=list)
    max_latency_samples: int = 100  # Keep last N samples

    # Error tracking
    recent_errors: List[tuple] = field(default_factory=list)  # (timestamp, error_type)
    max_error_history: int = 50

    # Thresholds
    failure_rate_threshold: float = 0.5  # 50% failure rate = unhealthy
    degraded_threshold: float = 0.2  # 20% failure rate = degraded
    latency_threshold: float = 5.0  # 5 seconds = degraded
    max_latency_threshold: float = 10.0  # 10 seconds = unhealthy

    def update_success(self, latency: float):
        """Update health after successful request"""
        self.total_requests += 1
        self.successful_requests += 1
        self.last_success = datetime.utcnow()
        self.last_check = datetime.utcnow()

        # Update latency
        self.latency_samples.append(latency)
        if len(self.latency_samples) > self.max_latency_samples:
            self.latency_samples.pop(0)

        self.average_latency = sum(self.latency_samples) / len(self.latency_samples)

        self._update_status()

    def update_failure(self, error_type: str = "unknown"):
        """Update health after failed request"""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_failure = datetime.utcnow()
        self.last_check = datetime.utcnow()

        # Track error
        self.recent_errors.append((datetime.utcnow(), error_type))
        if len(self.recent_errors) > self.max_error_history:
            self.recent_errors.pop(0)

        self._update_status()

    def _update_status(self):
        """Update provider status based on metrics"""
        if self.total_requests == 0:
            self.status = ProviderStatus.UNKNOWN
            return

        failure_rate = self.failed_requests / self.total_requests

        # Check recent failure rate (last 10 requests)
        recent_requests = min(10, self.total_requests)
        recent_failures = sum(1 for _, _ in self.recent_errors[-recent_requests:])
        recent_failure_rate = (
            recent_failures / recent_requests if recent_requests > 0 else 0
        )

        # Use recent failure rate if we have enough data
        effective_failure_rate = (
            recent_failure_rate if recent_requests >= 5 else failure_rate
        )

        # Determine status
        if effective_failure_rate >= self.failure_rate_threshold:
            self.status = ProviderStatus.UNHEALTHY
        elif (
            effective_failure_rate >= self.degraded_threshold
            or self.average_latency > self.latency_threshold
        ):
            self.status = ProviderStatus.DEGRADED
        elif self.average_latency > self.max_latency_threshold:
            self.status = ProviderStatus.UNHEALTHY
        else:
            self.status = ProviderStatus.HEALTHY

    def get_failure_rate(self) -> float:
        """Get current failure rate"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    def is_available(self) -> bool:
        """Check if provider is available for use"""
        return self.status in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "provider_name": self.provider_name,
            "status": self.status.value,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_success": (
                self.last_success.isoformat() if self.last_success else None
            ),
            "last_failure": (
                self.last_failure.isoformat() if self.last_failure else None
            ),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "failure_rate": self.get_failure_rate(),
            "average_latency": self.average_latency,
            "is_available": self.is_available(),
        }


class HealthMonitor:
    """
    Monitors health of multiple providers
    """

    def __init__(self):
        """Initialize health monitor"""
        self.providers: Dict[str, ProviderHealth] = {}
        self._lock = Lock()
        logger.info("Health monitor initialized")

    def register_provider(self, provider_name: str) -> ProviderHealth:
        """
        Register a provider for health monitoring

        Args:
            provider_name: Name of the provider

        Returns:
            ProviderHealth instance
        """
        with self._lock:
            if provider_name not in self.providers:
                self.providers[provider_name] = ProviderHealth(
                    provider_name=provider_name
                )
                logger.info(
                    f"Registered provider for health monitoring: {provider_name}"
                )
            return self.providers[provider_name]

    def get_provider_health(self, provider_name: str) -> Optional[ProviderHealth]:
        """Get health status for a provider"""
        return self.providers.get(provider_name)

    def record_success(self, provider_name: str, latency: float):
        """Record successful request"""
        with self._lock:
            if provider_name not in self.providers:
                self.register_provider(provider_name)
            self.providers[provider_name].update_success(latency)

    def record_failure(self, provider_name: str, error_type: str = "unknown"):
        """Record failed request"""
        with self._lock:
            if provider_name not in self.providers:
                self.register_provider(provider_name)
            self.providers[provider_name].update_failure(error_type)

    def get_healthy_providers(self) -> List[str]:
        """Get list of healthy providers"""
        with self._lock:
            return [
                name for name, health in self.providers.items() if health.is_available()
            ]

    def get_all_health_status(self) -> Dict[str, Dict]:
        """Get health status for all providers"""
        with self._lock:
            return {name: health.to_dict() for name, health in self.providers.items()}

    def reset_provider_stats(self, provider_name: str):
        """Reset statistics for a provider"""
        with self._lock:
            if provider_name in self.providers:
                health = self.providers[provider_name]
                health.total_requests = 0
                health.successful_requests = 0
                health.failed_requests = 0
                health.latency_samples.clear()
                health.recent_errors.clear()
                health.average_latency = 0.0
                health.status = ProviderStatus.UNKNOWN
                logger.info(f"Reset stats for provider: {provider_name}")
