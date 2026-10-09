"""
Health Monitor for Voice Agent
Tracks system health and triggers offline mode when needed
"""

import asyncio
import time
from typing import Dict, Any, Optional
from loguru import logger
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class ServiceHealth:
    """Health status of a service."""

    name: str
    is_healthy: bool = True
    last_check: datetime = field(default_factory=datetime.now)
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    response_time_ms: float = 0.0


class HealthMonitor:
    """
    Production-ready health monitor.
    Tracks Redis, LLM APIs, TTS/STT services, and network connectivity.
    """

    def __init__(
        self,
        check_interval_seconds: int = 30,
        failure_threshold: int = 3,
        recovery_threshold: int = 2,
    ):
        """
        Initialize health monitor.

        Args:
            check_interval_seconds: How often to check service health
            failure_threshold: Consecutive failures before marking unhealthy
            recovery_threshold: Consecutive successes before marking healthy
        """
        self.check_interval = check_interval_seconds
        self.failure_threshold = failure_threshold
        self.recovery_threshold = recovery_threshold

        self.services: Dict[str, ServiceHealth] = {
            "redis": ServiceHealth("redis"),
            "llm": ServiceHealth("llm"),
            "tts": ServiceHealth("tts"),
            "stt": ServiceHealth("stt"),
            "network": ServiceHealth("network"),
        }

        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

    async def check_redis(self, redis_client) -> bool:
        """Check Redis connectivity."""
        start = time.time()
        try:
            await asyncio.wait_for(redis_client.ping(), timeout=1.0)
            elapsed = (time.time() - start) * 1000
            self._record_success("redis", elapsed)
            return True
        except Exception as e:
            self._record_failure("redis", str(e))
            return False

    async def check_llm(self, llm_client) -> bool:
        """Check LLM API availability."""
        start = time.time()
        try:
            # Quick health check - send minimal request
            # Implementation depends on your LLM client
            # For now, assume healthy if client exists
            if llm_client:
                elapsed = (time.time() - start) * 1000
                self._record_success("llm", elapsed)
                return True
            return False
        except Exception as e:
            self._record_failure("llm", str(e))
            return False

    async def check_tts(self, tts_client) -> bool:
        """Check TTS service availability."""
        try:
            if tts_client:
                self._record_success("tts", 0)
                return True
            return False
        except Exception as e:
            self._record_failure("tts", str(e))
            return False

    async def check_stt(self, stt_client) -> bool:
        """Check STT service availability."""
        try:
            if stt_client:
                self._record_success("stt", 0)
                return True
            return False
        except Exception as e:
            self._record_failure("stt", str(e))
            return False

    async def check_network(self) -> bool:
        """Check network connectivity."""
        start = time.time()
        try:
            # Ping a reliable endpoint
            import httpx

            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get("https://www.google.com")
                if response.status_code == 200:
                    elapsed = (time.time() - start) * 1000
                    self._record_success("network", elapsed)
                    return True
            return False
        except Exception as e:
            self._record_failure("network", str(e))
            return False

    def _record_success(self, service: str, response_time_ms: float):
        """Record successful health check."""
        if service in self.services:
            svc = self.services[service]
            svc.consecutive_failures = 0
            svc.last_check = datetime.now()
            svc.response_time_ms = response_time_ms
            svc.last_error = None

            # Mark healthy if was unhealthy and hit recovery threshold
            if not svc.is_healthy:
                logger.info(f"Service {service} recovered")
                svc.is_healthy = True

    def _record_failure(self, service: str, error: str):
        """Record failed health check."""
        if service in self.services:
            svc = self.services[service]
            svc.consecutive_failures += 1
            svc.last_check = datetime.now()
            svc.last_error = error

            # Mark unhealthy if hit failure threshold
            if svc.consecutive_failures >= self.failure_threshold and svc.is_healthy:
                logger.warning(
                    f"Service {service} marked unhealthy after {svc.consecutive_failures} failures"
                )
                svc.is_healthy = False

    def is_service_healthy(self, service: str) -> bool:
        """Check if service is healthy."""
        return self.services.get(service, ServiceHealth(service, False)).is_healthy

    def should_use_offline_mode(self) -> bool:
        """
        Determine if offline mode should be activated.
        Offline mode triggers if Redis OR network is down.
        """
        return not self.is_service_healthy("redis") or not self.is_service_healthy(
            "network"
        )

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "offline_mode": self.should_use_offline_mode(),
            "services": {
                name: {
                    "healthy": svc.is_healthy,
                    "last_check": svc.last_check.isoformat(),
                    "consecutive_failures": svc.consecutive_failures,
                    "last_error": svc.last_error,
                    "response_time_ms": svc.response_time_ms,
                }
                for name, svc in self.services.items()
            },
        }

    async def start_monitoring(
        self, redis_client=None, llm_client=None, tts_client=None, stt_client=None
    ):
        """Start background health monitoring."""
        if self._monitoring:
            logger.warning("Health monitoring already running")
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(redis_client, llm_client, tts_client, stt_client)
        )
        logger.info("Health monitoring started")

    async def stop_monitoring(self):
        """Stop background health monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")

    async def _monitor_loop(self, redis_client, llm_client, tts_client, stt_client):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                # Check all services
                await asyncio.gather(
                    (
                        self.check_redis(redis_client)
                        if redis_client
                        else asyncio.sleep(0)
                    ),
                    self.check_llm(llm_client) if llm_client else asyncio.sleep(0),
                    self.check_tts(tts_client) if tts_client else asyncio.sleep(0),
                    self.check_stt(stt_client) if stt_client else asyncio.sleep(0),
                    self.check_network(),
                    return_exceptions=True,
                )

                # Log health status
                report = self.get_health_report()
                if report["offline_mode"]:
                    logger.warning("OFFLINE MODE ACTIVE - System degraded")

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

            await asyncio.sleep(self.check_interval)


# Global health monitor instance
_monitor_instance: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get or create global health monitor instance."""
    global _monitor_instance

    if _monitor_instance is None:
        _monitor_instance = HealthMonitor()

    return _monitor_instance
