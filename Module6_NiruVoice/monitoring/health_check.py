"""
Health check endpoints and status monitoring
"""

from enum import Enum
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
from loguru import logger


class HealthStatus(Enum):
    """Overall health status"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a component"""

    name: str
    status: HealthStatus
    message: str = ""
    last_check: Optional[datetime] = None
    details: Dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "details": self.details,
        }


class HealthChecker:
    """
    Performs health checks on voice agent components
    """

    def __init__(self):
        """Initialize health checker"""
        self.components: Dict[str, ComponentHealth] = {}
        logger.info("Health checker initialized")

    def register_component(
        self, name: str, check_func: callable, required: bool = True
    ):
        """
        Register a component for health checking

        Args:
            name: Component name
            check_func: Function that returns (status, message, details)
            required: Whether component is required for overall health
        """
        self.components[name] = {
            "check_func": check_func,
            "required": required,
            "last_check": None,
        }
        logger.info(f"Registered health check component: {name}")

    def check_component(self, name: str) -> ComponentHealth:
        """
        Check health of a specific component

        Args:
            name: Component name

        Returns:
            ComponentHealth instance
        """
        if name not in self.components:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Component '{name}' not registered",
            )

        component = self.components[name]
        check_func = component["check_func"]

        try:
            result = check_func()

            # Handle different return types
            if isinstance(result, tuple):
                if len(result) >= 2:
                    status_str, message = result[0], result[1]
                    details = result[2] if len(result) > 2 else {}
                else:
                    status_str, message, details = result[0], "", {}
            elif isinstance(result, dict):
                status_str = result.get("status", "unknown")
                message = result.get("message", "")
                details = result.get("details", {})
            else:
                status_str = str(result)
                message = ""
                details = {}

            # Convert string to enum
            try:
                status = HealthStatus(status_str.lower())
            except ValueError:
                # Try to infer from string
                if "healthy" in status_str.lower():
                    status = HealthStatus.HEALTHY
                elif "degraded" in status_str.lower():
                    status = HealthStatus.DEGRADED
                elif "unhealthy" in status_str.lower() or "error" in status_str.lower():
                    status = HealthStatus.UNHEALTHY
                else:
                    status = HealthStatus.UNKNOWN

            component["last_check"] = datetime.utcnow()

            return ComponentHealth(
                name=name,
                status=status,
                message=message,
                last_check=component["last_check"],
                details=details,
            )

        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}", exc_info=True)
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check error: {str(e)}",
                last_check=datetime.utcnow(),
            )

    def check_all(self) -> Dict[str, ComponentHealth]:
        """Check health of all components"""
        results = {}
        for name in self.components:
            results[name] = self.check_component(name)
        return results

    def get_overall_health(self) -> HealthStatus:
        """
        Get overall health status

        Returns:
            HealthStatus based on component health
        """
        if not self.components:
            return HealthStatus.UNKNOWN

        results = self.check_all()
        required_unhealthy = False
        any_unhealthy = False
        any_degraded = False

        for name, health in results.items():
            component = self.components[name]
            if component["required"]:
                if health.status == HealthStatus.UNHEALTHY:
                    required_unhealthy = True
                elif health.status == HealthStatus.DEGRADED:
                    any_degraded = True
            else:
                if health.status == HealthStatus.UNHEALTHY:
                    any_unhealthy = True
                elif health.status == HealthStatus.DEGRADED:
                    any_degraded = True

        if required_unhealthy:
            return HealthStatus.UNHEALTHY
        elif any_degraded or any_unhealthy:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def get_health_report(self) -> Dict:
        """
        Get comprehensive health report

        Returns:
            Dictionary with overall health and component details
        """
        overall = self.get_overall_health()
        components = self.check_all()

        return {
            "status": overall.value,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                name: health.to_dict() for name, health in components.items()
            },
        }
