"""
Policy Engine
Policy-based access control with conditions
"""

from datetime import datetime
from typing import Dict, Optional, Any, Callable
from sqlalchemy.orm import Session

from ..models.auth_models import User, Integration


class PolicyEngine:
    """Policy-based access control"""

    # Policy condition functions
    CONDITIONS: Dict[str, Callable] = {}

    @staticmethod
    def register_condition(name: str, condition_func: Callable):
        """Register a custom condition function"""
        PolicyEngine.CONDITIONS[name] = condition_func

    @staticmethod
    def check_time_based_policy(conditions: Dict[str, Any]) -> bool:
        """Check time-based policy conditions"""
        if "time_range" in conditions:
            time_range = conditions["time_range"]
            current_hour = datetime.utcnow().hour

            if "start_hour" in time_range and current_hour < time_range["start_hour"]:
                return False
            if "end_hour" in time_range and current_hour >= time_range["end_hour"]:
                return False

        return True

    @staticmethod
    def check_ip_based_policy(
        ip_address: Optional[str], conditions: Dict[str, Any]
    ) -> bool:
        """Check IP-based policy conditions"""
        if "ip_whitelist" in conditions:
            whitelist = conditions["ip_whitelist"]
            if ip_address and ip_address not in whitelist:
                return False

        if "ip_blacklist" in conditions:
            blacklist = conditions["ip_blacklist"]
            if ip_address and ip_address in blacklist:
                return False

        return True

    @staticmethod
    def check_rate_limit_policy(
        db: Session,
        user: Optional[User] = None,
        integration: Optional[Integration] = None,
        conditions: Dict[str, Any] = None,
    ) -> bool:
        """Check rate limit policy conditions"""
        # This would integrate with rate limiting system
        # For now, just return True
        return True

    @staticmethod
    def evaluate_policy(
        db: Session,
        user: Optional[User] = None,
        integration: Optional[Integration] = None,
        conditions: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
    ) -> bool:
        """Evaluate policy with all conditions"""
        if not conditions:
            return True

        # Check time-based conditions
        if not PolicyEngine.check_time_based_policy(conditions):
            return False

        # Check IP-based conditions
        if not PolicyEngine.check_ip_based_policy(ip_address, conditions):
            return False

        # Check rate limit conditions
        if not PolicyEngine.check_rate_limit_policy(db, user, integration, conditions):
            return False

        # Check custom conditions
        for condition_name, condition_value in conditions.items():
            if condition_name in PolicyEngine.CONDITIONS:
                condition_func = PolicyEngine.CONDITIONS[condition_name]
                if not condition_func(user, integration, condition_value):
                    return False

        return True
