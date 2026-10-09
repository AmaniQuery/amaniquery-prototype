"""
Authentication and Authorization Configuration
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class AuthConfig:
    """Authentication configuration"""

    # JWT Configuration
    JWT_SECRET_KEY: str = os.getenv(
        "JWT_SECRET_KEY", "change-me-in-production-use-strong-random-key"
    )
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "15")
    )
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = int(
        os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "30")
    )

    # Session Configuration
    SESSION_TOKEN_LENGTH: int = 64
    SESSION_EXPIRE_HOURS: int = int(os.getenv("SESSION_EXPIRE_HOURS", "24"))
    SESSION_REFRESH_THRESHOLD_HOURS: int = int(
        os.getenv("SESSION_REFRESH_THRESHOLD_HOURS", "12")
    )

    # Password Configuration
    PASSWORD_HASH_ALGORITHM: str = os.getenv(
        "PASSWORD_HASH_ALGORITHM", "bcrypt"
    )  # bcrypt or argon2
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_DIGIT: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = False

    # API Key Configuration
    API_KEY_PREFIX_LIVE: str = "aq_live_"
    API_KEY_PREFIX_TEST: str = "aq_test_"
    API_KEY_LENGTH: int = 32
    API_KEY_HASH_ALGORITHM: str = "bcrypt"

    # OAuth Configuration
    OAUTH_CLIENT_ID_LENGTH: int = 32
    OAUTH_CLIENT_SECRET_LENGTH: int = 64
    OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES: int = 10
    OAUTH_DEFAULT_SCOPES: List[str] = ["read", "write"]

    # Rate Limiting Defaults
    RATE_LIMIT_DEFAULT_PER_MINUTE: int = int(
        os.getenv("RATE_LIMIT_DEFAULT_PER_MINUTE", "60")
    )
    RATE_LIMIT_DEFAULT_PER_HOUR: int = int(
        os.getenv("RATE_LIMIT_DEFAULT_PER_HOUR", "1000")
    )
    RATE_LIMIT_DEFAULT_PER_DAY: int = int(
        os.getenv("RATE_LIMIT_DEFAULT_PER_DAY", "10000")
    )

    # Rate Limits by Tier
    RATE_LIMITS: Dict[str, Dict[str, int]] = {
        "free": {
            "per_minute": 10,
            "per_hour": 100,
            "per_day": 1000,
        },
        "basic": {
            "per_minute": 60,
            "per_hour": 1000,
            "per_day": 10000,
        },
        "premium": {
            "per_minute": 300,
            "per_hour": 10000,
            "per_day": 100000,
        },
        "enterprise": {
            "per_minute": 1000,
            "per_hour": 100000,
            "per_day": 1000000,
        },
    }

    # Endpoint-specific rate limits
    ENDPOINT_RATE_LIMITS: Dict[str, Dict[str, int]] = {
        "/query": {
            "per_minute": 30,
            "per_hour": 500,
            "per_day": 5000,
        },
        "/research/analyze-legal-query": {
            "per_minute": 5,
            "per_hour": 50,
            "per_day": 500,
        },
        "/alignment-check": {
            "per_minute": 10,
            "per_hour": 100,
            "per_day": 1000,
        },
    }

    # Permission Definitions
    PERMISSIONS: Dict[str, Dict[str, List[str]]] = {
        "query": {
            "read": ["query:read", "query:execute"],
            "write": ["query:read", "query:write"],
        },
        "research": {
            "read": ["research:read"],
            "write": ["research:read", "research:write"],
        },
        "admin": {
            "manage": ["admin:*", "user:manage", "integration:manage"],
        },
        "user": {
            "read": ["user:read"],
            "write": ["user:read", "user:write"],
            "manage": ["user:*"],
        },
        "integration": {
            "read": ["integration:read"],
            "write": ["integration:read", "integration:write"],
            "manage": ["integration:*"],
        },
    }

    # Default Roles
    DEFAULT_ROLES: Dict[str, Dict] = {
        "user": {
            "name": "user",
            "description": "Standard user with basic access",
            "role_type": "user",
            "permissions": ["query:read", "query:execute", "chat:read", "chat:write"],
            "is_system": True,
        },
        "admin": {
            "name": "admin",
            "description": "Administrator with full access",
            "role_type": "user",
            "permissions": ["*"],  # All permissions
            "is_system": True,
        },
        "integration_read_only": {
            "name": "integration_read_only",
            "description": "Read-only access for integrations",
            "role_type": "integration",
            "permissions": ["query:read", "query:execute"],
            "is_system": True,
        },
        "integration_read_write": {
            "name": "integration_read_write",
            "description": "Read and write access for integrations",
            "role_type": "integration",
            "permissions": [
                "query:read",
                "query:execute",
                "query:write",
                "research:read",
                "research:write",
            ],
            "is_system": True,
        },
    }

    # Security Settings
    MAX_LOGIN_ATTEMPTS: int = 5
    LOCKOUT_DURATION_MINUTES: int = 30
    EMAIL_VERIFICATION_REQUIRED: bool = (
        os.getenv("EMAIL_VERIFICATION_REQUIRED", "false").lower() == "true"
    )
    PASSWORD_RESET_TOKEN_EXPIRE_HOURS: int = 24
    EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS: int = 48

    # Social Auth (optional)
    GOOGLE_CLIENT_ID: Optional[str] = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET: Optional[str] = os.getenv("GOOGLE_CLIENT_SECRET")
    GITHUB_CLIENT_ID: Optional[str] = os.getenv("GITHUB_CLIENT_ID")
    GITHUB_CLIENT_SECRET: Optional[str] = os.getenv("GITHUB_CLIENT_SECRET")

    # Cost Tracking
    LLM_COST_PER_1K_TOKENS: Dict[str, float] = {
        "moonshot-v1-8k": 0.001,
        "gpt-4": 0.03,
        "gpt-3.5-turbo": 0.002,
        "claude-3-opus": 0.015,
        "gemini-2.5-flash": 0.0005,
    }

    # Webhook Configuration
    WEBHOOK_SECRET: Optional[str] = os.getenv("WEBHOOK_SECRET")
    WEBHOOK_TIMEOUT_SECONDS: int = 5

    # Database
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")

    @classmethod
    def get_rate_limit(
        cls, tier: str, endpoint: Optional[str] = None
    ) -> Dict[str, int]:
        """Get rate limit for tier and optionally endpoint"""
        limits = cls.RATE_LIMITS.get(tier, cls.RATE_LIMITS["basic"]).copy()

        if endpoint and endpoint in cls.ENDPOINT_RATE_LIMITS:
            endpoint_limits = cls.ENDPOINT_RATE_LIMITS[endpoint]
            # Use the more restrictive limit
            limits["per_minute"] = min(
                limits["per_minute"],
                endpoint_limits.get("per_minute", limits["per_minute"]),
            )
            limits["per_hour"] = min(
                limits["per_hour"], endpoint_limits.get("per_hour", limits["per_hour"])
            )
            limits["per_day"] = min(
                limits["per_day"], endpoint_limits.get("per_day", limits["per_day"])
            )

        return limits

    @classmethod
    def get_llm_cost(cls, model: str, tokens: int) -> float:
        """Calculate cost for LLM usage"""
        cost_per_1k = cls.LLM_COST_PER_1K_TOKENS.get(model, 0.001)
        return (tokens / 1000) * cost_per_1k


# Global config instance
config = AuthConfig()
