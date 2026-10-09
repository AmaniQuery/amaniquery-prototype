"""
Rate Limiter - Token bucket rate limiting per chat/session
"""

from typing import Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from loguru import logger


class AgentRateLimiter:
    """
    Token bucket rate limiter for agent requests
    """

    def __init__(self, tokens_per_minute: int = 1000, tokens_per_hour: int = 10000):
        """
        Initialize rate limiter

        Args:
            tokens_per_minute: Tokens allowed per minute
            tokens_per_hour: Tokens allowed per hour
        """
        self.tokens_per_minute = tokens_per_minute
        self.tokens_per_hour = tokens_per_hour

        # Track usage per session
        self.session_usage: Dict[str, Dict[str, Any]] = {}

    def check_rate_limit(
        self, session_id: str, tokens: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if request is within rate limits

        Args:
            session_id: Session ID
            tokens: Number of tokens for this request

        Returns:
            Tuple of (allowed, error_message)
        """
        now = datetime.utcnow()

        # Initialize session if needed
        if session_id not in self.session_usage:
            self.session_usage[session_id] = {"minute_tokens": [], "hour_tokens": []}

        usage = self.session_usage[session_id]

        # Clean old entries
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)

        usage["minute_tokens"] = [
            (ts, t) for ts, t in usage["minute_tokens"] if ts > minute_ago
        ]

        usage["hour_tokens"] = [
            (ts, t) for ts, t in usage["hour_tokens"] if ts > hour_ago
        ]

        # Check limits
        minute_total = sum(t for _, t in usage["minute_tokens"])
        hour_total = sum(t for _, t in usage["hour_tokens"])

        if minute_total + tokens > self.tokens_per_minute:
            return (
                False,
                f"Rate limit exceeded: {minute_total + tokens} tokens in last minute (limit: {self.tokens_per_minute})",
            )

        if hour_total + tokens > self.tokens_per_hour:
            return (
                False,
                f"Rate limit exceeded: {hour_total + tokens} tokens in last hour (limit: {self.tokens_per_hour})",
            )

        # Record usage
        usage["minute_tokens"].append((now, tokens))
        usage["hour_tokens"].append((now, tokens))

        return True, None

    def get_usage(self, session_id: str) -> Dict[str, int]:
        """Get current usage for a session"""
        if session_id not in self.session_usage:
            return {"minute_tokens": 0, "hour_tokens": 0}

        usage = self.session_usage[session_id]
        minute_total = sum(t for _, t in usage["minute_tokens"])
        hour_total = sum(t for _, t in usage["hour_tokens"])

        return {
            "minute_tokens": minute_total,
            "hour_tokens": hour_total,
            "minute_limit": self.tokens_per_minute,
            "hour_limit": self.tokens_per_hour,
        }
