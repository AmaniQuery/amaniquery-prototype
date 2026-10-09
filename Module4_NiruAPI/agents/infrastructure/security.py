"""
Security Manager - Handles security for agent operations
"""

from typing import Dict, Any, Optional, Tuple
import re
from loguru import logger


class SecurityManager:
    """
    Manages security for agent operations
    """

    def __init__(self):
        """Initialize security manager"""
        # Patterns for potentially dangerous operations
        self.dangerous_patterns = [
            r"rm\s+-rf",
            r"del\s+/f",
            r"format\s+",
            r"drop\s+table",
            r"delete\s+from",
            r"exec\s*\(",
            r"eval\s*\(",
        ]

    def validate_input(self, input_data: str) -> Tuple[bool, Optional[str]]:
        """
        Validate input for security

        Args:
            input_data: Input to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                return False, f"Potentially dangerous pattern detected: {pattern}"

        # Check for extremely long inputs (potential DoS)
        if len(input_data) > 100000:  # 100KB limit
            return False, "Input too long (potential DoS)"

        return True, None

    def sanitize_output(self, output: str) -> str:
        """
        Sanitize output to prevent injection

        Args:
            output: Output to sanitize

        Returns:
            Sanitized output
        """
        # Remove potential script tags
        output = re.sub(
            r"<script[^>]*>.*?</script>", "", output, flags=re.IGNORECASE | re.DOTALL
        )

        # Remove potential iframe tags
        output = re.sub(
            r"<iframe[^>]*>.*?</iframe>", "", output, flags=re.IGNORECASE | re.DOTALL
        )

        return output

    def validate_tool_call(
        self, tool_name: str, tool_args: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate tool call for security

        Args:
            tool_name: Name of tool
            tool_args: Tool arguments

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for file operations that might be dangerous
        if tool_name == "file_write":
            filename = tool_args.get("filename", "")
            # Prevent path traversal
            if ".." in filename or "/" in filename or "\\" in filename:
                return False, "Invalid filename: path traversal detected"

        # Check for URL fetch to potentially dangerous domains
        if tool_name == "url_fetch":
            url = tool_args.get("url", "")
            # In production, maintain a blocklist
            dangerous_domains = ["localhost", "127.0.0.1", "0.0.0.0"]
            if any(domain in url for domain in dangerous_domains):
                return False, "Access to localhost/internal URLs not allowed"

        return True, None
