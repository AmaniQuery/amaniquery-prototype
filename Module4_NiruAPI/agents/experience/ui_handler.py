"""
UI Handler - Handles user interface interactions
"""

from typing import Dict, Any, Optional
from loguru import logger


class UIHandler:
    """
    Handles UI interactions and formatting
    """

    def format_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format agent response for UI display

        Args:
            result: Raw agent result

        Returns:
            Formatted response for UI
        """
        formatted = {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.0),
            "metadata": result.get("metadata", {}),
        }

        # Format sources for display
        formatted_sources = []
        for source in formatted["sources"]:
            formatted_sources.append(
                {
                    "title": source.get("title", "Unknown"),
                    "url": source.get("url", ""),
                    "type": source.get("type", "unknown"),
                    "snippet": source.get("snippet", "")[:200],
                }
            )

        formatted["sources"] = formatted_sources

        # Format confidence as percentage
        formatted["confidence_percent"] = int(formatted["confidence"] * 100)

        return formatted

    def format_error(self, error: str) -> Dict[str, Any]:
        """
        Format error for UI display

        Args:
            error: Error message

        Returns:
            Formatted error response
        """
        return {
            "error": error,
            "answer": f"An error occurred: {error}",
            "sources": [],
            "confidence": 0.0,
        }
