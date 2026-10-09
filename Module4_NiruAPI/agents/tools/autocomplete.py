"""
Google Autocomplete Tool using SerpAPI
"""

from typing import Dict, Any, List, Optional
import os
from loguru import logger

try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None
    logger.warning(
        "google-search-results not installed. Install with: pip install google-search-results"
    )


class AutocompleteTool:
    """Google Autocomplete tool using SerpAPI"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize autocomplete tool

        Args:
            api_key: SerpAPI key (defaults to SERPAPI_API_KEY env var)
        """
        if GoogleSearch is None:
            raise ImportError("google-search-results package required")

        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            logger.warning("SERPAPI_API_KEY not set. Autocomplete will not work.")

    def execute(
        self,
        query: str,
        max_results: int = 10,
        location: Optional[str] = None,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Get Google autocomplete suggestions

        Args:
            query: Partial search query
            max_results: Maximum number of suggestions
            location: Location for localized suggestions (e.g., "Kenya")
            language: Language code (e.g., "en", "sw")

        Returns:
            Autocomplete suggestions with metadata
        """
        if not self.api_key:
            return {
                "query": query,
                "suggestions": [],
                "error": "SERPAPI_API_KEY not configured",
                "count": 0,
            }

        if not query or not query.strip():
            return {
                "query": query,
                "suggestions": [],
                "error": "Empty query provided",
                "count": 0,
            }

        try:
            params = {
                "q": query,
                "engine": "google_autocomplete",
                "api_key": self.api_key,
            }

            # Add optional parameters
            if location:
                params["location"] = location
            if language:
                params["hl"] = language

            search = GoogleSearch(params)
            results = search.get_dict()

            # Extract suggestions
            suggestions = []
            autocomplete_results = results.get("suggestions", [])

            for item in autocomplete_results[:max_results]:
                suggestion_text = (
                    item
                    if isinstance(item, str)
                    else item.get("value", item.get("suggestion", ""))
                )
                if suggestion_text:
                    suggestions.append(
                        {"text": suggestion_text, "query": suggestion_text}
                    )

            return {
                "query": query,
                "suggestions": suggestions,
                "count": len(suggestions),
                "location": location,
                "language": language,
            }
        except Exception as e:
            logger.error(f"Error in autocomplete: {e}")
            return {"query": query, "suggestions": [], "error": str(e), "count": 0}
