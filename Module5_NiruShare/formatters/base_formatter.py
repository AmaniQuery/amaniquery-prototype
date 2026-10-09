"""
Base formatter for social media posts
"""

from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import re


from ..utils.text_processor import TextProcessor


class BaseFormatter(ABC):
    """Base class for social media formatters"""

    def __init__(self, char_limit: Optional[int] = None):
        self.char_limit = char_limit

    @abstractmethod
    def format_post(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str] = None,
        include_hashtags: bool = True,
    ) -> Dict:
        """
        Format response into social media post

        Returns:
            Dictionary with formatted post and metadata
        """
        pass

    def _validate_input(self, answer: str, sources: List[Dict]) -> None:
        """Validate input parameters"""
        if not answer or not isinstance(answer, str):
            raise ValueError("Answer must be a non-empty string")
        if not isinstance(sources, list):
            raise ValueError("Sources must be a list")

    def _extract_key_points(self, text: str, max_points: int = 3) -> List[str]:
        """Extract key points from text using robust processor"""
        return TextProcessor.extract_key_points(text, max_points)

    def _generate_hashtags(
        self, text: str, sources: List[Dict], max_tags: int = 5
    ) -> List[str]:
        """Generate relevant hashtags"""
        if not text:
            text = ""
        if not sources:
            sources = []

        hashtags = set()

        # Category-based hashtags
        categories = set()
        for source in sources[:3]:
            if isinstance(source, dict):
                cat = source.get("category", "")
                if cat and isinstance(cat, str):
                    categories.add(cat)

        category_map = {
            "Kenyan Law": ["KenyanLaw", "LegalKenya", "KenyaConstitution"],
            "Parliament": ["KenyanParliament", "KenyaPolitics", "ParliamentKE"],
            "Kenyan News": ["KenyaNews", "KenyaToday", "KenyaUpdates"],
            "Global Trend": ["GlobalTech", "TechPolicy", "AIPolicy"],
        }

        for cat in categories:
            tags = category_map.get(cat, [])
            hashtags.update(tags[:2])

        # Always include Kenya tag
        hashtags.add("Kenya")

        # Keyword-based hashtags
        keywords = ["AI", "technology", "law", "parliament", "policy", "constitution"]
        text_lower = text.lower() if text else ""

        for keyword in keywords:
            if keyword.lower() in text_lower:
                hashtag = keyword.title().replace(" ", "")
                if hashtag:
                    hashtags.add(hashtag)

        # Limit and format
        hashtags_list = list(hashtags)[:max_tags]
        return [f"#{tag}" if not tag.startswith("#") else tag for tag in hashtags_list]

    def _truncate_smart(self, text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text intelligently"""
        return TextProcessor.smart_truncate(text, max_length, suffix)

    def _format_sources(self, sources: List[Dict], max_sources: int = 3) -> str:
        """Format source citations"""
        if not sources:
            return ""

        citations = []
        for i, source in enumerate(sources[:max_sources], 1):
            if not isinstance(source, dict):
                continue

            title = source.get("title", "Untitled") or "Untitled"
            url = source.get("url", "")

            title = str(title).strip()

            if url:
                url = str(url).strip()
                citations.append(f"{i}. {title}\n   {url}")
            else:
                citations.append(f"{i}. {title}")

        if not citations:
            return ""

        return "\n\n[DOCS] Sources:\n" + "\n".join(citations)

    def _format_sources_plain(self, sources: List[Dict], max_sources: int = 3) -> str:
        """Format source citations as plain text lines (no DOCS header)"""
        if not sources:
            return ""

        lines = []
        for i, source in enumerate(sources[:max_sources], 1):
            if not isinstance(source, dict):
                continue
            title = str(source.get("title", "Untitled") or "Untitled").strip()
            url = str(source.get("url", "") or "").strip()
            if url:
                lines.append(f"{i}. {title} - {url}")
            else:
                lines.append(f"{i}. {title}")

        return "\n".join(lines)
