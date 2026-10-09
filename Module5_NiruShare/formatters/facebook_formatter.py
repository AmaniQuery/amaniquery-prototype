"""
Facebook formatter
"""

from typing import List, Dict, Optional
from .base_formatter import BaseFormatter


class FacebookFormatter(BaseFormatter):
    """Format responses for Facebook"""

    OPTIMAL_LENGTH = 500  # For better engagement

    def __init__(self):
        super().__init__(char_limit=None)  # No strict limit

    def format_post(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str] = None,
        include_hashtags: bool = True,
    ) -> Dict:
        """Format response for Facebook"""
        # Validate input
        self._validate_input(answer, sources)

        # Engaging opening
        opening = self._create_opening(query)

        # Format main content
        main_content = self._format_content(answer)

        # Call to action
        cta = self._create_cta()

        # Sources
        sources_section = self._format_facebook_sources(sources)

        # Hashtags (less prominent on Facebook)
        hashtags = (
            self._generate_hashtags(answer, sources, max_tags=5)
            if include_hashtags
            else []
        )
        hashtag_text = "\n\n" + " ".join(hashtags) if hashtags else ""

        # Build sections efficiently
        sections = [opening, main_content, cta]

        if sources_section:
            sections.append(sources_section)

        # Combine with proper spacing
        full_post = "\n\n".join(s.strip() for s in sections if s.strip())
        full_post += hashtag_text

        return {
            "platform": "facebook",
            "content": full_post.strip(),
            "character_count": len(full_post.strip()),
            "hashtags": hashtags,
            "word_count": len(full_post.strip().split()),
        }

    def _create_opening(self, query: Optional[str]) -> str:
        """Create engaging opening (more natural)"""
        if query:
            return f'Someone asked: "{query}"\n\nHere\'s what I found:'
        return "Here's something interesting:"

    def _format_content(self, answer: str) -> str:
        """Format content for Facebook engagement"""
        if not answer:
            return ""

        # Keep it concise for better engagement
        if len(answer) > 800:
            # Use key points format
            key_points = self._extract_key_points(answer, max_points=4)

            if not key_points:
                # Fallback: truncate answer
                return self._truncate_smart(answer, 800, suffix="...")

            formatted_parts = ["Key points:"]
            for i, point in enumerate(key_points, 1):
                # Ensure each point isn't too long
                point_text = self._truncate_smart(point, 200, suffix="...")
                formatted_parts.append(f"{i}. {point_text}")

            return "\n\n".join(formatted_parts)
        else:
            return answer

    def _create_cta(self) -> str:
        """Create call to action (more natural)"""
        ctas = [
            "What do you think? Share your thoughts below.",
            "Found this helpful? Share it with others!",
            "Want more insights? Follow for updates.",
        ]

        # Return first CTA for consistency
        return ctas[0]

    def _format_facebook_sources(self, sources: List[Dict]) -> str:
        """Format sources for Facebook (more natural)"""
        if not sources:
            return "Powered by AmaniQuery"

        sources_parts = ["Learn more:"]

        for i, source in enumerate(sources[:3], 1):
            if not isinstance(source, dict):
                continue

            title = str(source.get("title", "Untitled")).strip() or "Untitled"
            url = str(source.get("url", "")).strip()

            if url:
                sources_parts.append(f"• {title} - {url}")
            else:
                sources_parts.append(f"• {title}")

        sources_parts.append("\nPowered by AmaniQuery")

        return "\n".join(sources_parts).strip()
