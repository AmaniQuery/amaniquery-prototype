"""
LinkedIn formatter
"""

from typing import List, Dict, Optional
from .base_formatter import BaseFormatter


class LinkedInFormatter(BaseFormatter):
    """Format responses for LinkedIn"""

    CHAR_LIMIT = 3000
    OPTIMAL_LENGTH = 1300  # Sweet spot for engagement

    def __init__(self):
        super().__init__(char_limit=self.CHAR_LIMIT)

    def format_post(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str] = None,
        include_hashtags: bool = True,
    ) -> Dict:
        """Format response for LinkedIn"""
        # Validate input
        self._validate_input(answer, sources)

        # Professional opening
        opening = self._create_opening(query)

        # Format main content
        main_content = self._format_content(answer)

        # Key insights section
        insights = self._create_insights_section(answer)

        # Sources section
        sources_section = self._format_linkedin_sources(sources)

        # Hashtags
        hashtags = (
            self._generate_hashtags(answer, sources, max_tags=10)
            if include_hashtags
            else []
        )
        hashtag_text = "\n\n" + " ".join(hashtags) if hashtags else ""

        # Build sections efficiently
        sections = [opening, main_content]

        if insights:
            sections.append(insights)

        if sources_section:
            sections.append(sources_section)

        # Combine sections with proper spacing
        full_post = "\n\n".join(s.strip() for s in sections if s.strip())
        full_post += hashtag_text

        # Ensure within limit - truncate intelligently
        if len(full_post) > self.CHAR_LIMIT:
            # Calculate what we need to remove
            overflow = len(full_post) - self.CHAR_LIMIT

            # Reserve space for closing message
            closing_msg = "\n\n[Full analysis available at AmaniQuery]"
            available_for_content = (
                self.CHAR_LIMIT
                - len(opening)
                - len(insights)
                - len(sources_section)
                - len(hashtag_text)
                - len(closing_msg)
                - 20
            )  # spacing

            # Truncate main content
            if available_for_content > 100:
                main_content = self._truncate_smart(
                    main_content, available_for_content, suffix="..."
                )
                main_content += closing_msg
            else:
                # If we're really tight, truncate more aggressively
                main_content = self._truncate_smart(
                    answer, available_for_content, suffix="..."
                )
                main_content += closing_msg

            # Rebuild
            sections = [opening, main_content]
            if insights:
                sections.append(insights)
            if sources_section:
                sections.append(sources_section)

            full_post = "\n\n".join(s.strip() for s in sections if s.strip())
            full_post += hashtag_text

        return {
            "platform": "linkedin",
            "content": full_post.strip(),
            "character_count": len(full_post.strip()),
            "hashtags": hashtags,
            "optimal_length": len(full_post.strip()) <= self.OPTIMAL_LENGTH,
        }

    def _create_opening(self, query: Optional[str]) -> str:
        """Create professional opening (more natural)"""
        if query:
            return f"Question: {query}\n\nHere's what I found:"
        return "Key insights:"

    def _format_content(self, answer: str) -> str:
        """Format main content with proper structure"""
        # Split into paragraphs
        paragraphs = answer.split("\n\n")

        # Add professional formatting
        formatted = []
        for para in paragraphs[:4]:  # Limit paragraphs
            if para.strip():
                formatted.append(para.strip())

        return "\n\n".join(formatted)

    def _create_insights_section(self, answer: str) -> str:
        """Create key insights section"""
        key_points = self._extract_key_points(answer, max_points=3)

        if not key_points:
            return ""

        insights = "Key takeaways:\n"
        for i, point in enumerate(key_points, 1):
            # Truncate each point
            point = self._truncate_smart(point, 150, suffix="...")
            insights += f"  {i}. {point}\n"

        return insights.strip()

    def _format_linkedin_sources(self, sources: List[Dict]) -> str:
        """Format sources for LinkedIn"""
        if not sources:
            return ""

        sources_parts = ["References:"]

        for i, source in enumerate(sources[:5], 1):
            if not isinstance(source, dict):
                continue

            title = str(source.get("title", "Untitled")).strip() or "Untitled"
            url = str(source.get("url", "")).strip()
            source_name = str(source.get("source_name", "")).strip()
            category = str(source.get("category", "")).strip()

            # LinkedIn format with category (more natural)
            if category:
                sources_parts.append(f"{i}. {title} ({category})")
            else:
                sources_parts.append(f"{i}. {title}")

            if source_name:
                sources_parts.append(f"   {source_name}")
            if url:
                sources_parts.append(f"   {url}")

        sources_parts.append("\nPowered by AmaniQuery")

        return "\n".join(sources_parts).strip()
