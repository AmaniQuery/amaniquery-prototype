from typing import List, Dict, Optional
from .base_formatter import BaseFormatter


class BlueskyFormatter(BaseFormatter):
    def __init__(self):
        super().__init__()
        self.MAX_CHARS = 300

    def format_post(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str] = None,
        include_hashtags: bool = True,
    ) -> Dict:
        limit = self.MAX_CHARS

        content = ""
        if query:
            content += f"{query}\n\n"

        key_points = self._extract_key_points(answer, max_points=2)
        if key_points:
            content += " ".join(key_points)
        else:
            content += answer

        reserved_space = 50
        available_content = limit - reserved_space

        if len(content) > available_content:
            content = self._truncate_smart(content, available_content)

        if include_hashtags:
            hashtags = self._generate_hashtags(answer, sources, max_tags=3)
            hashtag_str = " ".join(hashtags)
        else:
            hashtags = []
            hashtag_str = ""

        final_post = content
        if hashtag_str:
            final_post += f"\n\n{hashtag_str}"

        return {
            "platform": "bluesky",
            "content": final_post,
            "character_count": len(final_post),
            "hashtags": hashtags,
        }
