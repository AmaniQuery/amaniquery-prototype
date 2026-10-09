from typing import List, Dict, Optional
from .base_formatter import BaseFormatter


class TikTokFormatter(BaseFormatter):
    def __init__(self):
        super().__init__()
        self.MAX_CHARS = 2200

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
            content += f"Question: {query}\n\n"

        content += f"Here's the breakdown:\n\n{answer}\n\n"

        sources_text = self._format_sources_plain(sources)
        if sources_text:
            content += f"[DOCS] Sources:\n{sources_text}\n\n"

        if include_hashtags:
            hashtags = self._generate_hashtags(answer, sources, max_tags=7)
            hashtags.append("#learning")
            hashtags.append("#facts")
        else:
            hashtags = []

        content += " ".join(hashtags)

        if len(content) > limit:
            content = self._truncate_smart(content, limit)

        return {
            "platform": "tiktok",
            "content": content,
            "character_count": len(content),
            "hashtags": hashtags,
        }
