from typing import List, Dict, Optional
from .base_formatter import BaseFormatter


class RedditFormatter(BaseFormatter):
    CHAR_LIMIT = 40000

    def __init__(self):
        super().__init__(char_limit=self.CHAR_LIMIT)

    def format_post(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str] = None,
        include_hashtags: bool = False,
    ) -> Dict:
        self._validate_input(answer, sources)

        post_parts = []

        if query:
            post_parts.append(f"**Question:** {query}\n\n")

        post_parts.append("**Answer:**\n\n")
        post_parts.append(answer)

        if sources:
            post_parts.append("\n\n**Sources:**\n\n")
            for i, source in enumerate(sources[:5], 1):
                if isinstance(source, dict):
                    title = str(source.get("title", "")).strip()
                    url = str(source.get("url", "")).strip()
                    if url:
                        post_parts.append(f"{i}. [{title}]({url})\n\n")
                    else:
                        post_parts.append(f"{i}. {title}\n\n")

        post = "".join(post_parts)

        return {
            "platform": "reddit",
            "content": post.strip(),
            "character_count": len(post.strip()),
            "hashtags": [],
        }
