from typing import List, Dict, Optional
from .base_formatter import BaseFormatter


class ThreadsFormatter(BaseFormatter):
    def __init__(self):
        super().__init__()
        self.MAX_CHARS = 500
        self.SINGLE_POST_LIMIT = 500

    def format_post(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str] = None,
        include_hashtags: bool = True,
    ) -> Dict:
        limit = self.MAX_CHARS
        posts = self._format_thread_structure(
            answer, sources, query, limit, include_hashtags
        )

        return {
            "platform": "threads",
            "content": posts,
            "character_count": sum(len(p) for p in posts),
            "hashtags": (
                self._generate_hashtags(answer, sources, max_tags=5)
                if include_hashtags
                else []
            ),
        }

    def _format_thread_structure(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str],
        limit: int,
        include_hashtags: bool,
    ) -> List[str]:
        posts = []

        start_text = ""
        if query:
            start_text = f"[THREAD] {query}\n\n"

        key_points = self._extract_key_points(answer, max_points=10)

        if not key_points:
            return self._smart_chunk(answer, limit)

        current_post = start_text

        for point in key_points:
            if len(current_post) + len(point) + 5 < limit:
                current_post += f"• {point}\n\n"
            else:
                if current_post.strip():
                    posts.append(current_post.strip())
                current_post = f"• {point}\n\n"

        if current_post.strip():
            posts.append(current_post.strip())

        sources_text = self._format_sources_plain(sources)
        hashtags = (
            self._generate_hashtags(answer, sources, max_tags=3)
            if include_hashtags
            else []
        )
        tags_text = " ".join(hashtags)

        footer = ""
        if sources_text:
            footer += f"\n\nSources:\n{sources_text}"
        if tags_text:
            footer += f"\n\n{tags_text}"

        if posts:
            if len(posts[-1] + footer) < limit:
                posts[-1] += footer
            else:
                posts.append(footer.strip())

        return posts

    def _smart_chunk(self, text: str, limit: int) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_len = 0

        for word in words:
            if current_len + len(word) + 1 > limit:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_len = len(word)
            else:
                current_chunk.append(word)
                current_len += len(word) + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
