from typing import List, Dict, Optional
from .base_formatter import BaseFormatter


class InstagramFormatter(BaseFormatter):
    CHAR_LIMIT = 2200

    def __init__(self):
        super().__init__(char_limit=self.CHAR_LIMIT)

    def format_post(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str] = None,
        include_hashtags: bool = True,
    ) -> Dict:
        self._validate_input(answer, sources)

        caption_parts = []

        if query:
            caption_parts.append(f"{query}\n\n")

        main_content = self._truncate_smart(answer, 1500)
        caption_parts.append(main_content)

        if sources:
            caption_parts.append("\n\nSources:")
            for i, source in enumerate(sources[:3], 1):
                if isinstance(source, dict):
                    title = str(source.get("title", "")).strip()
                    url = str(source.get("url", "")).strip()
                    if title:
                        if url:
                            caption_parts.append(f"{i}. {title} - {url}")
                        else:
                            caption_parts.append(f"{i}. {title}")

        hashtags = (
            self._generate_hashtags(answer, sources, max_tags=15)
            if include_hashtags
            else []
        )
        if hashtags:
            caption_parts.append("\n\n" + " ".join(hashtags))

        caption = "".join(caption_parts)

        return {
            "platform": "instagram",
            "content": caption.strip(),
            "character_count": len(caption.strip()),
            "hashtags": hashtags,
        }
