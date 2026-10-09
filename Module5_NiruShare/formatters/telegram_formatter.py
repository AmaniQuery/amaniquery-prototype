from typing import List, Dict, Optional
from .base_formatter import BaseFormatter


class TelegramFormatter(BaseFormatter):
    CHAR_LIMIT = 4096

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

        message_parts = []

        if query:
            message_parts.append(f"*{query}*\n\n")

        message_parts.append(answer)

        if sources:
            message_parts.append("\n\n*Sources:*")
            for i, source in enumerate(sources[:5], 1):
                if isinstance(source, dict):
                    title = str(source.get("title", "")).strip()
                    url = str(source.get("url", "")).strip()
                    if url:
                        message_parts.append(f"\n{i}. [{title}]({url})")
                    else:
                        message_parts.append(f"\n{i}. {title}")

        hashtags = (
            self._generate_hashtags(answer, sources, max_tags=10)
            if include_hashtags
            else []
        )
        if hashtags:
            message_parts.append("\n\n" + " ".join(hashtags))

        message = "".join(message_parts)

        return {
            "platform": "telegram",
            "content": message.strip(),
            "character_count": len(message.strip()),
            "hashtags": hashtags,
        }
