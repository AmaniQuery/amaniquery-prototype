from typing import List, Dict, Optional
from .base_formatter import BaseFormatter


class WhatsAppFormatter(BaseFormatter):
    CHAR_LIMIT = None

    def __init__(self):
        super().__init__(char_limit=None)

    def format_post(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str] = None,
        include_hashtags: bool = False,
    ) -> Dict:
        self._validate_input(answer, sources)

        message_parts = []

        if query:
            message_parts.append(f"*{query}*\n\n")

        if len(answer) > 1000:
            answer = self._truncate_smart(answer, 1000, suffix="...")

        message_parts.append(answer)

        if sources:
            message_parts.append("\n\n*Sources:*")
            for i, source in enumerate(sources[:3], 1):
                if isinstance(source, dict):
                    title = str(source.get("title", "")).strip()
                    url = str(source.get("url", "")).strip()
                    if url:
                        message_parts.append(f"\n{i}. {title}\n   {url}")
                    else:
                        message_parts.append(f"\n{i}. {title}")

        message = "".join(message_parts)

        return {
            "platform": "whatsapp",
            "content": message.strip(),
            "character_count": len(message.strip()),
            "hashtags": [],
        }
