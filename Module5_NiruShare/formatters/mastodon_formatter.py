from typing import List, Dict, Optional
from .base_formatter import BaseFormatter


class MastodonFormatter(BaseFormatter):
    CHAR_LIMIT = 500

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

        toot_parts = []

        if query:
            query_text = str(query).strip()
            toot_parts.append(f"{query_text}\n\n")

        available_space = self.CHAR_LIMIT - len("".join(toot_parts))
        main_content = self._truncate_smart(answer, available_space - 50)
        toot_parts.append(main_content)

        if sources:
            source = sources[0]
            if isinstance(source, dict):
                url = str(source.get("url", "")).strip()
                if url:
                    toot_parts.append(f"\n\n{url}")

        hashtags = (
            self._generate_hashtags(answer, sources, max_tags=5)
            if include_hashtags
            else []
        )
        if hashtags:
            hashtag_text = " " + " ".join(hashtags)
            if len("".join(toot_parts)) + len(hashtag_text) <= self.CHAR_LIMIT:
                toot_parts.append(hashtag_text)

        toot = "".join(toot_parts)

        if len(toot) > self.CHAR_LIMIT:
            toot = self._truncate_smart(toot, self.CHAR_LIMIT)

        return {
            "platform": "mastodon",
            "content": toot.strip(),
            "character_count": len(toot.strip()),
            "hashtags": hashtags,
        }
