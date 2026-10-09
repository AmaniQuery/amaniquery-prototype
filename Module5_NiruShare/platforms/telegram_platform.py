"""
Telegram platform plugin
"""

from typing import List, Dict, Optional, Union
from urllib.parse import quote

from .base_platform import BasePlatform, PlatformMetadata
from ..formatters.telegram_formatter import TelegramFormatter


class TelegramPlatform(BasePlatform):
    """Telegram platform handler"""

    def __init__(self):
        self.formatter = TelegramFormatter()
        super().__init__()

    def get_metadata(self) -> PlatformMetadata:
        """Return Telegram platform metadata"""
        return PlatformMetadata(
            name="telegram",
            display_name="Telegram",
            char_limit=4096,
            supports_threads=False,
            supports_images=True,
            supports_video=True,
            posting_supported=False,  # Requires Telegram Bot API
            requires_auth=True,
            features=["markdown", "links", "images", "videos", "channels"],
        )

    def format_post(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str] = None,
        include_hashtags: bool = True,
        style: Optional[str] = None,
    ) -> Dict:
        """Format response for Telegram"""
        result = self.formatter.format_post(
            answer=answer,
            sources=sources,
            query=query,
            include_hashtags=include_hashtags,
        )
        if style:
            result["style"] = style
        return result

    def generate_share_link(
        self,
        content: Union[str, List[str]],
        url: Optional[str] = None,
    ) -> str:
        """Generate Telegram share link"""
        if url:
            encoded_url = quote(url)
            text = content[0] if isinstance(content, list) else str(content)
            encoded_text = quote(text[:100])  # Limit text length
            return f"https://t.me/share/url?url={encoded_url}&text={encoded_text}"
        return "https://t.me/share/url"
