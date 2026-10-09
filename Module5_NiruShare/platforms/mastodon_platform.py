"""
Mastodon platform plugin
"""

from typing import List, Dict, Optional, Union
from urllib.parse import quote

from .base_platform import BasePlatform, PlatformMetadata
from ..formatters.mastodon_formatter import MastodonFormatter


class MastodonPlatform(BasePlatform):
    """Mastodon platform handler"""

    def __init__(self):
        self.formatter = MastodonFormatter()
        super().__init__()

    def get_metadata(self) -> PlatformMetadata:
        """Return Mastodon platform metadata"""
        return PlatformMetadata(
            name="mastodon",
            display_name="Mastodon",
            char_limit=500,
            supports_threads=True,  # Threads via replies
            supports_images=True,
            supports_video=True,
            posting_supported=False,  # Requires Mastodon API
            requires_auth=True,
            features=["hashtags", "mentions", "links", "images", "threads"],
        )

    def format_post(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str] = None,
        include_hashtags: bool = True,
        style: Optional[str] = None,
    ) -> Dict:
        """Format response for Mastodon"""
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
        """Generate Mastodon share link"""
        # Mastodon doesn't have a universal share URL
        # Return a generic share URL (would need instance URL)
        text = content[0] if isinstance(content, list) else str(content)
        encoded_text = quote(text)

        # Generic format (would need instance URL in production)
        if url:
            encoded_url = quote(url)
            return (
                f"https://mastodon.social/share?text={encoded_text}&url={encoded_url}"
            )
        return f"https://mastodon.social/share?text={encoded_text}"
