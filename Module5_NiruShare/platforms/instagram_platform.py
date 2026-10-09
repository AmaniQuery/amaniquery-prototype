"""
Instagram platform plugin
"""

from typing import List, Dict, Optional, Union
from urllib.parse import quote

from .base_platform import BasePlatform, PlatformMetadata
from ..formatters.instagram_formatter import InstagramFormatter


class InstagramPlatform(BasePlatform):
    """Instagram platform handler"""

    def __init__(self):
        self.formatter = InstagramFormatter()
        super().__init__()

    def get_metadata(self) -> PlatformMetadata:
        """Return Instagram platform metadata"""
        return PlatformMetadata(
            name="instagram",
            display_name="Instagram",
            char_limit=2200,
            supports_threads=False,
            supports_images=True,
            supports_video=True,
            posting_supported=False,  # Requires Instagram Graph API
            requires_auth=True,
            features=["hashtags", "mentions", "images", "stories", "reels"],
        )

    def format_post(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str] = None,
        include_hashtags: bool = True,
        style: Optional[str] = None,
    ) -> Dict:
        """Format response for Instagram"""
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
        """Generate Instagram share link"""
        # Instagram doesn't support direct text sharing via URL
        # Return Instagram app URL
        if url:
            return f"https://www.instagram.com/"
        return "https://www.instagram.com/"
