from typing import List, Dict, Optional, Any
from .base_platform import BasePlatform, PlatformMetadata
from ..formatters.tiktok_formatter import TikTokFormatter


class TikTokPlatform(BasePlatform):
    def __init__(self):
        self.formatter = TikTokFormatter()

    def get_metadata(self) -> PlatformMetadata:
        return PlatformMetadata(
            name="TikTok",
            display_name="TikTok",
            char_limit=2200,
            supports_threads=False,
            supports_images=True,
            supports_video=True,
            posting_supported=False,
            requires_auth=False,
            features=["hashtags", "mentions", "links", "video"],
        )

    def format_post(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str] = None,
        include_hashtags: bool = True,
        style: Optional[str] = None,
    ) -> Dict:
        result = self.formatter.format_post(
            answer=answer,
            sources=sources,
            query=query,
            include_hashtags=include_hashtags,
        )
        if style:
            result["style"] = style
        return result

    def generate_share_link(self, content: str, url: Optional[str] = None) -> str:
        # TikTok doesn't have a simple "web intent" for text posting like Twitter/Threads
        # It's video first. Usually web share links are for sharing specific videos.
        # However, for mobile deep linking, we might try a generic approach or just copy content.
        # Since this is a web app, the best fallback is often just copying the text or opening the app.

        # We will return the homepage for now, as TikTok requires the app for creation.
        # The frontend handles "copy text" which is the primary use case for TikTok (copy caption).
        return "https://www.tiktok.com/"

    def post_content(
        self, content: str, media_urls: List[str] = None, auth_token: str = None
    ) -> Dict:
        return {
            "success": False,
            "error": "Direct posting to TikTok is not supported via this API. Please copy the caption and use the TikTok app.",
        }
