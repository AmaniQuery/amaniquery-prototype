from typing import List, Dict, Optional, Any
from .base_platform import BasePlatform, PlatformMetadata
from ..formatters.bluesky_formatter import BlueskyFormatter
import os
import urllib.parse

try:
    from atproto import Client as BlueskyClient
except ImportError:
    BlueskyClient = None


class BlueskyPlatform(BasePlatform):
    def __init__(self):
        self.formatter = BlueskyFormatter()
        self._client = None

    def get_metadata(self) -> PlatformMetadata:
        return PlatformMetadata(
            name="Bluesky",
            display_name="Bluesky",
            char_limit=300,
            supports_threads=False,
            supports_images=True,
            supports_video=False,
            posting_supported=True,
            requires_auth=True,
            features=["hashtags", "links"],
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
        base_url = "https://bsky.app/intent/compose"
        text_param = content
        if url:
            text_param += f"\n\n{url}"
        params = {"text": text_param}
        return f"{base_url}?{urllib.parse.urlencode(params)}"

    def _get_client(self):
        if self._client is not None:
            return self._client
        username = os.getenv("BLUESKY_USERNAME")
        password = os.getenv("BLUESKY_APP_PASSWORD")
        if not username or not password:
            return None
        if BlueskyClient is None:
            raise ImportError(
                "atproto not installed. Install with: pip install atproto"
            )
        client = BlueskyClient()
        client.login(username, password)
        self._client = client
        return self._client

    def post_content(
        self, content: str, media_urls: List[str] = None, auth_token: str = None
    ) -> Dict:
        try:
            client = self._get_client()
            if client is None:
                return {
                    "success": False,
                    "error": "Bluesky credentials not configured. Set BLUESKY_USERNAME and BLUESKY_APP_PASSWORD.",
                }
            post = client.send_post(text=content)
            return {
                "success": True,
                "post_id": post.uri,
                "post_url": f"https://bsky.app/profile/{os.getenv('BLUESKY_USERNAME')}/post/{post.uri.split('/')[-1]}",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to post to Bluesky: {e}",
            }
