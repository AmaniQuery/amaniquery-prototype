import os
import urllib.parse
from typing import List, Dict, Optional, Any
from .base_platform import BasePlatform, PlatformMetadata
from ..formatters.threads_formatter import ThreadsFormatter

try:
    import requests
except ImportError:
    requests = None


class ThreadsPlatform(BasePlatform):
    def __init__(self):
        self.formatter = ThreadsFormatter()

    def get_metadata(self) -> PlatformMetadata:
        return PlatformMetadata(
            name="Threads",
            display_name="Threads",
            char_limit=500,
            supports_threads=True,
            supports_images=True,
            supports_video=True,
            posting_supported=True,
            requires_auth=True,
            features=["hashtags", "links", "images"],
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
        base_url = "https://threads.net/intent/post"
        text_param = content
        if url:
            text_param += f"\n\n{url}"
        params = {"text": text_param}
        return f"{base_url}?{urllib.parse.urlencode(params)}"

    def post_content(
        self, content: str, media_urls: List[str] = None, auth_token: str = None
    ) -> Dict:
        access_token = auth_token or os.getenv("THREADS_ACCESS_TOKEN")
        user_id = os.getenv("THREADS_USER_ID")
        if not access_token or not user_id:
            return {
                "success": False,
                "error": "Threads API credentials not configured. Set THREADS_ACCESS_TOKEN and THREADS_USER_ID.",
            }
        if requests is None:
            return {
                "success": False,
                "error": "requests not installed. Install with: pip install requests",
            }

        try:
            # Step 1: Create media container
            create_url = f"https://graph.threads.net/v1.0/{user_id}/threads"
            create_params = {
                "media_type": "TEXT",
                "text": content,
                "access_token": access_token,
            }
            create_resp = requests.post(create_url, params=create_params)
            create_resp.raise_for_status()
            container_id = create_resp.json().get("id")
            if not container_id:
                return {
                    "success": False,
                    "error": "Failed to get container ID from Threads API",
                }

            # Step 2: Publish the container
            publish_url = f"https://graph.threads.net/v1.0/{user_id}/threads_publish"
            publish_params = {
                "creation_id": container_id,
                "access_token": access_token,
            }
            publish_resp = requests.post(publish_url, params=publish_params)
            publish_resp.raise_for_status()
            media_id = publish_resp.json().get("id")

            return {
                "success": True,
                "post_id": media_id,
                "post_url": f"https://threads.net/t/{media_id}",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to post to Threads: {e}",
            }
