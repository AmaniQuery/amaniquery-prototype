"""
Facebook platform plugin
"""

from typing import List, Dict, Optional, Union
from urllib.parse import quote
import os
import requests
from datetime import datetime

from .base_platform import BasePlatform, PlatformMetadata
from ..formatters.facebook_formatter import FacebookFormatter


class FacebookPlatform(BasePlatform):
    """Facebook platform handler"""

    def __init__(self):
        self.formatter = FacebookFormatter()
        super().__init__()

    def get_metadata(self) -> PlatformMetadata:
        """Return Facebook platform metadata"""
        return PlatformMetadata(
            name="facebook",
            display_name="Facebook",
            char_limit=None,  # No strict limit
            supports_threads=False,
            supports_images=True,
            supports_video=True,
            posting_supported=True,
            requires_auth=True,
            features=["hashtags", "mentions", "links", "rich_media", "video"],
        )

    def format_post(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str] = None,
        include_hashtags: bool = True,
        style: Optional[str] = None,
    ) -> Dict:
        """Format response for Facebook"""
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
        """Generate Facebook share link"""
        if url:
            encoded_url = quote(url)
            return f"https://www.facebook.com/sharer/sharer.php?u={encoded_url}"
        else:
            return "https://www.facebook.com/sharer/sharer.php"

    def _post_impl(
        self,
        content: Union[str, List[str]],
        access_token: str,
        message_id: Optional[str] = None,
    ) -> Dict:
        """Post to Facebook"""
        try:
            if isinstance(content, list):
                text = content[0] if content else ""
            else:
                text = str(content)

            url = f"https://graph.facebook.com/me/feed"

            data = {
                "message": text,
                "access_token": access_token,
            }

            response = requests.post(url, data=data)
            response.raise_for_status()

            result = response.json()
            post_id = result.get("id")

            return {
                "platform": "facebook",
                "post_id": post_id,
                "status": "success",
                "message": "Facebook post created successfully",
                "url": f"https://www.facebook.com/{post_id}" if post_id else None,
                "metadata": {
                    "message_id": message_id,
                    "posted_at": datetime.utcnow().isoformat(),
                },
            }
        except requests.exceptions.RequestException as e:
            return {
                "platform": "facebook",
                "status": "error",
                "message": f"Failed to post to Facebook: {str(e)}",
                "metadata": {"message_id": message_id},
            }

    def _get_auth_url_impl(self, redirect_uri: Optional[str] = None) -> Dict:
        """Get Facebook OAuth URL"""
        app_id = os.getenv("FACEBOOK_APP_ID")
        redirect_uri = redirect_uri or os.getenv(
            "FACEBOOK_REDIRECT_URI", "http://localhost:8000/share/auth/callback"
        )

        if not app_id:
            return {
                "platform": "facebook",
                "status": "error",
                "message": "Facebook API credentials not configured",
            }

        scope = "pages_manage_posts,publish_to_groups"
        auth_url = (
            f"https://www.facebook.com/v18.0/dialog/oauth?"
            f"client_id={app_id}&redirect_uri={quote(redirect_uri)}"
            f"&scope={quote(scope)}&response_type=code&state=facebook"
        )

        return {
            "platform": "facebook",
            "status": "needs_auth",
            "auth_url": auth_url,
            "message": "Click to authenticate with Facebook",
        }
