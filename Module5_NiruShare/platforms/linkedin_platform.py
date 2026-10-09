"""
LinkedIn platform plugin
"""

from typing import List, Dict, Optional, Union
from urllib.parse import quote
import os
import requests
from datetime import datetime

from .base_platform import BasePlatform, PlatformMetadata
from ..formatters.linkedin_formatter import LinkedInFormatter


class LinkedInPlatform(BasePlatform):
    """LinkedIn platform handler"""

    def __init__(self):
        self.formatter = LinkedInFormatter()
        super().__init__()

    def get_metadata(self) -> PlatformMetadata:
        """Return LinkedIn platform metadata"""
        return PlatformMetadata(
            name="linkedin",
            display_name="LinkedIn",
            char_limit=3000,
            supports_threads=False,
            supports_images=True,
            supports_video=False,
            posting_supported=True,
            requires_auth=True,
            features=["hashtags", "mentions", "links", "rich_media"],
        )

    def format_post(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str] = None,
        include_hashtags: bool = True,
        style: Optional[str] = None,
    ) -> Dict:
        """Format response for LinkedIn"""
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
        """Generate LinkedIn share link"""
        # LinkedIn doesn't support pre-filled text via URL
        if url:
            encoded_url = quote(url)
            return f"https://www.linkedin.com/sharing/share-offsite/?url={encoded_url}"
        else:
            return "https://www.linkedin.com/feed/"

    def _post_impl(
        self,
        content: Union[str, List[str]],
        access_token: str,
        message_id: Optional[str] = None,
    ) -> Dict:
        """Post to LinkedIn"""
        try:
            if isinstance(content, list):
                text = content[0] if content else ""
            else:
                text = str(content)

            # Get user info to get person URN
            headers = {"Authorization": f"Bearer {access_token}"}
            user_response = requests.get(
                "https://api.linkedin.com/v2/people/~", headers=headers
            )
            user_response.raise_for_status()
            user_data = user_response.json()
            person_urn = user_data.get("id")

            if not person_urn:
                return {
                    "platform": "linkedin",
                    "status": "error",
                    "message": "Could not retrieve LinkedIn user ID",
                    "metadata": {"message_id": message_id},
                }

            # Post to LinkedIn
            url = "https://api.linkedin.com/v2/ugcPosts"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "X-Restli-Protocol-Version": "2.0.0",
            }

            data = {
                "author": f"urn:li:person:{person_urn}",
                "lifecycleState": "PUBLISHED",
                "specificContent": {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {"text": text},
                        "shareMediaCategory": "NONE",
                    }
                },
                "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
            }

            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()

            result = response.json()
            post_id = result.get("id")

            return {
                "platform": "linkedin",
                "post_id": post_id,
                "status": "success",
                "message": "LinkedIn post created successfully",
                "url": (
                    f"https://www.linkedin.com/feed/update/{post_id}/"
                    if post_id
                    else None
                ),
                "metadata": {
                    "message_id": message_id,
                    "posted_at": datetime.utcnow().isoformat(),
                },
            }
        except requests.exceptions.RequestException as e:
            return {
                "platform": "linkedin",
                "status": "error",
                "message": f"Failed to post to LinkedIn: {str(e)}",
                "metadata": {"message_id": message_id},
            }

    def _get_auth_url_impl(self, redirect_uri: Optional[str] = None) -> Dict:
        """Get LinkedIn OAuth URL"""
        client_id = os.getenv("LINKEDIN_CLIENT_ID")
        redirect_uri = redirect_uri or os.getenv(
            "LINKEDIN_REDIRECT_URI", "http://localhost:8000/share/auth/callback"
        )

        if not client_id:
            return {
                "platform": "linkedin",
                "status": "error",
                "message": "LinkedIn API credentials not configured",
            }

        scope = "w_member_social,r_liteprofile"
        auth_url = (
            f"https://www.linkedin.com/oauth/v2/authorization?"
            f"response_type=code&client_id={client_id}&redirect_uri={quote(redirect_uri)}"
            f"&scope={quote(scope)}&state=linkedin"
        )

        return {
            "platform": "linkedin",
            "status": "needs_auth",
            "auth_url": auth_url,
            "message": "Click to authenticate with LinkedIn",
        }
