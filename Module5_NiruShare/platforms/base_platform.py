"""
Base platform interface for social media platforms
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class PlatformMetadata:
    """Metadata about a platform"""

    name: str
    display_name: str
    char_limit: Optional[int]
    supports_threads: bool = False
    supports_images: bool = True
    supports_video: bool = False
    posting_supported: bool = False
    requires_auth: bool = True
    features: List[str] = None

    def __post_init__(self):
        if self.features is None:
            self.features = []


class BasePlatform(ABC):
    """Abstract base class for social media platform handlers"""

    def __init__(self):
        self.metadata = self.get_metadata()

    @abstractmethod
    def get_metadata(self) -> PlatformMetadata:
        """Return platform metadata"""
        pass

    @abstractmethod
    def format_post(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str] = None,
        include_hashtags: bool = True,
        style: Optional[str] = None,
    ) -> Dict:
        """
        Format response for this platform

        Args:
            answer: The RAG answer
            sources: List of source dictionaries
            query: Original query
            include_hashtags: Whether to include hashtags
            style: Formatting style (professional, casual, engaging)

        Returns:
            Dictionary with formatted post and metadata
        """
        pass

    @abstractmethod
    def generate_share_link(
        self,
        content: Union[str, List[str]],
        url: Optional[str] = None,
    ) -> str:
        """
        Generate platform-specific share link

        Args:
            content: Formatted content (string or list for threads)
            url: Optional URL to include

        Returns:
            Share URL
        """
        pass

    def post_to_platform(
        self,
        content: Union[str, List[str]],
        access_token: str,
        message_id: Optional[str] = None,
    ) -> Dict:
        """
        Post content directly to platform (if supported)

        Args:
            content: Content to post
            access_token: OAuth access token
            message_id: Optional message ID for tracking

        Returns:
            Post result with ID and metadata
        """
        if not self.metadata.posting_supported:
            raise NotImplementedError(
                f"Direct posting not supported for {self.metadata.name}"
            )
        return self._post_impl(content, access_token, message_id)

    def _post_impl(
        self,
        content: Union[str, List[str]],
        access_token: str,
        message_id: Optional[str] = None,
    ) -> Dict:
        """Internal implementation of posting (override in subclasses)"""
        raise NotImplementedError(f"Posting not implemented for {self.metadata.name}")

    def get_auth_url(self, redirect_uri: Optional[str] = None) -> Dict:
        """
        Get OAuth authorization URL

        Args:
            redirect_uri: Optional redirect URI

        Returns:
            Dictionary with auth_url and metadata
        """
        if not self.metadata.requires_auth:
            return {
                "platform": self.metadata.name,
                "status": "not_required",
                "message": "Authentication not required for this platform",
            }
        return self._get_auth_url_impl(redirect_uri)

    def _get_auth_url_impl(self, redirect_uri: Optional[str] = None) -> Dict:
        """Internal implementation of auth URL generation (override in subclasses)"""
        return {
            "platform": self.metadata.name,
            "status": "not_implemented",
            "message": f"Authentication not yet implemented for {self.metadata.name}",
        }

    def validate_content(self, content: Union[str, List[str]]) -> bool:
        """
        Validate content against platform constraints

        Args:
            content: Content to validate

        Returns:
            True if valid, False otherwise
        """
        if isinstance(content, list):
            # For threads, validate each item
            for item in content:
                if not isinstance(item, str):
                    return False
                if self.metadata.char_limit and len(item) > self.metadata.char_limit:
                    return False
            return True
        else:
            if not isinstance(content, str):
                return False
            if self.metadata.char_limit and len(content) > self.metadata.char_limit:
                return False
            return True
