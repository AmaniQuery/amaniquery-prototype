"""
Social Media Sharing Service
Updated with robust rate limiting and error handling (2026)
"""

from typing import Dict, List, Optional, Union
import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from .platforms import (
    PlatformRegistry,
    TwitterPlatform,
    LinkedInPlatform,
    FacebookPlatform,
    InstagramPlatform,
    RedditPlatform,
    TelegramPlatform,
    WhatsAppPlatform,
    MastodonPlatform,
    ThreadsPlatform,
    BlueskyPlatform,
    TikTokPlatform,
)
from .image_generator import ImageGenerator
from .utils.cache import SimpleCache
from .utils.rate_limiter import _limiter, robust_api_call

logger = logging.getLogger(__name__)


class ShareService:
    """Service for social media sharing with platform registry and image generation"""

    def __init__(self, enable_cache: bool = True, cache_ttl: int = 3600):
        """
        Initialize service with platform registry

        Args:
            enable_cache: Enable caching for formatted posts
            cache_ttl: Cache time-to-live in seconds
        """
        self.registry = PlatformRegistry()
        self._register_default_platforms()

        # Initialize cache
        self.cache = SimpleCache(default_ttl=cache_ttl) if enable_cache else None

        # Initialize image generator
        try:
            self.image_generator = ImageGenerator()
        except ImportError:
            logger.warning("Pillow not available, image generation disabled")
            self.image_generator = None

        # Thread pool for sync operations
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _register_default_platforms(self):
        """Register all default platforms"""
        platforms = [
            TwitterPlatform(),
            LinkedInPlatform(),
            FacebookPlatform(),
            InstagramPlatform(),
            RedditPlatform(),
            TelegramPlatform(),
            WhatsAppPlatform(),
            MastodonPlatform(),
            ThreadsPlatform(),
            BlueskyPlatform(),
            TikTokPlatform(),
        ]

        for platform in platforms:
            try:
                self.registry.register(platform)
                logger.info(f"Registered platform: {platform.get_metadata().name}")
            except Exception as e:
                # Log error but continue
                logger.warning(
                    f"Failed to register platform {platform.__class__.__name__}: {e}"
                )

    def format_for_platform(
        self,
        answer: str,
        sources: List[Dict],
        platform: str,
        query: Optional[str] = None,
        include_hashtags: bool = True,
        style: Optional[str] = None,
    ) -> Dict:
        """
        Format response for specific platform
        """
        # Validate inputs
        if not answer or not isinstance(answer, str):
            raise ValueError("Answer must be a non-empty string")
        if not isinstance(sources, list):
            raise ValueError("Sources must be a list")
        if not platform or not isinstance(platform, str):
            raise ValueError("Platform must be a non-empty string")

        platform = platform.lower().strip()

        # Check cache first
        if self.cache:
            cache_key = self.cache._make_key(
                "format_post", platform, answer, sources, query, include_hashtags, style
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for platform: {platform}")
                return cached

        # Get platform from registry
        platform_handler = self.registry.get(platform)
        if not platform_handler:
            available = ", ".join(self.registry.list_platforms())
            raise ValueError(
                f"Unsupported platform: {platform}. Available: {available}"
            )

        # Normalize query
        if query is not None:
            query = str(query).strip() if query else None

        try:
            result = platform_handler.format_post(
                answer=answer,
                sources=sources,
                query=query,
                include_hashtags=include_hashtags,
                style=style,
            )

            # Cache result
            if self.cache:
                cache_key = self.cache._make_key(
                    "format_post",
                    platform,
                    answer,
                    sources,
                    query,
                    include_hashtags,
                    style,
                )
                self.cache.set(cache_key, result)

            return result
        except ValueError as e:
            raise
        except Exception as e:
            logger.error(f"Error formatting post for {platform}: {e}", exc_info=True)
            raise ValueError(f"Error formatting post for {platform}: {str(e)}") from e

    def generate_share_link(
        self,
        platform: str,
        formatted_content: Union[str, List[str]],
        url: Optional[str] = None,
    ) -> str:
        """Generate platform-specific share link"""
        if not platform or not isinstance(platform, str):
            raise ValueError("Platform must be a non-empty string")

        if not formatted_content:
            raise ValueError("Formatted content cannot be empty")

        platform = platform.lower().strip()

        # Get platform from registry
        platform_handler = self.registry.get(platform)
        if not platform_handler:
            available = ", ".join(self.registry.list_platforms())
            raise ValueError(
                f"Unsupported platform: {platform}. Available: {available}"
            )

        try:
            return platform_handler.generate_share_link(
                content=formatted_content,
                url=url,
            )
        except Exception as e:
            raise ValueError(
                f"Error generating share link for {platform}: {str(e)}"
            ) from e

    def preview_all_platforms(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str] = None,
        style: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """Preview formatted posts for all platforms"""
        if not answer or not isinstance(answer, str):
            raise ValueError("Answer must be a non-empty string")
        if not isinstance(sources, list):
            raise ValueError("Sources must be a list")

        previews = {}

        for platform_name in self.registry.list_platforms():
            try:
                previews[platform_name] = self.format_for_platform(
                    answer=answer,
                    sources=sources,
                    platform=platform_name,
                    query=query,
                    style=style,
                )
            except Exception as e:
                previews[platform_name] = {
                    "error": str(e),
                    "platform": platform_name,
                    "status": "error",
                }

        return previews

    def get_platform_stats(self, formatted_post: Dict) -> Dict:
        """Get statistics for formatted post"""
        platform = formatted_post.get("platform")
        content = formatted_post.get("content")

        if isinstance(content, list):
            total_chars = sum(len(str(tweet)) for tweet in content if tweet)
            return {
                "platform": platform,
                "type": "thread",
                "tweet_count": len(content),
                "total_characters": total_chars,
                "avg_chars_per_tweet": total_chars // len(content) if content else 0,
            }
        else:
            content_str = str(content) if content else ""
            return {
                "platform": platform,
                "type": "single",
                "character_count": len(content_str),
                "word_count": len(content_str.split()),
                "hashtag_count": len(formatted_post.get("hashtags", [])),
            }

    async def post_to_platform(
        self,
        platform: str,
        content: Union[str, List[str]],
        access_token: str,
        message_id: Optional[str] = None,
    ) -> Dict:
        """
        Post content to a social media platform (Async with Rate Limiting)
        """
        platform = platform.lower().strip()

        # Get platform from registry
        platform_handler = self.registry.get(platform)
        if not platform_handler:
            available = ", ".join(self.registry.list_platforms())
            raise ValueError(
                f"Unsupported platform: {platform}. Available: {available}"
            )

        try:
            # 1. Rate Limiting Check
            await _limiter.wait_for_token(platform)

            # 2. Execute with Retry Logic in ThreadPool
            # We use a helper function to wrap the sync call with tenacity retry
            @robust_api_call(max_retries=3)
            def _execute_post():
                return platform_handler.post_to_platform(
                    content=content,
                    access_token=access_token,
                    message_id=message_id,
                )

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self._executor, _execute_post)
            return result

        except NotImplementedError as e:
            return {
                "platform": platform,
                "status": "error",
                "message": str(e),
                "metadata": {"message_id": message_id},
            }
        except Exception as e:
            logger.error(f"Failed to post to {platform}: {e}")
            return {
                "platform": platform,
                "status": "error",
                "message": f"Failed to post: {str(e)}",
                "metadata": {"message_id": message_id},
            }

    def get_auth_url(self, platform: str, redirect_uri: Optional[str] = None) -> Dict:
        """Get OAuth authorization URL for a platform"""
        platform = platform.lower().strip()

        platform_handler = self.registry.get(platform)
        if not platform_handler:
            available = ", ".join(self.registry.list_platforms())
            raise ValueError(
                f"Unsupported platform: {platform}. Available: {available}"
            )

        return platform_handler.get_auth_url(redirect_uri=redirect_uri)

    def generate_image(
        self,
        text: str,
        title: Optional[str] = None,
        color_scheme: str = "default",
        width: Optional[int] = None,
        height: Optional[int] = None,
        format: str = "PNG",
    ) -> Dict:
        """Generate image from text content"""
        if not self.image_generator:
            raise ValueError(
                "Image generation not available. Install Pillow: pip install Pillow"
            )

        try:
            image_base64 = self.image_generator.generate_image_base64(
                text=text,
                title=title,
                color_scheme=color_scheme,
                width=width,
                height=height,
                format=format,
            )

            return {
                "status": "success",
                "format": format.lower(),
                "image_base64": image_base64,
                "width": width or self.image_generator.DEFAULT_WIDTH,
                "height": height or self.image_generator.DEFAULT_HEIGHT,
            }
        except Exception as e:
            raise ValueError(f"Error generating image: {str(e)}") from e

    def generate_image_from_post(
        self,
        post_content: str,
        query: Optional[str] = None,
        color_scheme: str = "default",
        format: str = "PNG",
        **kwargs,
    ) -> Dict:
        """Generate image from formatted post content"""
        if not self.image_generator:
            raise ValueError(
                "Image generation not available. Install Pillow: pip install Pillow"
            )

        try:
            image_base64 = self.image_generator.generate_image_base64(
                text=post_content,
                title=query,
                color_scheme=color_scheme,
                format=format,
                **kwargs,
            )

            return {
                "status": "success",
                "format": format.lower(),
                "image_base64": image_base64,
                "width": kwargs.get("width") or self.image_generator.DEFAULT_WIDTH,
                "height": kwargs.get("height") or self.image_generator.DEFAULT_HEIGHT,
            }
        except Exception as e:
            raise ValueError(f"Error generating image from post: {str(e)}") from e

    def get_supported_platforms(self) -> List[Dict]:
        """Get list of all supported platforms with metadata"""
        platforms = []
        for metadata in self.registry.list_metadata():
            platforms.append(
                {
                    "name": metadata.name,
                    "display_name": metadata.display_name,
                    "char_limit": metadata.char_limit,
                    "supports_threads": metadata.supports_threads,
                    "supports_images": metadata.supports_images,
                    "supports_video": metadata.supports_video,
                    "posting_supported": metadata.posting_supported,
                    "requires_auth": metadata.requires_auth,
                    "features": metadata.features,
                }
            )
        return platforms
