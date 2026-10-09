"""
Platform registry for managing social media platform plugins
"""

from typing import Dict, Optional, List
from .base_platform import BasePlatform, PlatformMetadata


class PlatformRegistry:
    """Registry for managing platform plugins"""

    def __init__(self):
        self._platforms: Dict[str, BasePlatform] = {}
        self._metadata: Dict[str, PlatformMetadata] = {}

    def register(self, platform: BasePlatform) -> None:
        """
        Register a platform plugin

        Args:
            platform: Platform instance to register
        """
        if not isinstance(platform, BasePlatform):
            raise TypeError("Platform must be an instance of BasePlatform")

        metadata = platform.get_metadata()
        platform_name = metadata.name.lower()

        if platform_name in self._platforms:
            raise ValueError(f"Platform '{platform_name}' is already registered")

        self._platforms[platform_name] = platform
        self._metadata[platform_name] = metadata

    def get(self, platform_name: str) -> Optional[BasePlatform]:
        """
        Get a platform by name

        Args:
            platform_name: Name of the platform (case-insensitive)

        Returns:
            Platform instance or None if not found
        """
        return self._platforms.get(platform_name.lower())

    def get_metadata(self, platform_name: str) -> Optional[PlatformMetadata]:
        """
        Get platform metadata by name

        Args:
            platform_name: Name of the platform (case-insensitive)

        Returns:
            PlatformMetadata or None if not found
        """
        return self._metadata.get(platform_name.lower())

    def list_platforms(self) -> List[str]:
        """
        List all registered platform names

        Returns:
            List of platform names
        """
        return list(self._platforms.keys())

    def list_metadata(self) -> List[PlatformMetadata]:
        """
        List metadata for all registered platforms

        Returns:
            List of PlatformMetadata objects
        """
        return list(self._metadata.values())

    def is_registered(self, platform_name: str) -> bool:
        """
        Check if a platform is registered

        Args:
            platform_name: Name of the platform (case-insensitive)

        Returns:
            True if registered, False otherwise
        """
        return platform_name.lower() in self._platforms

    def unregister(self, platform_name: str) -> bool:
        """
        Unregister a platform

        Args:
            platform_name: Name of the platform to unregister

        Returns:
            True if unregistered, False if not found
        """
        platform_name = platform_name.lower()
        if platform_name in self._platforms:
            del self._platforms[platform_name]
            del self._metadata[platform_name]
            return True
        return False

    def clear(self) -> None:
        """Clear all registered platforms"""
        self._platforms.clear()
        self._metadata.clear()
