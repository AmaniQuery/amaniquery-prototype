"""
Offline Cache Manager for Voice Agent
Handles caching of audio responses and graceful degradation
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
from datetime import datetime, timedelta


class OfflineCache:
    """
    Offline cache for voice responses.
    Survives internet outages, Redis failures, and LLM API downtime.
    """

    def __init__(
        self,
        cache_dir: str = "public/audio",
        max_cache_size_mb: int = 100,
        ttl_days: int = 30,
    ):
        """
        Initialize offline cache.

        Args:
            cache_dir: Directory to store cached audio files
            max_cache_size_mb: Maximum cache size in megabytes
            ttl_days: Time-to-live for cached items in days
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        self.ttl = timedelta(days=ttl_days)

        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache metadata: {e}")
                return {}
        return {}

    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get_cached_audio(self, text: str) -> Optional[bytes]:
        """
        Get cached audio for given text.

        Args:
            text: Text to find cached audio for

        Returns:
            Audio bytes if cached, None otherwise
        """
        cache_key = self._get_cache_key(text)

        if cache_key not in self.metadata:
            return None

        entry = self.metadata[cache_key]

        # Check TTL
        cached_time = datetime.fromisoformat(entry["timestamp"])
        if datetime.now() - cached_time > self.ttl:
            logger.debug(f"Cache entry expired: {cache_key}")
            self.remove_entry(cache_key)
            return None

        # Load audio file
        audio_path = self.cache_dir / entry["filename"]
        if not audio_path.exists():
            logger.warning(f"Cached audio file not found: {audio_path}")
            self.remove_entry(cache_key)
            return None

        try:
            return audio_path.read_bytes()
        except Exception as e:
            logger.error(f"Failed to read cached audio: {e}")
            return None

    def cache_audio(
        self, text: str, audio_bytes: bytes, metadata: Optional[Dict] = None
    ):
        """
        Cache audio for given text.

        Args:
            text: Text associated with audio
            audio_bytes: Audio data to cache
            metadata: Optional metadata to store
        """
        cache_key = self._get_cache_key(text)
        filename = f"{cache_key}.mp3"
        audio_path = self.cache_dir / filename

        try:
            # Write audio file
            audio_path.write_bytes(audio_bytes)

            # Update metadata
            self.metadata[cache_key] = {
                "filename": filename,
                "text": text[:100],  # Store preview
                "timestamp": datetime.now().isoformat(),
                "size": len(audio_bytes),
                "metadata": metadata or {},
            }

            self._save_metadata()

            # Clean up if cache too large
            self._enforce_cache_limit()

            logger.debug(f"Cached audio for text: {text[:50]}...")

        except Exception as e:
            logger.error(f"Failed to cache audio: {e}")

    def remove_entry(self, cache_key: str):
        """Remove cache entry."""
        if cache_key in self.metadata:
            entry = self.metadata[cache_key]
            audio_path = self.cache_dir / entry["filename"]

            # Delete file
            if audio_path.exists():
                try:
                    audio_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete cached file: {e}")

            # Remove from metadata
            del self.metadata[cache_key]
            self._save_metadata()

    def _enforce_cache_limit(self):
        """Enforce cache size limit by removing oldest entries."""
        total_size = sum(entry["size"] for entry in self.metadata.values())

        if total_size <= self.max_cache_size:
            return

        logger.info(
            f"Cache size ({total_size / 1024 / 1024:.1f} MB) exceeds limit, cleaning up..."
        )

        # Sort by timestamp (oldest first)
        sorted_entries = sorted(self.metadata.items(), key=lambda x: x[1]["timestamp"])

        # Remove oldest entries until under limit
        for cache_key, entry in sorted_entries:
            if total_size <= self.max_cache_size:
                break

            self.remove_entry(cache_key)
            total_size -= entry["size"]

        logger.info(f"Cache cleaned. New size: {total_size / 1024 / 1024:.1f} MB")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry["size"] for entry in self.metadata.values())

        return {
            "total_entries": len(self.metadata),
            "total_size_mb": total_size / 1024 / 1024,
            "cache_dir": str(self.cache_dir),
            "max_size_mb": self.max_cache_size / 1024 / 1024,
            "usage_percent": (
                (total_size / self.max_cache_size) * 100
                if self.max_cache_size > 0
                else 0
            ),
        }

    def clear_cache(self):
        """Clear all cached data."""
        for cache_key in list(self.metadata.keys()):
            self.remove_entry(cache_key)

        logger.info("Cache cleared")


# Global cache instance
_cache_instance: Optional[OfflineCache] = None


def get_cache() -> OfflineCache:
    """Get or create global cache instance."""
    global _cache_instance

    if _cache_instance is None:
        cache_dir = os.getenv("AUDIO_CACHE_DIR", "public/audio")
        _cache_instance = OfflineCache(cache_dir=cache_dir)

    return _cache_instance
