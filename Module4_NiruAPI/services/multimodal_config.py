"""
Multimodal Configuration - Feature flags and settings for vision/media processing

Environment variables:
- ENABLE_VISION_RAG: Enable image/PDF processing (default: true)
- ENABLE_VIDEO_RAG: Enable video processing (default: true)
- ENABLE_AUDIO_RAG: Enable audio transcription (default: true)
- MAX_IMAGE_SIZE_MB: Maximum image size in MB (default: 10)
- MAX_VIDEO_SIZE_MB: Maximum video size in MB (default: 100)
- MAX_AUDIO_SIZE_MB: Maximum audio size in MB (default: 50)
- MAX_PDF_SIZE_MB: Maximum PDF size in MB (default: 20)
- VIDEO_FRAME_COUNT: Number of frames to extract from videos (default: 10)
- MULTIMODAL_STORAGE_BACKEND: Storage backend ('database' or 'memory', default: database)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from loguru import logger


@dataclass
class MultimodalConfig:
    """Configuration for multimodal processing"""

    # Feature flags
    enable_vision_rag: bool = True
    enable_video_rag: bool = True
    enable_audio_rag: bool = True

    # File size limits (in MB)
    max_image_size_mb: int = 10
    max_video_size_mb: int = 100
    max_audio_size_mb: int = 50
    max_pdf_size_mb: int = 20

    # Video processing
    video_frame_count: int = 10
    video_frame_strategy: str = (
        "fixed_interval"  # fixed_interval, keyframes, scene_detection
    )
    video_extract_audio: bool = True

    # Audio processing
    audio_language: Optional[str] = None  # None for auto-detect
    audio_model: str = "whisper-1"  # OpenAI model or local model name
    audio_use_local_fallback: bool = True

    # Storage
    storage_backend: str = "database"  # database, memory
    asset_expiry_hours: int = 24
    cleanup_interval_hours: int = 1

    # OCR settings
    ocr_provider: str = "tesseract"  # tesseract, easyocr, paddleocr
    ocr_languages: List[str] = field(default_factory=lambda: ["en"])

    # Embedding settings
    embedding_model: str = "cohere-embed-4"
    embedding_batch_size: int = 10

    # Resilience
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: float = 60.0
    max_retries: int = 3
    retry_base_delay: float = 1.0

    # Performance
    max_concurrent_processing: int = 5
    processing_timeout_seconds: float = 300.0
    cache_embeddings: bool = True

    @classmethod
    def from_env(cls) -> "MultimodalConfig":
        """Load configuration from environment variables"""
        return cls(
            # Feature flags
            enable_vision_rag=os.getenv("ENABLE_VISION_RAG", "true").lower() == "true",
            enable_video_rag=os.getenv("ENABLE_VIDEO_RAG", "true").lower() == "true",
            enable_audio_rag=os.getenv("ENABLE_AUDIO_RAG", "true").lower() == "true",
            # File sizes
            max_image_size_mb=int(os.getenv("MAX_IMAGE_SIZE_MB", "10")),
            max_video_size_mb=int(os.getenv("MAX_VIDEO_SIZE_MB", "100")),
            max_audio_size_mb=int(os.getenv("MAX_AUDIO_SIZE_MB", "50")),
            max_pdf_size_mb=int(os.getenv("MAX_PDF_SIZE_MB", "20")),
            # Video
            video_frame_count=int(os.getenv("VIDEO_FRAME_COUNT", "10")),
            video_frame_strategy=os.getenv("VIDEO_FRAME_STRATEGY", "fixed_interval"),
            video_extract_audio=os.getenv("VIDEO_EXTRACT_AUDIO", "true").lower()
            == "true",
            # Audio
            audio_language=os.getenv("AUDIO_LANGUAGE"),
            audio_model=os.getenv("AUDIO_MODEL", "whisper-1"),
            audio_use_local_fallback=os.getenv(
                "AUDIO_USE_LOCAL_FALLBACK", "true"
            ).lower()
            == "true",
            # Storage
            storage_backend=os.getenv("MULTIMODAL_STORAGE_BACKEND", "database"),
            asset_expiry_hours=int(os.getenv("ASSET_EXPIRY_HOURS", "24")),
            cleanup_interval_hours=int(os.getenv("CLEANUP_INTERVAL_HOURS", "1")),
            # OCR
            ocr_provider=os.getenv("OCR_PROVIDER", "tesseract"),
            ocr_languages=os.getenv("OCR_LANGUAGES", "en").split(","),
            # Embedding
            embedding_model=os.getenv("EMBEDDING_MODEL", "cohere-embed-4"),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "10")),
            # Resilience
            enable_circuit_breaker=os.getenv("ENABLE_CIRCUIT_BREAKER", "true").lower()
            == "true",
            circuit_breaker_failure_threshold=int(
                os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")
            ),
            circuit_breaker_timeout_seconds=float(
                os.getenv("CIRCUIT_BREAKER_TIMEOUT_SECONDS", "60.0")
            ),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_base_delay=float(os.getenv("RETRY_BASE_DELAY", "1.0")),
            # Performance
            max_concurrent_processing=int(os.getenv("MAX_CONCURRENT_PROCESSING", "5")),
            processing_timeout_seconds=float(
                os.getenv("PROCESSING_TIMEOUT_SECONDS", "300.0")
            ),
            cache_embeddings=os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true",
        )

    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings"""
        warnings = []

        if self.max_image_size_mb > 50:
            warnings.append(
                f"Large image size limit ({self.max_image_size_mb}MB) may cause memory issues"
            )

        if self.max_video_size_mb > 500:
            warnings.append(
                f"Large video size limit ({self.max_video_size_mb}MB) may cause processing delays"
            )

        if self.video_frame_count > 30:
            warnings.append(
                f"High frame count ({self.video_frame_count}) may increase processing time"
            )

        if self.asset_expiry_hours < 1:
            warnings.append("Very short asset expiry may cause data loss")

        return warnings

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "enable_vision_rag": self.enable_vision_rag,
            "enable_video_rag": self.enable_video_rag,
            "enable_audio_rag": self.enable_audio_rag,
            "max_image_size_mb": self.max_image_size_mb,
            "max_video_size_mb": self.max_video_size_mb,
            "max_audio_size_mb": self.max_audio_size_mb,
            "max_pdf_size_mb": self.max_pdf_size_mb,
            "video_frame_count": self.video_frame_count,
            "storage_backend": self.storage_backend,
            "ocr_provider": self.ocr_provider,
            "embedding_model": self.embedding_model,
        }


# Global configuration instance
_config: Optional[MultimodalConfig] = None


def get_multimodal_config() -> MultimodalConfig:
    """Get global multimodal configuration"""
    global _config
    if _config is None:
        _config = MultimodalConfig.from_env()
        warnings = _config.validate()
        for warning in warnings:
            logger.warning(f"Multimodal config warning: {warning}")
        logger.info(
            f"Multimodal config loaded: vision={_config.enable_vision_rag}, video={_config.enable_video_rag}, audio={_config.enable_audio_rag}"
        )
    return _config


def reload_config() -> MultimodalConfig:
    """Reload configuration from environment"""
    global _config
    _config = MultimodalConfig.from_env()
    return _config


# =============================================================================
# SUPPORTED FILE TYPES
# =============================================================================

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf"}

MIME_TYPE_MAPPING = {
    # Images
    "image/jpeg": "image",
    "image/png": "image",
    "image/gif": "image",
    "image/webp": "image",
    "image/bmp": "image",
    "image/tiff": "image",
    # Videos
    "video/mp4": "video",
    "video/avi": "video",
    "video/quicktime": "video",
    "video/x-matroska": "video",
    "video/webm": "video",
    # Audio
    "audio/mpeg": "audio",
    "audio/wav": "audio",
    "audio/x-wav": "audio",
    "audio/mp4": "audio",
    "audio/flac": "audio",
    "audio/ogg": "audio",
    "audio/aac": "audio",
    # Documents
    "application/pdf": "pdf",
}


def get_media_type(filename: str, content_type: Optional[str] = None) -> str:
    """
    Determine media type from filename and content type

    Returns: 'image', 'video', 'audio', 'pdf', or 'unknown'
    """
    from pathlib import Path

    ext = Path(filename).suffix.lower()

    # Check by extension first
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    elif ext in SUPPORTED_VIDEO_EXTENSIONS:
        return "video"
    elif ext in SUPPORTED_AUDIO_EXTENSIONS:
        return "audio"
    elif ext in SUPPORTED_DOCUMENT_EXTENSIONS:
        return "pdf"

    # Fall back to content type
    if content_type:
        return MIME_TYPE_MAPPING.get(content_type, "unknown")

    return "unknown"


def is_media_type_enabled(
    media_type: str, config: Optional[MultimodalConfig] = None
) -> bool:
    """Check if a media type is enabled in configuration"""
    if config is None:
        config = get_multimodal_config()

    if media_type == "video":
        return config.enable_video_rag
    elif media_type == "audio":
        return config.enable_audio_rag
    elif media_type in ("image", "pdf"):
        return config.enable_vision_rag

    return False
