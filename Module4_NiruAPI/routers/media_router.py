"""
Media Router - Unified multimodal media upload and processing endpoints

Provides:
- File type detection and routing (images, PDFs, audio, video)
- Unified upload endpoint with automatic processing
- Session-based media management
- Processing status tracking
"""

import os
import uuid
import tempfile
import mimetypes
from datetime import datetime
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    File,
    UploadFile,
    BackgroundTasks,
    Query,
)
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/media", tags=["Media"])


# =============================================================================
# CONFIGURATION
# =============================================================================

# Feature flags (loaded from environment)
ENABLE_VISION_RAG = os.getenv("ENABLE_VISION_RAG", "true").lower() == "true"
ENABLE_VIDEO_RAG = os.getenv("ENABLE_VIDEO_RAG", "true").lower() == "true"
ENABLE_AUDIO_RAG = os.getenv("ENABLE_AUDIO_RAG", "true").lower() == "true"

# File size limits
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE_MB", "10")) * 1024 * 1024
MAX_VIDEO_SIZE = int(os.getenv("MAX_VIDEO_SIZE_MB", "100")) * 1024 * 1024
MAX_AUDIO_SIZE = int(os.getenv("MAX_AUDIO_SIZE_MB", "50")) * 1024 * 1024
MAX_PDF_SIZE = int(os.getenv("MAX_PDF_SIZE_MB", "20")) * 1024 * 1024

# Supported file types
SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
SUPPORTED_VIDEO_TYPES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SUPPORTED_AUDIO_TYPES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
SUPPORTED_DOCUMENT_TYPES = {".pdf"}


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class MediaUploadResponse(BaseModel):
    """Media upload response"""

    id: str
    session_id: str
    file_type: str
    file_name: str
    file_size: int
    processing_status: str
    message: str


class MediaAssetResponse(BaseModel):
    """Media asset response"""

    id: str
    session_id: str
    file_type: str
    file_name: str
    file_size: Optional[int] = None
    cloudinary_url: Optional[str] = None
    processing_status: str
    has_embedding: bool = False
    has_transcription: bool = False
    created_at: str


class MediaSessionResponse(BaseModel):
    """Media session response"""

    id: str
    name: Optional[str] = None
    asset_count: int
    modalities: List[str]
    created_at: str


class ProcessVideoRequest(BaseModel):
    """Video processing options"""

    num_frames: int = 10
    extract_audio: bool = True
    transcribe_audio: bool = True


class TranscribeAudioRequest(BaseModel):
    """Audio transcription options"""

    language: Optional[str] = None
    context_prompt: Optional[str] = None


# =============================================================================
# STATE & DEPENDENCIES
# =============================================================================


class RouterState:
    """State container for router dependencies"""

    vision_rag_service = None
    multimodal_storage = None
    cloudinary_service = None


_state = RouterState()


def get_storage():
    """Get multimodal storage with lazy initialization"""
    if _state.multimodal_storage is None:
        logger.warning(
            "Multimodal storage not initialized, attempting lazy initialization"
        )
        try:
            from Module4_NiruAPI.services.multimodal_storage import (
                create_multimodal_storage,
            )

            _state.multimodal_storage = create_multimodal_storage(use_database=True)
            logger.info("Multimodal storage lazily initialized")
        except Exception as e:
            logger.error(f"Failed to initialize multimodal storage: {e}")
            # Fall back to in-memory
            from Module4_NiruAPI.services.multimodal_storage import (
                InMemoryMultimodalStorage,
            )

            _state.multimodal_storage = InMemoryMultimodalStorage()
    return _state.multimodal_storage


def get_vision_service():
    """Get vision RAG service with lazy initialization"""
    if _state.vision_rag_service is None:
        if not ENABLE_VISION_RAG:
            return None
        try:
            from Module4_NiruAPI.services.vision_rag import VisionRAGService

            _state.vision_rag_service = VisionRAGService(
                enable_video=ENABLE_VIDEO_RAG,
                enable_audio=ENABLE_AUDIO_RAG,
            )
            logger.info("Vision RAG service lazily initialized")
        except Exception as e:
            logger.warning(f"Vision RAG service unavailable: {e}")
            return None
    return _state.vision_rag_service


def get_current_user_id(request: Request) -> Optional[str]:
    """Get current user ID from auth context"""
    try:
        auth_context = getattr(request.state, "auth_context", None)
        if auth_context and auth_context.user_id:
            return auth_context.user_id
    except Exception:
        pass
    return None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def detect_file_type(filename: str, content_type: Optional[str] = None) -> str:
    """
    Detect file type from filename and content type

    Returns: 'image', 'video', 'audio', 'pdf', or 'unknown'
    """
    ext = Path(filename).suffix.lower()

    if ext in SUPPORTED_IMAGE_TYPES:
        return "image"
    elif ext in SUPPORTED_VIDEO_TYPES:
        return "video"
    elif ext in SUPPORTED_AUDIO_TYPES:
        return "audio"
    elif ext in SUPPORTED_DOCUMENT_TYPES:
        return "pdf"

    # Fall back to content type
    if content_type:
        if content_type.startswith("image/"):
            return "image"
        elif content_type.startswith("video/"):
            return "video"
        elif content_type.startswith("audio/"):
            return "audio"
        elif content_type == "application/pdf":
            return "pdf"

    return "unknown"


def get_max_size_for_type(file_type: str) -> int:
    """Get maximum file size for type"""
    return {
        "image": MAX_IMAGE_SIZE,
        "video": MAX_VIDEO_SIZE,
        "audio": MAX_AUDIO_SIZE,
        "pdf": MAX_PDF_SIZE,
    }.get(file_type, MAX_IMAGE_SIZE)


def is_type_enabled(file_type: str) -> bool:
    """Check if file type processing is enabled"""
    if file_type == "video":
        return ENABLE_VIDEO_RAG
    elif file_type == "audio":
        return ENABLE_AUDIO_RAG
    elif file_type in ("image", "pdf"):
        return ENABLE_VISION_RAG
    return False


async def save_upload_file(file: UploadFile, dest_dir: str) -> str:
    """Save uploaded file and return path"""
    os.makedirs(dest_dir, exist_ok=True)

    # Generate unique filename
    ext = Path(file.filename).suffix
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(dest_dir, unique_name)

    # Save file
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return file_path


# =============================================================================
# BACKGROUND TASKS
# =============================================================================


async def process_image_background(
    asset_id: str,
    file_path: str,
    session_id: str,
):
    """Background task to process and embed image"""
    storage = get_storage()
    vision_service = get_vision_service()

    try:
        if not vision_service:
            storage.update_asset(
                asset_id,
                processing_status="failed",
                processing_error="Vision service unavailable",
            )
            return

        # Generate embedding
        embedding = vision_service.vision_embedder.embed_image(file_path)

        # Update asset
        storage.update_asset(
            asset_id,
            embedding=embedding.tolist(),
            processing_status="completed",
        )

        logger.info(f"Image processed: {asset_id}")

    except Exception as e:
        logger.error(f"Error processing image {asset_id}: {e}")
        storage.update_asset(
            asset_id, processing_status="failed", processing_error=str(e)
        )


async def process_video_background(
    asset_id: str,
    file_path: str,
    session_id: str,
    num_frames: int = 10,
    transcribe: bool = True,
):
    """Background task to process video"""
    storage = get_storage()
    vision_service = get_vision_service()

    try:
        if not vision_service or not vision_service.video_processor:
            storage.update_asset(
                asset_id,
                processing_status="failed",
                processing_error="Video processing unavailable",
            )
            return

        # Process video
        result = vision_service.process_video(
            video_path=file_path,
            num_frames=num_frames,
            extract_audio=transcribe,
            transcribe_audio=transcribe,
        )

        if not result.get("success"):
            storage.update_asset(
                asset_id,
                processing_status="failed",
                processing_error=result.get("error", "Unknown error"),
            )
            return

        # Store video frames as child assets
        for frame in result.get("frames", []):
            storage.add_asset(
                session_id=session_id,
                file_type="video_frame",
                file_name=f"frame_{frame['frame_number']}.jpg",
                file_path=frame.get("frame_path"),
                embedding=frame.get("embedding"),
                timestamp=frame.get("timestamp"),
                frame_number=frame.get("frame_number"),
                parent_asset_id=asset_id,
                metadata=frame.get("metadata"),
            )

        # Update main video asset with transcription
        transcription = result.get("transcription")
        if transcription:
            # Generate embedding for transcript
            transcript_embedding = None
            try:
                transcript_embedding = vision_service.vision_embedder.embed_text(
                    transcription["text"]
                )
                transcript_embedding = transcript_embedding.tolist()
            except Exception:
                pass

            storage.update_asset(
                asset_id,
                extracted_text=transcription["text"],
                embedding=transcript_embedding,
                segments=transcription.get("segments"),
                language=transcription.get("language"),
                processing_status="completed",
                metadata_json={
                    "duration": result.get("video_metadata", {}).get("duration"),
                    "frame_count": len(result.get("frames", [])),
                    "has_transcription": True,
                },
            )
        else:
            storage.update_asset(
                asset_id,
                processing_status="completed",
                metadata_json={
                    "duration": result.get("video_metadata", {}).get("duration"),
                    "frame_count": len(result.get("frames", [])),
                    "has_transcription": False,
                },
            )

        logger.info(
            f"Video processed: {asset_id} ({len(result.get('frames', []))} frames)"
        )

    except Exception as e:
        logger.error(f"Error processing video {asset_id}: {e}")
        storage.update_asset(
            asset_id, processing_status="failed", processing_error=str(e)
        )


async def process_audio_background(
    asset_id: str,
    file_path: str,
    session_id: str,
    language: Optional[str] = None,
):
    """Background task to transcribe audio"""
    storage = get_storage()
    vision_service = get_vision_service()

    try:
        if not vision_service or not vision_service.whisper_provider:
            storage.update_asset(
                asset_id,
                processing_status="failed",
                processing_error="Audio transcription unavailable",
            )
            return

        # Transcribe and embed
        result = vision_service.embed_audio_transcript(file_path, language=language)

        if not result.get("success"):
            storage.update_asset(
                asset_id,
                processing_status="failed",
                processing_error=result.get("error", "Transcription failed"),
            )
            return

        # Update asset
        storage.update_asset(
            asset_id,
            extracted_text=result.get("text"),
            embedding=result.get("embedding"),
            segments=result.get("segment_embeddings"),
            language=result.get("language"),
            processing_status="completed",
            metadata_json={
                "duration": result.get("duration"),
                "confidence": result.get("confidence"),
            },
        )

        logger.info(f"Audio transcribed: {asset_id}")

    except Exception as e:
        logger.error(f"Error transcribing audio {asset_id}: {e}")
        storage.update_asset(
            asset_id, processing_status="failed", processing_error=str(e)
        )


async def process_pdf_background(
    asset_id: str,
    file_path: str,
    session_id: str,
):
    """Background task to process PDF"""
    storage = get_storage()
    vision_service = get_vision_service()

    try:
        if not vision_service:
            storage.update_asset(
                asset_id,
                processing_status="failed",
                processing_error="Vision service unavailable",
            )
            return

        # Extract pages and embed
        from Module4_NiruAPI.services.pdf_page_extractor import PDFPageExtractor

        extractor = PDFPageExtractor()
        pages = extractor.extract_pages(file_path)

        # Process each page
        page_count = 0
        for page in pages:
            try:
                # Embed page image
                embedding = vision_service.vision_embedder.embed_image(
                    page["image_path"]
                )

                storage.add_asset(
                    session_id=session_id,
                    file_type="image",
                    file_name=f"page_{page['page_number']}.png",
                    file_path=page["image_path"],
                    embedding=embedding.tolist(),
                    parent_asset_id=asset_id,
                    metadata={
                        "page_number": page["page_number"],
                        "source_file": file_path,
                        "source_type": "pdf_page",
                    },
                )
                page_count += 1
            except Exception as e:
                logger.warning(f"Error processing PDF page {page['page_number']}: {e}")

        storage.update_asset(
            asset_id,
            processing_status="completed",
            metadata_json={"page_count": page_count},
        )

        logger.info(f"PDF processed: {asset_id} ({page_count} pages)")

    except Exception as e:
        logger.error(f"Error processing PDF {asset_id}: {e}")
        storage.update_asset(
            asset_id, processing_status="failed", processing_error=str(e)
        )


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/sessions", response_model=MediaSessionResponse)
async def create_media_session(
    request: Request,
    name: Optional[str] = None,
):
    """Create a new media session"""
    storage = get_storage()
    user_id = get_current_user_id(request)

    try:
        session = storage.create_session(user_id=user_id, name=name)
        return MediaSessionResponse(
            id=session["id"],
            name=session.get("name"),
            asset_count=session.get("asset_count", 0),
            modalities=session.get("modalities", []),
            created_at=session.get("created_at", datetime.utcnow().isoformat()),
        )
    except Exception as e:
        logger.error(f"Error creating media session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=MediaSessionResponse)
async def get_media_session(session_id: str, request: Request):
    """Get media session details"""
    storage = get_storage()

    session = storage.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return MediaSessionResponse(
        id=session["id"],
        name=session.get("name"),
        asset_count=session.get("asset_count", 0),
        modalities=session.get("modalities", []),
        created_at=session.get("created_at", datetime.utcnow().isoformat()),
    )


@router.delete("/sessions/{session_id}")
async def delete_media_session(session_id: str, request: Request):
    """Delete media session and all assets"""
    storage = get_storage()

    success = storage.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"message": "Session deleted successfully"}


@router.get("/sessions/{session_id}/assets", response_model=List[MediaAssetResponse])
async def list_session_assets(
    session_id: str,
    request: Request,
    file_type: Optional[str] = None,
):
    """List all assets in a session"""
    storage = get_storage()

    assets = storage.get_session_assets(session_id, file_type=file_type)

    return [
        MediaAssetResponse(
            id=a["id"],
            session_id=a["session_id"],
            file_type=a["file_type"],
            file_name=a["file_name"],
            file_size=a.get("file_size"),
            cloudinary_url=a.get("cloudinary_url"),
            processing_status=a.get("processing_status", "unknown"),
            has_embedding=a.get("embedding") is not None,
            has_transcription=a.get("extracted_text") is not None,
            created_at=a.get("created_at", datetime.utcnow().isoformat()),
        )
        for a in assets
    ]


@router.post("/upload", response_model=MediaUploadResponse)
async def upload_media(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: Optional[str] = Query(None, description="Session ID to add media to"),
):
    """
    Upload and process media file (image, PDF, audio, or video)

    The file type is auto-detected and routed to the appropriate processor.
    Processing happens in the background - check asset status for completion.
    """
    storage = get_storage()
    user_id = get_current_user_id(request)

    # Detect file type
    file_type = detect_file_type(file.filename, file.content_type)

    if file_type == "unknown":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: images, PDFs, audio, video",
        )

    # Check if type is enabled
    if not is_type_enabled(file_type):
        raise HTTPException(
            status_code=400, detail=f"{file_type.upper()} processing is disabled"
        )

    # Check file size
    content = await file.read()
    file_size = len(content)
    max_size = get_max_size_for_type(file_type)

    if file_size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size for {file_type}: {max_size // (1024*1024)}MB",
        )

    # Create session if needed
    if not session_id:
        session = storage.create_session(user_id=user_id)
        session_id = session["id"]

    # Save file
    upload_dir = os.path.join(tempfile.gettempdir(), "amaniquery_uploads", session_id)
    os.makedirs(upload_dir, exist_ok=True)

    ext = Path(file.filename).suffix
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(upload_dir, unique_name)

    with open(file_path, "wb") as f:
        f.write(content)

    # Create asset record
    asset = storage.add_asset(
        session_id=session_id,
        file_type=file_type,
        file_name=file.filename,
        file_path=file_path,
        file_size=file_size,
        mime_type=file.content_type,
        user_id=user_id,
        processing_status="processing",
    )

    asset_id = asset["id"]

    # Schedule background processing
    if file_type == "image":
        background_tasks.add_task(
            process_image_background, asset_id, file_path, session_id
        )
    elif file_type == "video":
        background_tasks.add_task(
            process_video_background, asset_id, file_path, session_id
        )
    elif file_type == "audio":
        background_tasks.add_task(
            process_audio_background, asset_id, file_path, session_id
        )
    elif file_type == "pdf":
        background_tasks.add_task(
            process_pdf_background, asset_id, file_path, session_id
        )

    return MediaUploadResponse(
        id=asset_id,
        session_id=session_id,
        file_type=file_type,
        file_name=file.filename,
        file_size=file_size,
        processing_status="processing",
        message=f"{file_type.capitalize()} upload successful. Processing in background.",
    )


@router.get("/assets/{asset_id}", response_model=MediaAssetResponse)
async def get_asset(asset_id: str, request: Request):
    """Get asset details and processing status"""
    storage = get_storage()

    asset = storage.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    return MediaAssetResponse(
        id=asset["id"],
        session_id=asset["session_id"],
        file_type=asset["file_type"],
        file_name=asset["file_name"],
        file_size=asset.get("file_size"),
        cloudinary_url=asset.get("cloudinary_url"),
        processing_status=asset.get("processing_status", "unknown"),
        has_embedding=asset.get("embedding") is not None,
        has_transcription=asset.get("extracted_text") is not None,
        created_at=asset.get("created_at", datetime.utcnow().isoformat()),
    )


@router.delete("/assets/{asset_id}")
async def delete_asset(asset_id: str, request: Request):
    """Delete an asset"""
    storage = get_storage()

    success = storage.delete_asset(asset_id)
    if not success:
        raise HTTPException(status_code=404, detail="Asset not found")

    return {"message": "Asset deleted successfully"}


@router.post("/assets/{asset_id}/reprocess")
async def reprocess_asset(
    asset_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Reprocess an asset (e.g., after a failed processing)"""
    storage = get_storage()

    asset = storage.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    file_path = asset.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="Source file no longer available")

    # Reset status
    storage.update_asset(
        asset_id, processing_status="processing", processing_error=None
    )

    # Schedule reprocessing
    file_type = asset["file_type"]
    session_id = asset["session_id"]

    if file_type == "image":
        background_tasks.add_task(
            process_image_background, asset_id, file_path, session_id
        )
    elif file_type == "video":
        background_tasks.add_task(
            process_video_background, asset_id, file_path, session_id
        )
    elif file_type == "audio":
        background_tasks.add_task(
            process_audio_background, asset_id, file_path, session_id
        )
    elif file_type == "pdf":
        background_tasks.add_task(
            process_pdf_background, asset_id, file_path, session_id
        )

    return {"message": "Reprocessing started"}


@router.get("/health")
async def media_health():
    """Check media processing health status"""
    vision_service = get_vision_service()

    status = {
        "vision_rag_enabled": ENABLE_VISION_RAG,
        "video_rag_enabled": ENABLE_VIDEO_RAG,
        "audio_rag_enabled": ENABLE_AUDIO_RAG,
        "vision_service_available": vision_service is not None,
        "video_processor_available": (
            vision_service.video_processor is not None if vision_service else False
        ),
        "audio_transcriber_available": (
            vision_service.whisper_provider is not None if vision_service else False
        ),
    }

    all_healthy = all(
        [
            status["vision_service_available"] or not ENABLE_VISION_RAG,
            status["video_processor_available"] or not ENABLE_VIDEO_RAG,
            status["audio_transcriber_available"] or not ENABLE_AUDIO_RAG,
        ]
    )

    return {
        "status": "healthy" if all_healthy else "degraded",
        "components": status,
    }


@router.get("/stats")
async def get_storage_stats(request: Request):
    """Get storage statistics (admin only)"""
    storage = get_storage()

    try:
        stats = storage.get_storage_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting storage stats: {e}")
        return {"error": str(e)}


@router.post("/cleanup")
async def trigger_cleanup(request: Request):
    """Trigger cleanup of expired assets (admin only)"""
    storage = get_storage()

    try:
        deleted = storage.cleanup_expired()
        return {"message": f"Cleaned up {deleted} expired items"}
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))
