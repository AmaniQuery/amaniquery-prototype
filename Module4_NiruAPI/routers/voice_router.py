"""
Voice Router - Simplified REST API for Voice Agent with VibeVoice TTS

Provides endpoints for text-to-speech and full voice conversations.
Uses browser's Web Speech API for transcription (client-side).
"""

import os
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
import time

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from loguru import logger

router = APIRouter(prefix="/api/v1/voice", tags=["voice"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "Wayne"
    cfg_scale: Optional[float] = 1.5


class TTSResponse(BaseModel):
    audio_url: str
    duration_ms: float
    text_length: int
    voice: str


class VoiceChatRequest(BaseModel):
    """Request for voice chat - text input (transcription done client-side)"""

    text: str
    category: str = "Kenyan Law"
    voice: Optional[str] = "Wayne"


class VoiceChatResponse(BaseModel):
    """Response with answer text and audio"""

    answer: str
    sources: List[Dict[str, Any]]
    audio_url: str
    duration_ms: float


class VoiceInfo(BaseModel):
    name: str
    language: str = "en"


class HealthResponse(BaseModel):
    status: str
    tts_provider: str
    voices: List[str]
    device: str


# =============================================================================
# MODULE STATE
# =============================================================================


class _State:
    """Global state for voice module"""

    tts = None
    rag_integration = None
    start_time = time.time()
    audio_files: Dict[str, float] = {}  # filename -> creation_time


_state = _State()


def get_tts():
    """Get or create TTS instance"""
    if _state.tts is None:
        try:
            from Module6_NiruVoice.vibevoice_tts import VibeVoiceTTS

            _state.tts = VibeVoiceTTS()
            logger.info("VibeVoice TTS initialized")
        except Exception as e:
            logger.error(f"Failed to initialize VibeVoice TTS: {e}")
            raise HTTPException(status_code=500, detail=f"TTS not available: {e}")
    return _state.tts


def get_rag_integration():
    """Get or create RAG integration"""
    if _state.rag_integration is None:
        try:
            from Module6_NiruVoice.rag_integration import VoiceRAGIntegration

            _state.rag_integration = VoiceRAGIntegration()
            logger.info("RAG integration initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RAG integration: {e}")
            raise HTTPException(status_code=500, detail="RAG integration not available")
    return _state.rag_integration


def cleanup_old_audio_files():
    """Clean up audio files older than 5 minutes"""
    now = time.time()
    to_remove = []

    for filename, created_time in _state.audio_files.items():
        if now - created_time > 300:  # 5 minutes
            try:
                Path(filename).unlink(missing_ok=True)
                to_remove.append(filename)
            except Exception:
                pass

    for f in to_remove:
        _state.audio_files.pop(f, None)


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/speak", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    Convert text to speech using VibeVoice

    Returns audio URL that can be fetched separately.
    """
    start_time = time.time()

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    if len(request.text) > 10000:
        raise HTTPException(status_code=400, detail="Text too long (max 10000 chars)")

    try:
        tts = get_tts()

        # Generate audio
        audio_bytes = await tts.synthesize(
            text=request.text,
            voice=request.voice,
            cfg_scale=request.cfg_scale,
        )

        # Save to temp file
        filename = f"tts_{int(time.time() * 1000)}.wav"
        filepath = Path(tempfile.gettempdir()) / filename
        filepath.write_bytes(audio_bytes)

        # Track for cleanup
        _state.audio_files[str(filepath)] = time.time()
        background_tasks.add_task(cleanup_old_audio_files)

        duration_ms = (time.time() - start_time) * 1000

        return TTSResponse(
            audio_url=f"/api/v1/voice/audio/{filename}",
            duration_ms=duration_ms,
            text_length=len(request.text),
            voice=request.voice or "Wayne",
        )

    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")


@router.post("/speak/stream")
async def text_to_speech_stream(request: TTSRequest):
    """
    Convert text to speech and return audio directly (streaming response)
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        tts = get_tts()
        audio_bytes = await tts.synthesize(
            text=request.text,
            voice=request.voice,
            cfg_scale=request.cfg_scale,
        )

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="speech.wav"'},
        )

    except Exception as e:
        logger.error(f"TTS stream failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")


@router.post("/chat", response_model=VoiceChatResponse)
async def voice_chat(request: VoiceChatRequest, background_tasks: BackgroundTasks):
    """
    Full voice conversation pipeline:
    1. Receive transcribed text (client does STT via Web Speech API)
    2. Query RAG for answer
    3. Generate speech response (TTS)

    Returns answer text and audio URL.
    """
    start_time = time.time()

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        # Step 1: Query RAG
        logger.info(f"[Voice Chat] Query: {request.text[:50]}...")
        rag = get_rag_integration()

        rag_response = None
        for attempt in range(3):
            try:
                rag_response = rag.query(request.text)
                break
            except Exception as e:
                logger.warning(f"[Voice Chat] RAG attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    rag_response = {
                        "text": "I apologize, but I'm having trouble processing your query right now. Please try again.",
                        "sources": [],
                    }
                else:
                    await asyncio.sleep(1)

        answer = rag_response.get("text", "No answer generated.")
        sources = rag_response.get("sources", [])

        logger.info(f"[Voice Chat] Answer: {len(answer)} chars")

        # Step 2: Generate TTS
        tts = get_tts()
        audio_bytes = await tts.synthesize(
            text=answer,
            voice=request.voice,
        )

        # Save audio
        filename = f"chat_{int(time.time() * 1000)}.wav"
        filepath = Path(tempfile.gettempdir()) / filename
        filepath.write_bytes(audio_bytes)

        _state.audio_files[str(filepath)] = time.time()
        background_tasks.add_task(cleanup_old_audio_files)

        duration_ms = (time.time() - start_time) * 1000

        return VoiceChatResponse(
            answer=answer,
            sources=sources,
            audio_url=f"/api/v1/voice/audio/{filename}",
            duration_ms=duration_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Voice Chat] Failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice chat error: {str(e)}")


@router.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve generated audio files"""
    # Check temp directory
    filepath = Path(tempfile.gettempdir()) / filename

    if not filepath.exists():
        # Also check current directory (legacy)
        filepath = Path(filename)
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(filepath, media_type="audio/wav", filename=filename)


@router.get("/voices", response_model=List[VoiceInfo])
async def list_voices():
    """List available voice presets"""
    try:
        tts = get_tts()
        voices = tts.get_available_voices()

        return [VoiceInfo(name=v, language="en") for v in voices]
    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        # Return default voice on error
        return [VoiceInfo(name="Wayne", language="en")]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Voice module health check"""
    try:
        tts = get_tts()
        health = tts.health_check()

        return HealthResponse(
            status=health.get("status", "unknown"),
            tts_provider="vibevoice",
            voices=health.get("voices", []),
            device=health.get("device", "unknown"),
        )
    except Exception as e:
        return HealthResponse(
            status="error",
            tts_provider="vibevoice",
            voices=[],
            device="unknown",
        )
