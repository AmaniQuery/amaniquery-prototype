"""
Whisper Provider - Audio transcription using OpenAI Whisper

Provides speech-to-text transcription with support for:
- OpenAI Whisper API (cloud)
- Local Whisper model (offline fallback)
- Multiple languages with auto-detection
"""

import os
import asyncio
from typing import Dict, Optional, Union, List, Generator
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger

from Module6_NiruVoice.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
)
from Module6_NiruVoice.resilience.retry_handler import RetryHandler, RetryConfig


@dataclass
class TranscriptionResult:
    """Result of audio transcription"""

    text: str  # Transcribed text
    language: str  # Detected or specified language
    duration: float  # Audio duration in seconds
    confidence: float  # Confidence score (0-1)
    segments: List[Dict] = field(default_factory=list)  # Word/segment timestamps
    model_used: str = "whisper-1"

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "language": self.language,
            "duration": self.duration,
            "confidence": self.confidence,
            "segments": self.segments,
            "model_used": self.model_used,
        }


@dataclass
class WhisperConfig:
    """Configuration for Whisper provider"""

    # API configuration
    api_key: Optional[str] = None
    model: str = "whisper-1"  # OpenAI model name
    language: Optional[str] = None  # Language code (None for auto-detect)

    # Local model configuration
    local_model: str = "base"  # tiny, base, small, medium, large
    device: str = "auto"  # cpu, cuda, auto
    compute_type: str = "float16"  # float16, int8

    # Processing options
    use_local_fallback: bool = True
    response_format: str = "verbose_json"  # json, text, srt, verbose_json, vtt
    temperature: float = 0.0

    # Resilience
    timeout: float = 120.0
    max_retries: int = 3
    enable_circuit_breaker: bool = True


class WhisperProvider:
    """
    Whisper-based audio transcription provider

    Supports both OpenAI's Whisper API and local Whisper model execution
    with automatic failover between them.
    """

    def __init__(self, config: Optional[WhisperConfig] = None):
        """
        Initialize Whisper provider

        Args:
            config: Whisper configuration
        """
        self.config = config or WhisperConfig()

        # Get API key from config or environment
        self.api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")

        # Initialize OpenAI client
        self.openai_client = None
        if self.api_key:
            try:
                from openai import OpenAI, AsyncOpenAI

                self.openai_client = OpenAI(api_key=self.api_key)
                self.async_openai_client = AsyncOpenAI(api_key=self.api_key)
                logger.info("OpenAI Whisper API initialized")
            except ImportError:
                logger.warning("openai package not available, using local Whisper only")

        # Initialize local Whisper (lazy loading)
        self._local_model = None
        self._local_available = None

        # Initialize resilience components
        if self.config.enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                name="whisper_api",
                config=CircuitBreakerConfig(
                    failure_threshold=3,
                    timeout=60.0,
                ),
            )
        else:
            self.circuit_breaker = None

        self.retry_handler = RetryHandler(
            config=RetryConfig(
                max_retries=self.config.max_retries,
                base_delay=1.0,
                max_delay=30.0,
            )
        )

        logger.info(
            f"Whisper provider initialized "
            f"(API: {'available' if self.api_key else 'not configured'}, "
            f"local fallback: {self.config.use_local_fallback})"
        )

    @property
    def local_available(self) -> bool:
        """Check if local Whisper is available"""
        if self._local_available is None:
            try:
                import whisper

                self._local_available = True
            except ImportError:
                self._local_available = False
                logger.debug("Local Whisper not available (pip install openai-whisper)")
        return self._local_available

    def _get_local_model(self):
        """Lazy load local Whisper model"""
        if self._local_model is None and self.local_available:
            import whisper
            import torch

            device = self.config.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(
                f"Loading local Whisper model: {self.config.local_model} on {device}"
            )
            self._local_model = whisper.load_model(
                self.config.local_model,
                device=device,
            )
            logger.info("Local Whisper model loaded")

        return self._local_model

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio file

        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detect)
            prompt: Optional context prompt

        Returns:
            TranscriptionResult with transcribed text
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        language = language or self.config.language

        # Try OpenAI API first
        if self.openai_client:
            try:
                return self._transcribe_api(audio_path, language, prompt)
            except Exception as e:
                logger.warning(f"OpenAI API transcription failed: {e}")
                if self.config.use_local_fallback and self.local_available:
                    logger.info("Falling back to local Whisper")
                else:
                    raise

        # Fall back to local Whisper
        if self.local_available:
            return self._transcribe_local(audio_path, language, prompt)

        raise RuntimeError(
            "No transcription backend available. "
            "Set OPENAI_API_KEY or install openai-whisper package."
        )

    async def transcribe_async(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio file asynchronously

        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detect)
            prompt: Optional context prompt

        Returns:
            TranscriptionResult with transcribed text
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        language = language or self.config.language

        # Try OpenAI API first
        if self.async_openai_client:
            try:
                return await self._transcribe_api_async(audio_path, language, prompt)
            except Exception as e:
                logger.warning(f"OpenAI API transcription failed: {e}")
                if self.config.use_local_fallback and self.local_available:
                    logger.info("Falling back to local Whisper")
                else:
                    raise

        # Fall back to local Whisper (run in executor)
        if self.local_available:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self._transcribe_local(audio_path, language, prompt)
            )

        raise RuntimeError(
            "No transcription backend available. "
            "Set OPENAI_API_KEY or install openai-whisper package."
        )

    def _transcribe_api(
        self,
        audio_path: Path,
        language: Optional[str],
        prompt: Optional[str],
    ) -> TranscriptionResult:
        """Transcribe using OpenAI API"""
        logger.info(f"Transcribing via OpenAI API: {audio_path.name}")

        with open(audio_path, "rb") as audio_file:
            kwargs = {
                "model": self.config.model,
                "file": audio_file,
                "response_format": self.config.response_format,
                "temperature": self.config.temperature,
            }

            if language:
                kwargs["language"] = language
            if prompt:
                kwargs["prompt"] = prompt

            response = self.openai_client.audio.transcriptions.create(**kwargs)

        # Parse response based on format
        if self.config.response_format == "verbose_json":
            text = response.text
            detected_language = getattr(response, "language", language or "unknown")
            duration = getattr(response, "duration", 0.0)
            segments = getattr(response, "segments", [])

            # Convert segments to dict format
            segment_dicts = []
            for seg in segments:
                segment_dicts.append(
                    {
                        "start": getattr(seg, "start", 0),
                        "end": getattr(seg, "end", 0),
                        "text": getattr(seg, "text", ""),
                    }
                )
        else:
            text = response.text if hasattr(response, "text") else str(response)
            detected_language = language or "unknown"
            duration = 0.0
            segment_dicts = []

        logger.info(
            f"Transcription complete: {len(text)} chars, language: {detected_language}"
        )

        return TranscriptionResult(
            text=text,
            language=detected_language,
            duration=duration,
            confidence=0.95,  # API doesn't provide confidence, assume high
            segments=segment_dicts,
            model_used=self.config.model,
        )

    async def _transcribe_api_async(
        self,
        audio_path: Path,
        language: Optional[str],
        prompt: Optional[str],
    ) -> TranscriptionResult:
        """Transcribe using OpenAI API (async)"""
        logger.info(f"Transcribing via OpenAI API (async): {audio_path.name}")

        with open(audio_path, "rb") as audio_file:
            kwargs = {
                "model": self.config.model,
                "file": audio_file,
                "response_format": self.config.response_format,
                "temperature": self.config.temperature,
            }

            if language:
                kwargs["language"] = language
            if prompt:
                kwargs["prompt"] = prompt

            response = await self.async_openai_client.audio.transcriptions.create(
                **kwargs
            )

        # Parse response
        if self.config.response_format == "verbose_json":
            text = response.text
            detected_language = getattr(response, "language", language or "unknown")
            duration = getattr(response, "duration", 0.0)
            segments = getattr(response, "segments", [])

            segment_dicts = []
            for seg in segments:
                segment_dicts.append(
                    {
                        "start": getattr(seg, "start", 0),
                        "end": getattr(seg, "end", 0),
                        "text": getattr(seg, "text", ""),
                    }
                )
        else:
            text = response.text if hasattr(response, "text") else str(response)
            detected_language = language or "unknown"
            duration = 0.0
            segment_dicts = []

        logger.info(
            f"Transcription complete: {len(text)} chars, language: {detected_language}"
        )

        return TranscriptionResult(
            text=text,
            language=detected_language,
            duration=duration,
            confidence=0.95,
            segments=segment_dicts,
            model_used=self.config.model,
        )

    def _transcribe_local(
        self,
        audio_path: Path,
        language: Optional[str],
        prompt: Optional[str],
    ) -> TranscriptionResult:
        """Transcribe using local Whisper model"""
        logger.info(f"Transcribing locally: {audio_path.name}")

        model = self._get_local_model()
        if model is None:
            raise RuntimeError("Local Whisper model not available")

        # Transcribe
        options = {
            "verbose": False,
            "temperature": self.config.temperature,
        }

        if language:
            options["language"] = language
        if prompt:
            options["initial_prompt"] = prompt

        result = model.transcribe(str(audio_path), **options)

        # Extract segments
        segment_dicts = []
        for seg in result.get("segments", []):
            segment_dicts.append(
                {
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", ""),
                }
            )

        # Calculate average confidence from segments
        confidences = [
            seg.get("avg_logprob", -0.5) for seg in result.get("segments", [])
        ]
        if confidences:
            # Convert log probability to confidence (rough approximation)
            avg_logprob = sum(confidences) / len(confidences)
            confidence = min(1.0, max(0.0, 1.0 + avg_logprob))
        else:
            confidence = 0.8

        detected_language = result.get("language", language or "unknown")

        # Calculate duration from last segment
        duration = 0.0
        if segment_dicts:
            duration = segment_dicts[-1].get("end", 0)

        logger.info(
            f"Local transcription complete: {len(result['text'])} chars, "
            f"language: {detected_language}, confidence: {confidence:.2f}"
        )

        return TranscriptionResult(
            text=result["text"],
            language=detected_language,
            duration=duration,
            confidence=confidence,
            segments=segment_dicts,
            model_used=f"whisper-local-{self.config.local_model}",
        )

    def transcribe_stream(
        self,
        audio_chunks: Generator[bytes, None, None],
        language: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Stream transcription for real-time audio

        NOTE: This is a placeholder for future streaming support.
        Currently accumulates chunks and transcribes at end.

        Args:
            audio_chunks: Generator yielding audio bytes
            language: Language code

        Yields:
            Transcribed text chunks
        """
        import tempfile
        import wave

        # Accumulate audio data
        audio_data = b"".join(audio_chunks)

        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            # Assume 16kHz mono audio
            with wave.open(f, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(audio_data)

        try:
            result = self.transcribe(temp_path, language=language)
            yield result.text
        finally:
            os.unlink(temp_path)


# Factory function
def create_whisper_provider(
    api_key: Optional[str] = None,
    use_local: bool = True,
    local_model: str = "base",
) -> WhisperProvider:
    """
    Create a Whisper provider with sensible defaults

    Args:
        api_key: OpenAI API key (or from OPENAI_API_KEY env)
        use_local: Enable local Whisper fallback
        local_model: Local model size

    Returns:
        Configured WhisperProvider
    """
    config = WhisperConfig(
        api_key=api_key,
        use_local_fallback=use_local,
        local_model=local_model,
    )
    return WhisperProvider(config)
