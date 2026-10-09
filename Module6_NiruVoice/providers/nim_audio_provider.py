"""
NVIDIA NIM Audio Provider for Voice Agent
Provides STT (speech-to-text) and TTS (text-to-speech) via NVIDIA NIM's OpenAI-compatible API
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
from loguru import logger


class NimAudioProvider:
    """
    Provider for NVIDIA NIM audio models via OpenAI-compatible API

    Uses:
    - STT: /v1/audio/transcriptions (Whisper or NIM-hosted ASR models)
    - TTS: /v1/audio/speech (NIM-hosted TTS models)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        stt_model: str = "whisper-large-v3-turbo",
        tts_model: str = "nvidia/parakeet-tts-1.1b",
        tts_voice: str = "default",
    ):
        self.api_key = api_key or os.getenv("NVIDIA_NIM_API_KEY", "")
        self.base_url = base_url or os.getenv(
            "NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
        self.stt_model = stt_model or os.getenv(
            "NIM_STT_MODEL", "whisper-large-v3-turbo"
        )
        self.tts_model = tts_model or os.getenv(
            "NIM_TTS_MODEL", "nvidia/parakeet-tts-1.1b"
        )
        self.tts_voice = tts_voice or os.getenv("NIM_TTS_VOICE", "default")
        self._client = None
        self.is_loaded = False

        logger.info(
            f"NIM provider initialized: base_url={self.base_url}, "
            f"stt={self.stt_model}, tts={self.tts_model}"
        )

    def _get_client(self):
        """Lazy-load OpenAI client pointed at NIM endpoint"""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                self.is_loaded = True
                logger.info(f"NIM OpenAI client initialized: {self.base_url}")
            except Exception as e:
                logger.error(f"Failed to initialize NIM OpenAI client: {e}")
                raise
        return self._client

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        """
        Transcribe audio to text using NIM STT

        Args:
            audio_path: Path to audio file
            language: Language code

        Returns:
            Transcribed text
        """
        try:
            client = self._get_client()
            logger.info(f"[NIM STT] Transcribing {audio_path}...")

            with open(audio_path, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model=self.stt_model,
                    file=f,
                    language=language,
                    response_format="text",
                )

            text = transcript if isinstance(transcript, str) else transcript.text
            logger.info(f"[NIM STT] Success: {text[:50]}...")
            return text

        except Exception as e:
            logger.error(f"[NIM STT] Failed: {e}")
            raise

    def synthesize(
        self,
        text: str,
        output_path: str,
        voice: str = "",
        language: str = "en",
    ) -> str:
        """
        Synthesize speech from text using NIM TTS

        Args:
            text: Text to convert to speech
            output_path: Path to save audio file
            voice: Voice selection
            language: Language code

        Returns:
            Path to generated audio file
        """
        try:
            client = self._get_client()
            voice = voice or self.tts_voice
            logger.info(f"[NIM TTS] Generating speech for: {text[:50]}...")

            response = client.audio.speech.create(
                model=self.tts_model,
                voice=voice,
                input=text,
                response_format="wav",
            )

            response.stream_to_file(output_path)
            logger.info(f"[NIM TTS] Success: saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"[NIM TTS] Failed: {e}")
            raise

    async def transcribe_async(self, audio_bytes: bytes, language: str = "en") -> str:
        """
        Async transcribe audio bytes using NIM STT

        Args:
            audio_bytes: Raw audio bytes
            language: Language code

        Returns:
            Transcribed text
        """
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            return self.transcribe(tmp_path, language)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    async def synthesize_async(
        self,
        text: str,
        voice: str = "",
        language: str = "en",
    ) -> bytes:
        """
        Async synthesize speech to audio bytes

        Args:
            text: Text to convert to speech
            voice: Voice selection
            language: Language code

        Returns:
            WAV audio bytes
        """
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self.synthesize(text, tmp_path, voice, language)
            return Path(tmp_path).read_bytes()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def health_check(self) -> Dict[str, Any]:
        """Check provider health"""
        try:
            self._get_client()
            return {
                "provider": "nvidia_nim",
                "status": "healthy",
                "is_loaded": self.is_loaded,
                "base_url": self.base_url,
                "stt_model": self.stt_model,
                "tts_model": self.tts_model,
            }
        except Exception as e:
            return {
                "provider": "nvidia_nim",
                "status": "unhealthy",
                "error": str(e),
                "is_loaded": False,
            }


_nim_provider: Optional[NimAudioProvider] = None


def get_nim_provider(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    stt_model: Optional[str] = None,
    tts_model: Optional[str] = None,
    tts_voice: Optional[str] = None,
) -> NimAudioProvider:
    """Get or create NIM provider instance (singleton)"""
    global _nim_provider
    if _nim_provider is None:
        _nim_provider = NimAudioProvider(
            api_key=api_key,
            base_url=base_url,
            stt_model=stt_model,
            tts_model=tts_model,
            tts_voice=tts_voice,
        )
    return _nim_provider
