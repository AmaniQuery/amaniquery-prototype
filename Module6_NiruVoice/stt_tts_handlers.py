"""
Speech-to-Text and Text-to-Speech Handlers
"""

from typing import Optional, Dict
from loguru import logger


class STTHandler:
    """Speech-to-Text handler interface"""

    def __init__(self, provider: str, config: Dict):
        """
        Initialize STT handler

        Args:
            provider: STT provider (openai, assemblyai, kimi, nvidia_nim)
            config: Provider-specific configuration
        """
        self.provider = provider
        self.config = config
        self._client = None
        logger.info(f"STT handler initialized with provider: {provider}")

        if provider == "nvidia_nim":
            self._init_nim()

    def _init_nim(self):
        try:
            from .providers.nim_audio_provider import get_nim_provider

            self._client = get_nim_provider()
            logger.info("NVIDIA NIM STT client loaded")
        except Exception as e:
            logger.warning(f"Failed to load NVIDIA NIM STT: {e}")

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        """Transcribe audio file to text"""
        if self.provider == "nvidia_nim" and self._client:
            return self._client.transcribe(audio_path, language)
        logger.warning(f"STT transcribe not implemented for {self.provider}")
        return ""


class TTSHandler:
    """Text-to-Speech handler interface"""

    def __init__(self, provider: str, config: Dict):
        """
        Initialize TTS handler

        Args:
            provider: TTS provider (vibevoice, openai, silero, kimi, nvidia_nim)
            config: Provider-specific configuration
        """
        self.provider = provider
        self.config = config
        self._client = None
        logger.info(f"TTS handler initialized with provider: {provider}")

        if provider == "vibevoice":
            self._init_vibevoice()
        elif provider == "nvidia_nim":
            self._init_nim()

    def _init_vibevoice(self):
        try:
            from .vibevoice_tts import VibeVoiceTTS

            self._client = VibeVoiceTTS()
            logger.info("VibeVoice TTS client loaded")
        except ImportError:
            try:
                from vibevoice_tts import VibeVoiceTTS

                self._client = VibeVoiceTTS()
                logger.info("VibeVoice TTS client loaded")
            except Exception as e:
                logger.warning(f"Failed to load VibeVoice TTS: {e}")

    def _init_nim(self):
        try:
            from .providers.nim_audio_provider import get_nim_provider

            self._client = get_nim_provider()
            logger.info("NVIDIA NIM TTS client loaded")
        except Exception as e:
            logger.warning(f"Failed to load NVIDIA NIM TTS: {e}")

    def synthesize(
        self, text: str, output_path: str = "", voice: str = "", language: str = "en"
    ) -> str:
        """Synthesize text to audio file"""
        if self.provider == "vibevoice" and self._client:
            import asyncio

            try:
                audio_bytes = asyncio.run(
                    self._client.synthesize(text, voice=voice or None)
                )
                if output_path:
                    with open(output_path, "wb") as f:
                        f.write(audio_bytes)
                    return output_path
                return str(len(audio_bytes))
            except Exception as e:
                logger.error(f"VibeVoice synthesis failed: {e}")
                return ""
        elif self.provider == "nvidia_nim" and self._client:
            return self._client.synthesize(text, output_path, voice, language)
        logger.warning(f"TTS synthesize not implemented for {self.provider}")
        return ""


def create_stt_handler(provider: str, config: Dict) -> STTHandler:
    """Factory function to create STT handler"""
    return STTHandler(provider, config)


def create_tts_handler(provider: str, config: Dict) -> TTSHandler:
    """Factory function to create TTS handler"""
    return TTSHandler(provider, config)
