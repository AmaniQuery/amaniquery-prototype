"""
Kimi-Audio-7B-Instruct Provider for Voice Agent
Supports both ASR (speech-to-text) and TTS (text-to-speech)
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
from loguru import logger


class KimiAudioProvider:
    """
    Provider for Kimi-Audio-7B-Instruct model

    Supports:
    - ASR: Convert speech to text
    - TTS: Convert text to speech
    """

    def __init__(
        self,
        model_path: str = "moonshotai/Kimi-Audio-7B-Instruct",
        device: str = "cpu",
        load_in_4bit: bool = True,
    ):
        """
        Initialize Kimi provider

        Args:
            model_path: Path to Kimi model
            device: Device to run on ('cuda' or 'cpu')
            load_in_4bit: Use 4-bit quantization (reduces VRAM from 24GB to 8GB)
        """
        self.model_path = model_path
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.is_loaded = False

        logger.info(f"Kimi provider initialized (model not loaded yet)")

    def load_model(self):
        """Load Kimi model (lazy loading)"""
        if self.is_loaded:
            return

        try:
            logger.info(f"Loading Kimi model from {self.model_path}...")

            # Try to import Kimi Audio library
            try:
                from transformers import AutoModelForCausalLM, AutoProcessor

                # Load model with optimizations
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map=self.device,
                    load_in_4bit=self.load_in_4bit if self.device == "cuda" else False,
                    trust_remote_code=True,
                )

                self.processor = AutoProcessor.from_pretrained(
                    self.model_path, trust_remote_code=True
                )

                self.is_loaded = True
                logger.info(f"[OK] Kimi model loaded successfully on {self.device}")

            except ImportError as e:
                logger.error(f"Kimi Audio library not available: {e}")
                raise ImportError(
                    "Kimi Audio model requires transformers and torch. "
                    "Install with: pip install transformers torch accelerate"
                )

        except Exception as e:
            logger.error(f"Failed to load Kimi model: {e}")
            raise

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        """
        Transcribe audio to text using Kimi ASR

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            language: Language code ('en' or 'sw')

        Returns:
            Transcribed text
        """
        self.load_model()

        try:
            logger.info(f"[Kimi ASR] Transcribing {audio_path}...")

            # Prepare messages for Kimi model
            messages = [
                {"role": "user", "message_type": "audio", "content": audio_path}
            ]

            # Generate transcription (output_type="text" for ASR)
            inputs = self.processor(messages=messages, return_tensors="pt").to(
                self.device
            )

            outputs = self.model.generate(
                **inputs,
                output_type="text",
                max_new_tokens=512,
                temperature=0.1,  # Low temperature for accurate transcription
            )

            transcription = self.processor.decode(outputs[0], skip_special_tokens=True)

            logger.info(f"[Kimi ASR] Success: {transcription[:50]}...")
            return transcription

        except Exception as e:
            logger.error(f"[Kimi ASR] Failed: {e}")
            raise

    def synthesize(
        self, text: str, output_path: str, voice: str = "default", language: str = "en"
    ) -> str:
        """
        Synthesize speech from text using Kimi TTS

        Args:
            text: Text to convert to speech
            output_path: Path to save audio file
            voice: Voice selection (ignored for Kimi, uses default)
            language: Language code

        Returns:
            Path to generated audio file
        """
        self.load_model()

        try:
            logger.info(f"[Kimi TTS] Generating speech for: {text[:50]}...")

            # Prepare messages for Kimi TTS
            messages = [{"role": "user", "message_type": "text", "content": text}]

            # Generate speech (output_type="audio" for TTS)
            inputs = self.processor(messages=messages, return_tensors="pt").to(
                self.device
            )

            outputs = self.model.generate(
                **inputs,
                output_type="audio",
                temperature=0.7,
                stream=True,  # Streaming generation
            )

            # Save audio to file
            import soundfile as sf
            import torch

            # Kimi outputs tensor at 24kHz
            sample_rate = 24000

            # Convert to numpy and save
            waveform = (
                outputs.cpu().numpy() if isinstance(outputs, torch.Tensor) else outputs
            )
            sf.write(output_path, waveform, sample_rate)

            logger.info(f"[Kimi TTS] Success: saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"[Kimi TTS] Failed: {e}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """
        Check provider health

        Returns:
            Health status dict
        """
        try:
            # Try to load model if not loaded
            if not self.is_loaded:
                self.load_model()

            return {
                "provider": "kimi",
                "status": "healthy",
                "is_loaded": self.is_loaded,
                "device": self.device,
                "model_path": self.model_path,
            }
        except Exception as e:
            return {
                "provider": "kimi",
                "status": "unhealthy",
                "error": str(e),
                "is_loaded": False,
            }


# Singleton instance
_kimi_provider: Optional[KimiAudioProvider] = None


def get_kimi_provider(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    load_in_4bit: Optional[bool] = None,
) -> KimiAudioProvider:
    """
    Get or create Kimi provider instance (singleton)

    Args:
        model_path: Override model path
        device: Override device
        load_in_4bit: Override quantization setting

    Returns:
        KimiAudioProvider instance
    """
    global _kimi_provider

    if _kimi_provider is None:
        # Load defaults from environment
        model_path = model_path or os.getenv(
            "KIMI_MODEL_PATH", "moonshotai/Kimi-Audio-7B-Instruct"
        )
        device = device or os.getenv("KIMI_DEVICE", "cuda")
        load_in_4bit = (
            load_in_4bit
            if load_in_4bit is not None
            else os.getenv("KIMI_LOAD_IN_4BIT", "true").lower() == "true"
        )

        _kimi_provider = KimiAudioProvider(
            model_path=model_path, device=device, load_in_4bit=load_in_4bit
        )

    return _kimi_provider
