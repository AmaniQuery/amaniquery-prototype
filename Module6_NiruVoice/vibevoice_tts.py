"""
VibeVoice TTS - Text-to-Speech using Microsoft VibeVoice-Realtime-0.5B

A simple, streamlined TTS provider that replaces the complex LiveKit-based system.
Optimized for HuggingFace Spaces deployment.
"""

import os
import io
import copy
import time
from typing import Optional, List, Dict, Any
from pathlib import Path
from loguru import logger

# Lazy imports to avoid loading heavy dependencies at startup
_model = None
_processor = None
_voice_cache = {}


def get_device():
    """Get the best available device"""
    import torch

    device = os.getenv("VIBEVOICE_DEVICE", "auto")

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device


def get_model_and_processor():
    """Lazy load VibeVoice model and processor"""
    global _model, _processor

    if _model is None:
        import torch
        from VibeVoice.vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        from VibeVoice.vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor,
        )

        model_path = os.getenv(
            "VIBEVOICE_MODEL_PATH", "microsoft/VibeVoice-Realtime-0.5B"
        )
        device = get_device()

        logger.info(f"Loading VibeVoice model from {model_path} on {device}...")

        # Load processor
        _processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)

        # Determine dtype and attention implementation based on device
        if device == "cuda":
            load_dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
        else:
            load_dtype = torch.float32
            attn_impl = "sdpa"

        try:
            _model = (
                VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    model_path,
                    torch_dtype=load_dtype,
                    device_map=device if device != "mps" else None,
                    attn_implementation=attn_impl,
                )
            )
            if device == "mps":
                _model.to("mps")
        except Exception as e:
            logger.warning(f"Failed with {attn_impl}, falling back to sdpa: {e}")
            _model = (
                VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    model_path,
                    torch_dtype=load_dtype,
                    device_map=device if device != "mps" else None,
                    attn_implementation="sdpa",
                )
            )
            if device == "mps":
                _model.to("mps")

        _model.eval()
        _model.set_ddpm_inference_steps(num_steps=5)

        logger.info(f"VibeVoice model loaded successfully on {device}")

    return _model, _processor


def get_voice_presets() -> Dict[str, str]:
    """Get available voice presets"""
    voices_dir = (
        Path(__file__).parent.parent
        / "VibeVoice"
        / "demo"
        / "voices"
        / "streaming_model"
    )

    if not voices_dir.exists():
        logger.warning(f"Voices directory not found: {voices_dir}")
        return {}

    presets = {}
    for pt_file in voices_dir.glob("*.pt"):
        name = pt_file.stem
        # Clean up name (remove prefix if present)
        if "_" in name:
            name = name.split("_")[0]
        if "-" in name:
            name = name.split("-")[-1]
        presets[name] = str(pt_file)

    return presets


def load_voice_prompt(voice_name: str):
    """Load cached voice prompt"""
    if voice_name in _voice_cache:
        return _voice_cache[voice_name]

    presets = get_voice_presets()

    if voice_name not in presets:
        # Try to find a matching preset
        for preset_name, path in presets.items():
            if (
                preset_name.lower() in voice_name.lower()
                or voice_name.lower() in preset_name.lower()
            ):
                voice_name = preset_name
                break
        else:
            # Default to first available voice
            if presets:
                voice_name = list(presets.keys())[0]
                logger.warning(f"Voice not found, using default: {voice_name}")
            else:
                raise ValueError("No voice presets available")

    import torch

    device = get_device()
    voice_path = presets[voice_name]

    _voice_cache[voice_name] = torch.load(
        voice_path, map_location=device, weights_only=False
    )

    return _voice_cache[voice_name]


class VibeVoiceTTS:
    """
    Simple VibeVoice TTS wrapper

    Usage:
        tts = VibeVoiceTTS()
        audio_bytes = await tts.synthesize("Hello world!")
    """

    def __init__(self, voice: str = None):
        """Initialize TTS with optional default voice"""
        self.default_voice = voice or os.getenv("VIBEVOICE_VOICE", "Wayne")
        self.sample_rate = 24000  # VibeVoice uses 24kHz
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization"""
        if not self._initialized:
            # Just verify we can load the model - don't actually load yet
            get_voice_presets()
            self._initialized = True

    async def synthesize(
        self,
        text: str,
        voice: str = None,
        cfg_scale: float = 1.5,
    ) -> bytes:
        """
        Convert text to speech

        Args:
            text: Text to synthesize
            voice: Voice preset name (Wayne, Carter, etc.)
            cfg_scale: Classifier-free guidance scale (1.0-2.0)

        Returns:
            WAV audio bytes
        """
        import torch
        import numpy as np
        from scipy.io import wavfile

        voice = voice or self.default_voice

        try:
            model, processor = get_model_and_processor()
            voice_prompt = load_voice_prompt(voice)
            device = get_device()

            # Prepare text (clean up special characters)
            text = text.replace("'", "'").replace('"', '"').replace('"', '"')

            start_time = time.time()

            # Process input with cached voice prompt
            inputs = processor.process_input_with_cached_prompt(
                text=text,
                cached_prompt=voice_prompt,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Move to device
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(device)

            # Generate audio
            outputs = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                all_prefilled_outputs=copy.deepcopy(voice_prompt),
            )

            generation_time = time.time() - start_time

            if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
                raise ValueError("No audio generated")

            # Get audio array
            audio = outputs.speech_outputs[0]
            if torch.is_tensor(audio):
                audio = audio.cpu().numpy()

            # Ensure proper shape
            if len(audio.shape) > 1:
                audio = audio.squeeze()

            # Normalize to int16
            audio = np.clip(audio, -1.0, 1.0)
            audio_int16 = (audio * 32767).astype(np.int16)

            # Convert to WAV bytes
            buffer = io.BytesIO()
            wavfile.write(buffer, self.sample_rate, audio_int16)
            wav_bytes = buffer.getvalue()

            audio_duration = len(audio) / self.sample_rate
            rtf = generation_time / audio_duration if audio_duration > 0 else 0

            logger.info(
                f"VibeVoice TTS: {len(text)} chars → {audio_duration:.1f}s audio "
                f"in {generation_time:.1f}s (RTF: {rtf:.2f}x)"
            )

            return wav_bytes

        except Exception as e:
            logger.error(f"VibeVoice TTS error: {e}")
            raise

    def synthesize_sync(
        self, text: str, voice: str = None, cfg_scale: float = 1.5
    ) -> bytes:
        """Synchronous version of synthesize"""
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.synthesize(text, voice, cfg_scale))
        finally:
            loop.close()

    def get_available_voices(self) -> List[str]:
        """Get list of available voice presets"""
        return list(get_voice_presets().keys())

    def health_check(self) -> Dict[str, Any]:
        """Check if TTS is operational"""
        try:
            voices = self.get_available_voices()
            device = get_device()

            return {
                "status": "healthy",
                "device": device,
                "voices": voices,
                "model": os.getenv(
                    "VIBEVOICE_MODEL_PATH", "microsoft/VibeVoice-Realtime-0.5B"
                ),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }


# Convenience function for simple usage
_tts_instance = None


def get_tts() -> VibeVoiceTTS:
    """Get singleton TTS instance"""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = VibeVoiceTTS()
    return _tts_instance


async def synthesize(text: str, voice: str = None) -> bytes:
    """Quick synthesize function"""
    return await get_tts().synthesize(text, voice)
