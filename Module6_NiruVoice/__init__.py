"""
Module6_NiruVoice - Voice Module for AmaniQuery

Provides text-to-speech using Microsoft VibeVoice and integrates with RAG pipeline.
This is a simplified module that uses HTTP endpoints instead of LiveKit WebSockets.

Exports are loaded lazily (PEP 562) so that importing a submodule such as
``Module6_NiruVoice.resilience`` does not pull in the torch / Module4 chain.
"""

import importlib
from typing import Any

__all__ = [
    "VibeVoiceTTS",
    "get_tts",
    "synthesize",
    "VoiceRAGIntegration",
]

__version__ = "2.0.0"  # Major version bump for refactor

_LAZY_EXPORTS = {
    "VibeVoiceTTS": "Module6_NiruVoice.vibevoice_tts",
    "get_tts": "Module6_NiruVoice.vibevoice_tts",
    "synthesize": "Module6_NiruVoice.vibevoice_tts",
    "VoiceRAGIntegration": "Module6_NiruVoice.rag_integration",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(globals()) | set(__all__))
