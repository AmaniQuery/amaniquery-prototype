"""
Mock STT/TTS providers for testing
"""

import asyncio
from typing import Optional


class MockSTT:
    """Mock STT provider for testing"""

    def __init__(self, should_fail: bool = False, delay: float = 0.0):
        self.should_fail = should_fail
        self.delay = delay
        self.call_count = 0

    async def transcribe(self, audio_data: bytes) -> str:
        """Mock transcription"""
        self.call_count += 1

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.should_fail:
            raise ConnectionError("Mock STT failure")

        return "This is a mock transcription"


class MockTTS:
    """Mock TTS provider for testing"""

    def __init__(self, should_fail: bool = False, delay: float = 0.0):
        self.should_fail = should_fail
        self.delay = delay
        self.call_count = 0

    async def synthesize(self, text: str) -> bytes:
        """Mock synthesis"""
        self.call_count += 1

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.should_fail:
            raise ConnectionError("Mock TTS failure")

        return b"mock_audio_data"
