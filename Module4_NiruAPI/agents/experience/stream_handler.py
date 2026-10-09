"""
Stream Handler - Handles streaming responses
"""

from typing import AsyncIterator, Dict, Any
from loguru import logger


class StreamHandler:
    """
    Handles streaming of agent responses
    """

    async def stream_response(self, result: Dict[str, Any]) -> AsyncIterator[str]:
        """
        Stream response chunks

        Args:
            result: Agent result

        Yields:
            Response chunks
        """
        answer = result.get("answer", "")

        # Stream answer in chunks
        chunk_size = 50
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i : i + chunk_size]
            yield chunk
            # Small delay for streaming effect
            import asyncio

            await asyncio.sleep(0.05)

        # Stream sources
        sources = result.get("sources", [])
        if sources:
            yield "\n\nSources:\n"
            for i, source in enumerate(sources, 1):
                source_text = f"[{i}] {source.get('title', 'Unknown')}\n"
                yield source_text
                await asyncio.sleep(0.02)
