"""Hansard Tool - Retrieve Kenyan parliamentary debate transcripts"""

from typing import Dict, Any, Optional
from loguru import logger


class HansardTool:
    """Retrieve Kenyan parliamentary debate transcripts (Hansard) by topic or date"""

    name = "hansard"
    description = (
        "Retrieve parliamentary debate transcripts (Hansard) by date, topic, or speaker"
    )

    def __init__(self):
        self._vector_store = None

    def execute(
        self, topic: str = "", date: str = "", speaker: str = ""
    ) -> Dict[str, Any]:
        try:
            logger.info(
                f"[Tool] hansard: topic='{topic}', date='{date}', speaker='{speaker}'"
            )
            vs = self._get_vector_store()
            if not vs:
                return {
                    "success": False,
                    "error": "Vector store not available",
                    "suggestion": "Try kb_search for parliamentary information instead",
                }

            search_query = (
                topic or f"parliamentary debate {date}" or "Kenya parliamentary debate"
            )
            filters = {"category": "hansard"}
            if date:
                filters["date"] = date
            if speaker:
                filters["speaker"] = speaker

            results = vs.search(
                query=search_query.replace("_", " "), filters=filters, top_k=10
            )
            if results:
                return {
                    "success": True,
                    "topic": topic or date,
                    "speaker": speaker or "any",
                    "transcripts": results,
                    "count": len(results),
                }
            return {
                "success": False,
                "error": "No Hansard transcripts found",
                "suggestion": "Try a broader topic or search kb_search",
            }
        except Exception as e:
            logger.error(f"Hansard lookup failed: {e}")
            return {"success": False, "error": str(e)}

    def _get_vector_store(self):
        if self._vector_store is None:
            try:
                from Module3_NiruDB.qdrant_cloud import get_vector_store
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                    future = asyncio.run_coroutine_threadsafe(get_vector_store(), loop)
                    self._vector_store = future.result(timeout=5)
                except RuntimeError:
                    self._vector_store = asyncio.run(get_vector_store())
            except Exception as e:
                logger.warning(f"Could not load vector store: {e}")
        return self._vector_store
