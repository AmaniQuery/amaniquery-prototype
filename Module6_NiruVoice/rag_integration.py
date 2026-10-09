"""
RAG Pipeline Integration for Voice Agent
"""

import os
import re
from typing import Dict, Optional, List
from loguru import logger
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Module4_NiruAPI.rag_pipeline import RAGPipeline
from Module3_NiruDB.vector_store import VectorStore
from Module4_NiruAPI.config_manager import ConfigManager


class VoiceRAGIntegration:
    """Wrapper around RAGPipeline optimized for voice responses"""

    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        vector_store: Optional[VectorStore] = None,
        config_manager: Optional[ConfigManager] = None,
        top_k: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ):
        """
        Initialize RAG integration

        Args:
            rag_pipeline: Existing RAGPipeline instance (creates new if None)
            vector_store: VectorStore instance (for creating new RAGPipeline)
            config_manager: ConfigManager instance
            top_k: Number of documents to retrieve
            temperature: LLM temperature
            max_tokens: Maximum tokens in response
        """
        if rag_pipeline:
            self.rag_pipeline = rag_pipeline
        else:
            # Create new RAG pipeline
            llm_provider = os.getenv("LLM_PROVIDER", "moonshot")
            model = os.getenv("DEFAULT_MODEL", "moonshot-v1-8k")

            self.rag_pipeline = RAGPipeline(
                vector_store=vector_store,
                llm_provider=llm_provider,
                model=model,
                config_manager=config_manager,
            )

        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info("Voice RAG integration initialized")

    def query(
        self, query_text: str, conversation_context: Optional[List[str]] = None
    ) -> Dict:
        """
        Process a voice query through RAG pipeline

        Args:
            query_text: User's spoken query
            conversation_context: Previous conversation turns for context

        Returns:
            Dictionary with formatted response for voice output
        """
        try:
            # Enhance query with context if available
            enhanced_query = self._enhance_query_with_context(
                query_text, conversation_context
            )

            # Run RAG query
            result = self.rag_pipeline.query(
                query=enhanced_query,
                top_k=self.top_k,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_context_length=3000,
            )

            # Format response for voice
            voice_response = self._format_for_voice(result)

            return voice_response

        except Exception as e:
            logger.error(f"Error processing voice query: {e}")
            return {
                "text": "I apologize, but I encountered an error processing your query. Please try again.",
                "sources": [],
                "has_error": True,
            }

    def _enhance_query_with_context(
        self, query_text: str, conversation_context: Optional[List[str]] = None
    ) -> str:
        """Enhance query with conversation context"""
        if not conversation_context or len(conversation_context) == 0:
            return query_text

        # Add recent context (last 2-3 turns)
        recent_context = (
            conversation_context[-3:]
            if len(conversation_context) > 3
            else conversation_context
        )

        context_str = "Previous conversation context:\n"
        for i, turn in enumerate(recent_context, 1):
            context_str += f"{i}. {turn}\n"

        enhanced = f"{context_str}\nCurrent question: {query_text}"
        return enhanced

    def _format_for_voice(self, rag_result: Dict) -> Dict:
        """
        Format RAG response for voice output

        Converts markdown to natural speech, adds pauses, formats sources
        """
        answer = rag_result.get("answer", "")
        sources = rag_result.get("sources", [])

        # Convert markdown to natural speech
        voice_text = self._markdown_to_speech(answer)

        # Add source citations naturally
        if sources:
            voice_text = self._add_source_citations(voice_text, sources)

        # Ensure response is concise (limit length)
        voice_text = self._truncate_for_voice(voice_text, max_words=500)

        return {
            "text": voice_text,
            "sources": sources,
            "has_error": False,
            "query_time": rag_result.get("query_time", 0),
            "retrieved_chunks": rag_result.get("retrieved_chunks", 0),
        }

    def _markdown_to_speech(self, text: str) -> str:
        """Convert markdown formatting to natural speech"""
        # Remove markdown headers but keep the text
        text = re.sub(r"^#+\s+(.+)$", r"\1", text, flags=re.MULTILINE)

        # Convert bold to emphasis
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"__(.+?)__", r"\1", text)

        # Convert italic to natural speech
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"_(.+?)_", r"\1", text)

        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`(.+?)`", r"\1", text)

        # Convert lists to natural speech
        text = re.sub(r"^\s*[-*+]\s+(.+)$", r"\1", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+(.+)$", r"\1", text, flags=re.MULTILINE)

        # Remove extra whitespace
        text = re.sub(r"\n\s*\n", ". ", text)
        text = re.sub(r"\s+", " ", text)

        # Add natural pauses after periods
        text = text.replace(". ", ". ... ")

        return text.strip()

    def _add_source_citations(self, text: str, sources: List[Dict]) -> str:
        """Add natural source citations to voice text"""
        if not sources:
            return text

        # Mention top sources naturally
        top_sources = sources[:2]  # Mention top 2 sources

        citation_text = " This information is based on "

        if len(top_sources) == 1:
            source_name = top_sources[0].get("source_name", "available sources")
            citation_text += f"{source_name}."
        else:
            source_names = [s.get("source_name", "sources") for s in top_sources]
            citation_text += f"{source_names[0]} and {source_names[1]}."

        return text + citation_text

    def _truncate_for_voice(self, text: str, max_words: int = 500) -> str:
        """Truncate text to maximum word count for voice response"""
        words = text.split()

        if len(words) <= max_words:
            return text

        # Truncate and add ellipsis
        truncated = " ".join(words[:max_words])

        # Try to end at a sentence boundary
        last_period = truncated.rfind(".")
        if last_period > len(truncated) * 0.8:  # If period is in last 20%
            truncated = truncated[: last_period + 1]
        else:
            truncated += "..."

        return truncated
