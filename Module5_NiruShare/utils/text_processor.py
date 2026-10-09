"""
Advanced Text Processing Utilities for Social Media Formatting
Uses robust sentence segmentation and intelligent truncation.
"""

import re
from typing import List, Optional
from loguru import logger

try:
    from textblob import TextBlob

    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning(
        "TextBlob not available, falling back to regex for sentence splitting"
    )


class TextProcessor:
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        Split text into sentences using NLP if available, or robust regex.
        """
        if not text:
            return []

        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                return [str(s) for s in blob.sentences]
            except Exception as e:
                logger.warning(f"TextBlob failed: {e}, falling back to regex")

        # Robust regex fallback
        # Handles . ! ? followed by space or newline, but ignores common abbreviations like Mr. Dr. etc.
        # This is a simplified version; production might need more complex rules
        pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def extract_key_points(
        text: str, max_points: int = 3, min_length: int = 20
    ) -> List[str]:
        """
        Extract key points from text based on sentence significance.
        For now, uses first N sentences, but can be enhanced with summarization logic.
        """
        sentences = TextProcessor.split_sentences(text)

        # Filter short/empty sentences
        valid_sentences = [s for s in sentences if len(s) >= min_length]

        if not valid_sentences:
            # If no valid sentences found (e.g. one long run-on), return chunk
            return [text[:200] + "..." if len(text) > 200 else text]

        return valid_sentences[:max_points]

    @staticmethod
    def smart_truncate(text: str, max_length: int, suffix: str = "...") -> str:
        """
        Truncate text intelligently at sentence or word boundaries.
        """
        if not text:
            return ""

        if len(text) <= max_length:
            return text

        suffix_len = len(suffix)
        target_len = max_length - suffix_len

        if target_len <= 0:
            return suffix[:max_length]

        # Try to cut at sentence ending
        sentences = TextProcessor.split_sentences(text)
        current_len = 0
        truncated_sentences = []

        for sent in sentences:
            if current_len + len(sent) + 1 <= target_len:  # +1 for space
                truncated_sentences.append(sent)
                current_len += len(sent) + 1
            else:
                break

        if truncated_sentences:
            return " ".join(truncated_sentences) + suffix

        # Fallback: Cut at last space
        truncated = text[:target_len]
        last_space = truncated.rfind(" ")
        if last_space != -1:
            return truncated[:last_space] + suffix

        # Hard cut
        return truncated + suffix

    @staticmethod
    def clean_text(text: str) -> str:
        """Remove excessive whitespace and normalize text."""
        if not text:
            return ""
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        # Normalize newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Remove double newlines if needed, or keep for paragraphs
        return text.strip()
