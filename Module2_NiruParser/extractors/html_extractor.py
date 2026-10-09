"""
HTML Text Extractor using Trafilatura
"""

import trafilatura
from typing import Dict, Optional
from loguru import logger


class HTMLExtractor:
    """Extract clean text from HTML using Trafilatura"""

    def __init__(self):
        self.config = trafilatura.settings.use_config()
        # Configure for maximum extraction
        self.config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "100")
        self.config.set("DEFAULT", "MIN_OUTPUT_SIZE", "100")

    def extract(self, html: str, url: Optional[str] = None) -> Dict[str, str]:
        """
        Extract clean text from HTML

        Args:
            html: Raw HTML string
            url: Optional URL for better extraction

        Returns:
            Dictionary with extracted fields
        """
        try:
            # Extract with metadata
            extracted = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=True,
                output_format="txt",
                config=self.config,
                with_metadata=True,
            )

            if not extracted:
                logger.warning(
                    "Trafilatura failed to extract text, trying bare extraction"
                )
                # Fallback to bare extraction
                text = trafilatura.extract(
                    html,
                    no_fallback=False,
                    include_tables=True,
                )
                return {
                    "text": text or "",
                    "title": "",
                    "author": "",
                    "date": "",
                }

            # Extract metadata
            metadata = trafilatura.metadata.extract_metadata(html, default_url=url)

            return {
                "text": extracted,
                "title": metadata.title if metadata else "",
                "author": metadata.author if metadata else "",
                "date": metadata.date if metadata else "",
                "description": metadata.description if metadata else "",
            }

        except Exception as e:
            logger.error(f"Error extracting HTML: {e}")
            return {
                "text": "",
                "title": "",
                "author": "",
                "date": "",
            }

    def extract_from_file(self, file_path: str) -> Dict[str, str]:
        """Extract from HTML file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()
            return self.extract(html)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {"text": "", "title": "", "author": "", "date": ""}
