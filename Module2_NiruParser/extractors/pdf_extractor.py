"""
PDF Text Extractor using pdfplumber
"""

import pdfplumber
from typing import Dict, List, Any
from pathlib import Path
from loguru import logger


class PDFExtractor:
    """Extract text from PDF files"""

    def __init__(self):
        self.extraction_settings = {
            "x_tolerance": 3,
            "y_tolerance": 3,
        }

    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF file

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with text and metadata
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Extract metadata
                metadata = pdf.metadata or {}

                # Extract text from all pages
                pages_text = []
                for i, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text(**self.extraction_settings)
                        if text:
                            pages_text.append(text)
                    except Exception as e:
                        logger.warning(f"Failed to extract page {i+1}: {e}")
                        continue

                # Combine all pages
                full_text = "\n\n".join(pages_text)

                return {
                    "text": full_text,
                    "title": metadata.get("Title", ""),
                    "author": metadata.get("Author", ""),
                    "creation_date": metadata.get("CreationDate", ""),
                    "subject": metadata.get("Subject", ""),
                    "num_pages": len(pdf.pages),
                }

        except Exception as e:
            logger.error(f"Error extracting PDF {pdf_path}: {e}")
            return {
                "text": "",
                "title": "",
                "author": "",
                "creation_date": "",
                "num_pages": 0,
            }

    def extract_with_layout(self, pdf_path: str) -> List[Dict]:
        """Extract text while preserving layout information"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_data = []

                for i, page in enumerate(pdf.pages):
                    try:
                        # Extract words with positions
                        words = page.extract_words(**self.extraction_settings)
                        text = page.extract_text(**self.extraction_settings)

                        pages_data.append(
                            {
                                "page_number": i + 1,
                                "text": text,
                                "words": words,
                                "width": page.width,
                                "height": page.height,
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to extract page {i+1} with layout: {e}")
                        continue

                return pages_data

        except Exception as e:
            logger.error(f"Error extracting PDF with layout {pdf_path}: {e}")
            return []
