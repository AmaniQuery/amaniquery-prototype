"""
PDF Page Extractor - Convert PDF pages to images
"""

import os
from pathlib import Path
from typing import List, Optional, Union
from loguru import logger
from PIL import Image

try:
    from pdf2image import convert_from_path, convert_from_bytes

    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not available. Install with: pip install pdf2image")


class PDFPageExtractor:
    """Extract PDF pages as images"""

    def __init__(
        self,
        dpi: int = 200,
        fmt: str = "RGB",
        poppler_path: Optional[str] = None,
    ):
        """
        Initialize PDF page extractor

        Args:
            dpi: DPI for image conversion (higher = better quality, larger files)
            fmt: Image format (RGB, RGBA, etc.)
            poppler_path: Path to poppler binaries (if not in PATH)
        """
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError(
                "pdf2image not available. Install with: pip install pdf2image"
            )

        self.dpi = dpi
        self.fmt = fmt
        self.poppler_path = poppler_path

        logger.info(f"PDF page extractor initialized (DPI: {dpi}, Format: {fmt})")

    def extract_pages_from_file(
        self,
        pdf_path: Union[str, Path],
        first_page: Optional[int] = None,
        last_page: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Extract pages from PDF file as PIL Images

        Args:
            pdf_path: Path to PDF file
            first_page: First page to extract (1-indexed, None = first page)
            last_page: Last page to extract (1-indexed, None = last page)

        Returns:
            List of PIL Image objects, one per page
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            # Convert PDF pages to images
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                fmt=self.fmt,
                first_page=first_page,
                last_page=last_page,
                poppler_path=self.poppler_path,
            )

            logger.info(f"Extracted {len(images)} page(s) from PDF: {pdf_path.name}")
            return images

        except Exception as e:
            logger.error(f"Error extracting PDF pages from file: {e}")
            raise

    def extract_pages_from_bytes(
        self,
        pdf_bytes: bytes,
        first_page: Optional[int] = None,
        last_page: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Extract pages from PDF bytes as PIL Images

        Args:
            pdf_bytes: PDF file content as bytes
            first_page: First page to extract (1-indexed, None = first page)
            last_page: Last page to extract (1-indexed, None = last page)

        Returns:
            List of PIL Image objects, one per page
        """
        try:
            # Convert PDF pages to images from bytes
            images = convert_from_bytes(
                pdf_bytes,
                dpi=self.dpi,
                fmt=self.fmt,
                first_page=first_page,
                last_page=last_page,
                poppler_path=self.poppler_path,
            )

            logger.info(f"Extracted {len(images)} page(s) from PDF bytes")
            return images

        except Exception as e:
            logger.error(f"Error extracting PDF pages from bytes: {e}")
            raise

    def extract_all_pages(
        self,
        pdf_source: Union[str, Path, bytes],
        first_page: Optional[int] = None,
        last_page: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Extract all pages from PDF (convenience method)

        Args:
            pdf_source: PDF file path or bytes
            first_page: First page to extract (1-indexed, None = first page)
            last_page: Last page to extract (1-indexed, None = last page)

        Returns:
            List of PIL Image objects, one per page
        """
        if isinstance(pdf_source, bytes):
            return self.extract_pages_from_bytes(pdf_source, first_page, last_page)
        else:
            return self.extract_pages_from_file(pdf_source, first_page, last_page)

    def save_pages(
        self,
        images: List[Image.Image],
        output_dir: Union[str, Path],
        base_name: str = "page",
        format: str = "PNG",
    ) -> List[Path]:
        """
        Save extracted page images to disk

        Args:
            images: List of PIL Image objects
            output_dir: Directory to save images
            base_name: Base name for saved files (will append page number)
            format: Image format (PNG, JPEG, etc.)

        Returns:
            List of paths to saved image files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for i, image in enumerate(images, start=1):
            output_path = output_dir / f"{base_name}_{i:04d}.{format.lower()}"
            image.save(output_path, format=format)
            saved_paths.append(output_path)
            logger.debug(f"Saved page {i} to {output_path}")

        logger.info(f"Saved {len(saved_paths)} page images to {output_dir}")
        return saved_paths
