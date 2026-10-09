"""
Unified Article Extractor with Multiple Strategies
Uses Trafilatura, Newspaper3k, and site-specific rules
"""

import sys
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import trafilatura

    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    logger.warning("Trafilatura not available")

try:
    from newspaper import Article as NewspaperArticle

    NEWSPAPER3K_AVAILABLE = True
except ImportError:
    NEWSPAPER3K_AVAILABLE = False
    logger.warning("Newspaper3k not available")

from .site_specific import SiteSpecificExtractor


class ArticleExtractor:
    """
    Unified article extractor with fallback strategies:
    1. Site-specific extractors (for known Kenyan sources)
    2. Trafilatura (primary general extractor)
    3. Newspaper3k (fallback)
    """

    def __init__(self):
        self.site_extractor = SiteSpecificExtractor()

        # Configure Trafilatura
        if TRAFILATURA_AVAILABLE:
            self.trafilatura_config = trafilatura.settings.use_config()
            self.trafilatura_config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "100")
            self.trafilatura_config.set("DEFAULT", "MIN_OUTPUT_SIZE", "100")

    def extract(self, html: str, url: Optional[str] = None) -> Dict:
        """
        Extract article content and metadata from HTML

        Args:
            html: Raw HTML content
            url: Article URL (helps with site-specific extraction)

        Returns:
            Dictionary with:
            - text: Extracted article text
            - title: Article title
            - author: Author name(s)
            - date: Publication date
            - description: Article description/summary
            - category: Article category
            - tags: Article tags
            - images: List of image URLs
            - extraction_method: Which method was used
        """
        if not html or len(html.strip()) < 100:
            return self._empty_result("No HTML content provided")

        domain = None
        if url:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
            except Exception:
                pass

        # Try site-specific extractor first
        if domain:
            site_result = self.site_extractor.extract(html, url, domain)
            if (
                site_result
                and site_result.get("text")
                and len(site_result.get("text", "")) > 200
            ):
                site_result["extraction_method"] = "site_specific"
                logger.debug(f"Extracted using site-specific rules for {domain}")
                return site_result

        # Try Trafilatura
        if TRAFILATURA_AVAILABLE:
            trafilatura_result = self._extract_trafilatura(html, url)
            if (
                trafilatura_result
                and trafilatura_result.get("text")
                and len(trafilatura_result.get("text", "")) > 200
            ):
                trafilatura_result["extraction_method"] = "trafilatura"
                logger.debug(f"Extracted using Trafilatura for {url or 'unknown'}")
                return trafilatura_result

        # Try Newspaper3k as fallback
        if NEWSPAPER3K_AVAILABLE and url:
            newspaper_result = self._extract_newspaper3k(html, url)
            if (
                newspaper_result
                and newspaper_result.get("text")
                and len(newspaper_result.get("text", "")) > 200
            ):
                newspaper_result["extraction_method"] = "newspaper3k"
                logger.debug(f"Extracted using Newspaper3k for {url}")
                return newspaper_result

        # If all methods fail, return empty result
        logger.warning(f"All extraction methods failed for {url or 'unknown URL'}")
        return self._empty_result("All extraction methods failed")

    def _extract_trafilatura(self, html: str, url: Optional[str] = None) -> Dict:
        """Extract using Trafilatura"""
        try:
            # Extract text with metadata
            extracted_text = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=True,
                output_format="txt",
                config=self.trafilatura_config,
                with_metadata=True,
            )

            if not extracted_text:
                # Try without metadata
                extracted_text = trafilatura.extract(
                    html,
                    no_fallback=False,
                    include_tables=True,
                )

            if not extracted_text:
                return None

            # Extract metadata
            metadata = trafilatura.metadata.extract_metadata(html, default_url=url)

            return {
                "text": extracted_text,
                "title": metadata.title if metadata else "",
                "author": metadata.author if metadata else "",
                "date": metadata.date if metadata else "",
                "description": metadata.description if metadata else "",
                "category": "",
                "tags": [],
                "images": [],
            }
        except Exception as e:
            logger.error(f"Trafilatura extraction error: {e}")
            return None

    def _extract_newspaper3k(self, html: str, url: str) -> Dict:
        """Extract using Newspaper3k"""
        try:
            article = NewspaperArticle(url)
            article.set_html(html)
            article.parse()

            if not article.text or len(article.text) < 200:
                return None

            return {
                "text": article.text,
                "title": article.title or "",
                "author": ", ".join(article.authors) if article.authors else "",
                "date": (
                    article.publish_date.isoformat() if article.publish_date else ""
                ),
                "description": article.meta_description or "",
                "category": "",
                "tags": article.keywords or [],
                "images": article.images or [],
            }
        except Exception as e:
            logger.error(f"Newspaper3k extraction error: {e}")
            return None

    def _empty_result(self, reason: str = "") -> Dict:
        """Return empty result structure"""
        return {
            "text": "",
            "title": "",
            "author": "",
            "date": "",
            "description": "",
            "category": "",
            "tags": [],
            "images": [],
            "extraction_method": "failed",
            "failure_reason": reason,
        }
