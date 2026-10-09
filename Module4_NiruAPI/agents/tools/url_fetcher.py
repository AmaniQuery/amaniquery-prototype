"""
URL Fetcher Tool - Fetches and extracts content from URLs
"""

from typing import Dict, Any, Optional
from loguru import logger

try:
    import httpx
except ImportError:
    httpx = None
    logger.warning("httpx not installed. URL fetching will not work.")


class URLFetcherTool:
    """Tool for fetching and extracting content from URLs"""

    def __init__(self, timeout: int = 30):
        """
        Initialize URL fetcher

        Args:
            timeout: Request timeout in seconds
        """
        if httpx is None:
            raise ImportError("httpx package required for URL fetching")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def execute_async(
        self, url: str, extract_text: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch URL content asynchronously

        Args:
            url: URL to fetch
            extract_text: Whether to extract text from HTML

        Returns:
            Fetched content with metadata
        """
        try:
            response = await self.client.get(url, follow_redirects=True)
            response.raise_for_status()

            content = response.text
            metadata = {
                "url": url,
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "content_length": len(content),
            }

            if extract_text and "text/html" in metadata["content_type"]:
                # Extract text from HTML
                try:
                    from bs4 import BeautifulSoup

                    soup = BeautifulSoup(content, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)
                    metadata["extracted_text"] = text[:5000]  # Limit text length
                except ImportError:
                    logger.warning("BeautifulSoup not available for text extraction")
                    metadata["extracted_text"] = content[:5000]

            return {
                "url": url,
                "content": content[:10000],  # Limit content length
                "metadata": metadata,
                "success": True,
            }
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return {"url": url, "content": "", "error": str(e), "success": False}

    def execute(self, url: str, extract_text: bool = True) -> Dict[str, Any]:
        """
        Fetch URL content synchronously

        Args:
            url: URL to fetch
            extract_text: Whether to extract text from HTML

        Returns:
            Fetched content with metadata
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.execute_async(url, extract_text))
