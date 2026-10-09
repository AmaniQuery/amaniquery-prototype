"""
Site-specific extractors for major Kenyan news sources
Provides custom extraction rules for better content quality
"""

import re
from typing import Dict, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from loguru import logger


class SiteSpecificExtractor:
    """
    Site-specific extraction rules for Kenyan news sources
    Handles unique HTML structures of major news sites
    """

    # Domain mappings to extraction strategies
    DOMAIN_RULES = {
        "nation.africa": "nation",
        "www.nation.co.ke": "nation",
        "standardmedia.co.ke": "standard",
        "www.standardmedia.co.ke": "standard",
        "the-star.co.ke": "star",
        "www.the-star.co.ke": "star",
        "businessdailyafrica.com": "business_daily",
        "www.businessdailyafrica.com": "business_daily",
        "capitalfm.co.ke": "capital_fm",
        "www.capitalfm.co.ke": "capital_fm",
        "citizen.digital": "citizen",
        "www.citizen.digital": "citizen",
        "ktnnews.co.ke": "ktn",
        "www.ktnnews.co.ke": "ktn",
        "ntv.co.ke": "ntv",
        "www.ntv.co.ke": "ntv",
        "tuko.co.ke": "tuko",
        "www.tuko.co.ke": "tuko",
        "hivisasa.com": "hivisasa",
        "www.hivisasa.com": "hivisasa",
        "pulselive.co.ke": "pulse",
        "www.pulselive.co.ke": "pulse",
    }

    def extract(self, html: str, url: Optional[str], domain: str) -> Optional[Dict]:
        """
        Extract using site-specific rules

        Args:
            html: Raw HTML
            url: Article URL
            domain: Domain name (lowercase)

        Returns:
            Extracted content dict or None if no rule matches
        """
        strategy = self.DOMAIN_RULES.get(domain)
        if not strategy:
            return None

        try:
            soup = BeautifulSoup(html, "lxml")

            if strategy == "nation":
                return self._extract_nation(soup, url)
            elif strategy == "standard":
                return self._extract_standard(soup, url)
            elif strategy == "star":
                return self._extract_star(soup, url)
            elif strategy == "business_daily":
                return self._extract_business_daily(soup, url)
            elif strategy == "capital_fm":
                return self._extract_capital_fm(soup, url)
            elif strategy == "citizen":
                return self._extract_citizen(soup, url)
            elif strategy == "ktn":
                return self._extract_ktn(soup, url)
            elif strategy == "ntv":
                return self._extract_ntv(soup, url)
            elif strategy == "tuko":
                return self._extract_tuko(soup, url)
            elif strategy == "hivisasa":
                return self._extract_hivisasa(soup, url)
            elif strategy == "pulse":
                return self._extract_pulse(soup, url)
        except Exception as e:
            logger.error(f"Site-specific extraction error for {domain}: {e}")
            return None

        return None

    def _extract_nation(self, soup: BeautifulSoup, url: Optional[str]) -> Dict:
        """Extract from Nation Africa"""
        # Title
        title = (
            soup.select_one("h1.article-title, h1.headline, article h1")
            or soup.select_one("h1")
            or soup.find("title")
        )
        title_text = title.get_text(strip=True) if title else ""

        # Author
        author = soup.select_one(
            ".article-author, .byline, [rel='author']"
        ) or soup.select_one("meta[property='article:author']")
        author_text = ""
        if author:
            if author.name == "meta":
                author_text = author.get("content", "")
            else:
                author_text = author.get_text(strip=True)

        # Date
        date = soup.select_one(
            "time.published, .article-date, [datetime]"
        ) or soup.select_one("meta[property='article:published_time']")
        date_text = ""
        if date:
            if date.name == "meta":
                date_text = date.get("content", "")
            elif date.get("datetime"):
                date_text = date.get("datetime")
            else:
                date_text = date.get_text(strip=True)

        # Content
        content_selectors = [
            ".article-body",
            ".story-body",
            "article .content",
            "[itemprop='articleBody']",
        ]
        content_parts = []
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                # Get all paragraphs
                paragraphs = content_elem.find_all("p")
                content_parts = [
                    p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
                ]
                if content_parts:
                    break

        # Images
        images = []
        img_tags = soup.select(".article-body img, .story-body img, article img")
        for img in img_tags:
            src = img.get("src") or img.get("data-src")
            if src:
                images.append(src)

        return {
            "text": "\n\n".join(content_parts),
            "title": title_text,
            "author": author_text,
            "date": date_text,
            "description": "",
            "category": "",
            "tags": [],
            "images": images,
        }

    def _extract_standard(self, soup: BeautifulSoup, url: Optional[str]) -> Dict:
        """Extract from Standard Media"""
        title = soup.select_one("h1.article-title, h1.headline, article h1, h1")
        title_text = title.get_text(strip=True) if title else ""

        author = soup.select_one(".author-name, .byline, [rel='author']")
        author_text = author.get_text(strip=True) if author else ""

        date = soup.select_one("time, .published-date, [datetime]")
        date_text = ""
        if date:
            date_text = date.get("datetime") or date.get_text(strip=True)

        content_elem = soup.select_one(
            ".article-content, .story-content, article .body"
        )
        content_parts = []
        if content_elem:
            paragraphs = content_elem.find_all("p")
            content_parts = [
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            ]

        return {
            "text": "\n\n".join(content_parts),
            "title": title_text,
            "author": author_text,
            "date": date_text,
            "description": "",
            "category": "",
            "tags": [],
            "images": [],
        }

    def _extract_star(self, soup: BeautifulSoup, url: Optional[str]) -> Dict:
        """Extract from The Star"""
        title = soup.select_one("h1.article-title, h1, article h1")
        title_text = title.get_text(strip=True) if title else ""

        author = soup.select_one(".author, .byline")
        author_text = author.get_text(strip=True) if author else ""

        date = soup.select_one("time, .date")
        date_text = ""
        if date:
            date_text = date.get("datetime") or date.get_text(strip=True)

        content_elem = soup.select_one(".article-body, .content, article")
        content_parts = []
        if content_elem:
            paragraphs = content_elem.find_all("p")
            content_parts = [
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            ]

        return {
            "text": "\n\n".join(content_parts),
            "title": title_text,
            "author": author_text,
            "date": date_text,
            "description": "",
            "category": "",
            "tags": [],
            "images": [],
        }

    def _extract_business_daily(self, soup: BeautifulSoup, url: Optional[str]) -> Dict:
        """Extract from Business Daily"""
        title = soup.select_one("h1, .headline, article h1")
        title_text = title.get_text(strip=True) if title else ""

        author = soup.select_one(".author, .byline")
        author_text = author.get_text(strip=True) if author else ""

        date = soup.select_one("time, .date")
        date_text = ""
        if date:
            date_text = date.get("datetime") or date.get_text(strip=True)

        content_elem = soup.select_one(".article-body, .story-body, article")
        content_parts = []
        if content_elem:
            paragraphs = content_elem.find_all("p")
            content_parts = [
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            ]

        return {
            "text": "\n\n".join(content_parts),
            "title": title_text,
            "author": author_text,
            "date": date_text,
            "description": "",
            "category": "",
            "tags": [],
            "images": [],
        }

    def _extract_capital_fm(self, soup: BeautifulSoup, url: Optional[str]) -> Dict:
        """Extract from Capital FM"""
        title = soup.select_one("h1, .title, article h1")
        title_text = title.get_text(strip=True) if title else ""

        author = soup.select_one(".author, .byline")
        author_text = author.get_text(strip=True) if author else ""

        date = soup.select_one("time, .date")
        date_text = ""
        if date:
            date_text = date.get("datetime") or date.get_text(strip=True)

        content_elem = soup.select_one(".content, .article-content, article")
        content_parts = []
        if content_elem:
            paragraphs = content_elem.find_all("p")
            content_parts = [
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            ]

        return {
            "text": "\n\n".join(content_parts),
            "title": title_text,
            "author": author_text,
            "date": date_text,
            "description": "",
            "category": "",
            "tags": [],
            "images": [],
        }

    def _extract_citizen(self, soup: BeautifulSoup, url: Optional[str]) -> Dict:
        """Extract from Citizen TV"""
        title = soup.select_one("h1, .article-title, article h1")
        title_text = title.get_text(strip=True) if title else ""

        author = soup.select_one(".author, .byline")
        author_text = author.get_text(strip=True) if author else ""

        date = soup.select_one("time, .date")
        date_text = ""
        if date:
            date_text = date.get("datetime") or date.get_text(strip=True)

        content_elem = soup.select_one(".article-body, .content, article")
        content_parts = []
        if content_elem:
            paragraphs = content_elem.find_all("p")
            content_parts = [
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            ]

        return {
            "text": "\n\n".join(content_parts),
            "title": title_text,
            "author": author_text,
            "date": date_text,
            "description": "",
            "category": "",
            "tags": [],
            "images": [],
        }

    def _extract_ktn(self, soup: BeautifulSoup, url: Optional[str]) -> Dict:
        """Extract from KTN News"""
        title = soup.select_one("h1, .title, article h1")
        title_text = title.get_text(strip=True) if title else ""

        author = soup.select_one(".author, .byline")
        author_text = author.get_text(strip=True) if author else ""

        date = soup.select_one("time, .date")
        date_text = ""
        if date:
            date_text = date.get("datetime") or date.get_text(strip=True)

        content_elem = soup.select_one(".content, .article-content, article")
        content_parts = []
        if content_elem:
            paragraphs = content_elem.find_all("p")
            content_parts = [
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            ]

        return {
            "text": "\n\n".join(content_parts),
            "title": title_text,
            "author": author_text,
            "date": date_text,
            "description": "",
            "category": "",
            "tags": [],
            "images": [],
        }

    def _extract_ntv(self, soup: BeautifulSoup, url: Optional[str]) -> Dict:
        """Extract from NTV Kenya"""
        title = soup.select_one("h1, .title, article h1")
        title_text = title.get_text(strip=True) if title else ""

        author = soup.select_one(".author, .byline")
        author_text = author.get_text(strip=True) if author else ""

        date = soup.select_one("time, .date")
        date_text = ""
        if date:
            date_text = date.get("datetime") or date.get_text(strip=True)

        content_elem = soup.select_one(".content, .article-content, article")
        content_parts = []
        if content_elem:
            paragraphs = content_elem.find_all("p")
            content_parts = [
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            ]

        return {
            "text": "\n\n".join(content_parts),
            "title": title_text,
            "author": author_text,
            "date": date_text,
            "description": "",
            "category": "",
            "tags": [],
            "images": [],
        }

    def _extract_tuko(self, soup: BeautifulSoup, url: Optional[str]) -> Dict:
        """Extract from Tuko.co.ke"""
        title = soup.select_one("h1, .article-title, article h1")
        title_text = title.get_text(strip=True) if title else ""

        author = soup.select_one(".author, .byline")
        author_text = author.get_text(strip=True) if author else ""

        date = soup.select_one("time, .date")
        date_text = ""
        if date:
            date_text = date.get("datetime") or date.get_text(strip=True)

        content_elem = soup.select_one(".article-body, .content, article")
        content_parts = []
        if content_elem:
            paragraphs = content_elem.find_all("p")
            content_parts = [
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            ]

        return {
            "text": "\n\n".join(content_parts),
            "title": title_text,
            "author": author_text,
            "date": date_text,
            "description": "",
            "category": "",
            "tags": [],
            "images": [],
        }

    def _extract_hivisasa(self, soup: BeautifulSoup, url: Optional[str]) -> Dict:
        """Extract from Hivisasa"""
        title = soup.select_one("h1, .title, article h1")
        title_text = title.get_text(strip=True) if title else ""

        author = soup.select_one(".author, .byline")
        author_text = author.get_text(strip=True) if author else ""

        date = soup.select_one("time, .date")
        date_text = ""
        if date:
            date_text = date.get("datetime") or date.get_text(strip=True)

        content_elem = soup.select_one(".content, .article-content, article")
        content_parts = []
        if content_elem:
            paragraphs = content_elem.find_all("p")
            content_parts = [
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            ]

        return {
            "text": "\n\n".join(content_parts),
            "title": title_text,
            "author": author_text,
            "date": date_text,
            "description": "",
            "category": "",
            "tags": [],
            "images": [],
        }

    def _extract_pulse(self, soup: BeautifulSoup, url: Optional[str]) -> Dict:
        """Extract from Pulse Live"""
        title = soup.select_one("h1, .article-title, article h1")
        title_text = title.get_text(strip=True) if title else ""

        author = soup.select_one(".author, .byline")
        author_text = author.get_text(strip=True) if author else ""

        date = soup.select_one("time, .date")
        date_text = ""
        if date:
            date_text = date.get("datetime") or date.get_text(strip=True)

        content_elem = soup.select_one(".article-body, .content, article")
        content_parts = []
        if content_elem:
            paragraphs = content_elem.find_all("p")
            content_parts = [
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            ]

        return {
            "text": "\n\n".join(content_parts),
            "title": title_text,
            "author": author_text,
            "date": date_text,
            "description": "",
            "category": "",
            "tags": [],
            "images": [],
        }
