"""
Sitemap Spider - Fetches articles from sitemaps
Useful for sources that don't have RSS feeds
"""

import scrapy
from datetime import datetime
from dateutil import parser as date_parser
from urllib.parse import urljoin, urlparse
from ..items import DocumentItem
from ..extractors import ArticleExtractor
import re


class SitemapSpider(scrapy.Spider):
    name = "sitemap"

    def __init__(self, *args, **kwargs):
        super(SitemapSpider, self).__init__(*args, **kwargs)
        self.article_extractor = ArticleExtractor()

    # Sitemap URLs for Kenyan news sources and government sites
    sitemap_urls = [
        # News sources
        "https://nation.africa/sitemap.xml",
        "https://www.standardmedia.co.ke/sitemap.xml",
        "https://www.the-star.co.ke/sitemap.xml",
        "https://www.businessdailyafrica.com/sitemap.xml",
        "https://www.citizen.digital/sitemap.xml",
        "https://www.ktnnews.com/sitemap.xml",
        "https://www.ntvkenya.co.ke/sitemap.xml",
        # Government sources
        "http://www.parliament.go.ke/sitemap.xml",
        "http://kenyalaw.org/sitemap.xml",
    ]

    custom_settings = {
        "DOWNLOAD_DELAY": 2.0,
        "ROBOTSTXT_OBEY": True,
    }

    def start_requests(self):
        """Generate requests for each sitemap"""
        for sitemap_url in self.sitemap_urls:
            yield scrapy.Request(
                url=sitemap_url,
                callback=self.parse_sitemap,
                meta={"sitemap_url": sitemap_url},
                dont_filter=True,
            )

    def parse_sitemap(self, response):
        """Parse sitemap XML"""
        sitemap_url = response.meta["sitemap_url"]
        domain = urlparse(sitemap_url).netloc

        self.logger.info(f"Parsing sitemap: {sitemap_url}")

        # Parse sitemap XML
        urls = response.xpath("//url/loc/text()").getall()

        if not urls:
            # Try sitemap index
            sitemap_indexes = response.xpath("//sitemap/loc/text()").getall()
            if sitemap_indexes:
                self.logger.info(
                    f"Found sitemap index with {len(sitemap_indexes)} sitemaps"
                )
                for sitemap_index in sitemap_indexes[:10]:  # Limit to 10 sitemaps
                    yield scrapy.Request(
                        url=sitemap_index,
                        callback=self.parse_sitemap,
                        meta={"sitemap_url": sitemap_index},
                    )
                return

        self.logger.info(f"Found {len(urls)} URLs in sitemap: {sitemap_url}")

        # Filter for news/article URLs
        article_urls = []
        for url in urls[:500]:  # Limit to 500 URLs per sitemap
            if self._is_article_url(url):
                article_urls.append(url)

        self.logger.info(f"Filtered to {len(article_urls)} article URLs")

        # Request each article
        for url in article_urls[:100]:  # Limit to 100 articles per sitemap
            yield scrapy.Request(
                url=url,
                callback=self.parse_article,
                meta={"source_name": domain, "sitemap_url": sitemap_url},
                errback=self.errback_article,
            )

    def _is_article_url(self, url: str) -> bool:
        """Check if URL is likely an article"""
        # Common article URL patterns
        article_patterns = [
            r"/news/",
            r"/article/",
            r"/story/",
            r"/post/",
            r"/202\d/",  # Year in URL (common for news)
            r"/kenya/",
        ]

        # Exclude non-article URLs
        exclude_patterns = [
            r"/category/",
            r"/tag/",
            r"/author/",
            r"/page/",
            r"/search",
            r"/contact",
            r"/about",
            r"\.(jpg|png|gif|pdf|css|js)$",
        ]

        url_lower = url.lower()

        # Check excludes first
        for pattern in exclude_patterns:
            if re.search(pattern, url_lower):
                return False

        # Check if matches article pattern
        for pattern in article_patterns:
            if re.search(pattern, url_lower):
                return True

        return False

    def parse_article(self, response):
        """Parse article from URL"""
        source_name = response.meta.get("source_name", "Unknown")

        # Use enhanced article extractor
        extracted = self.article_extractor.extract(response.text, url=response.url)

        # Use extracted content
        content = extracted.get("text", "")
        if not content or len(content) < 100:
            self.logger.warning(f"Extraction failed for {response.url}")
            return

        # Use extracted title
        title = extracted.get("title", "")
        if not title:
            title = response.css("title::text").get() or "Untitled"

        # Use extracted author
        author = extracted.get("author", "")

        # Use extracted date
        publication_date = extracted.get("date", "")

        # Use extracted description
        summary = extracted.get("description", "")

        # Create document item
        yield DocumentItem(
            url=response.url,
            title=title,
            content=content,
            content_type="html",
            category="Kenyan News",
            source_name=source_name,
            author=author,
            publication_date=publication_date,
            summary=summary,
            raw_html=response.text,
            status_code=response.status,
        )

    def errback_article(self, failure):
        """Handle failed article fetches"""
        self.logger.error(f"Failed to fetch article: {failure.request.url}")
