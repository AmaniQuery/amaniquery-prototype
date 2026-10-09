"""
News RSS Spider - Fetches Kenyan news from RSS feeds
Enhanced with robust error handling and graceful shutdown
"""

import scrapy
import signal
import feedparser
from datetime import datetime
from dateutil import parser as date_parser
from scrapy import signals
from scrapy.exceptions import CloseSpider
from ..items import RSSItem, DocumentItem
from ..extractors import ArticleExtractor
import time


class NewsRSSSpider(scrapy.Spider):
    name = "news_rss"

    # Custom settings for robust operation
    custom_settings = {
        "DOWNLOAD_DELAY": 2.0,
        "ROBOTSTXT_OBEY": False,  # RSS feeds often not in robots.txt
        "DOWNLOAD_TIMEOUT": 30,
        "RETRY_TIMES": 3,
        "RETRY_HTTP_CODES": [500, 502, 503, 504, 408, 429],
        "CONCURRENT_REQUESTS": 8,
        "CLOSESPIDER_TIMEOUT": 1800,  # 30 minutes max
        "CLOSESPIDER_ERRORCOUNT": 100,  # Stop after 100 errors
    }

    def __init__(self, *args, **kwargs):
        super(NewsRSSSpider, self).__init__(*args, **kwargs)
        self.article_extractor = ArticleExtractor()
        self._shutdown_requested = False
        self.error_count = 0
        self.success_count = 0
        self.max_consecutive_errors = 20  # Stop if 20 consecutive errors
        self.consecutive_errors = 0

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(NewsRSSSpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        crawler.signals.connect(spider.spider_error, signal=signals.spider_error)
        return spider

    def spider_closed(self, spider, reason):
        """Handle spider close - log final stats"""
        self.logger.info(f"Spider closed: {reason}")
        self.logger.info(
            f"Final stats - Success: {self.success_count}, Errors: {self.error_count}"
        )

    def spider_error(self, failure, response, spider):
        """Handle spider errors"""
        self.error_count += 1
        self.consecutive_errors += 1

        if self.consecutive_errors >= self.max_consecutive_errors:
            self.logger.error(
                f"Too many consecutive errors ({self.consecutive_errors}). Requesting shutdown."
            )
            self._shutdown_requested = True
            raise CloseSpider(
                f"Consecutive error limit reached: {self.consecutive_errors}"
            )

    def _check_shutdown(self):
        """Check if shutdown was requested"""
        if self._shutdown_requested:
            raise CloseSpider("Shutdown requested")

    def _reset_consecutive_errors(self):
        """Reset consecutive error counter on success"""
        self.consecutive_errors = 0
        self.success_count += 1

    # Kenyan news RSS feeds - Updated with verified 2025 URLs
    rss_feeds = [
        # Standard Media - WORKING (4 feeds verified)
        {
            "url": "https://www.standardmedia.co.ke/rss/headlines.php",
            "name": "Standard Media - Headlines",
        },
        {
            "url": "https://www.standardmedia.co.ke/rss/kenya.php",
            "name": "Standard Media - Kenya",
        },
        {
            "url": "https://www.standardmedia.co.ke/rss/politics.php",
            "name": "Standard Media - Politics",
        },
        {
            "url": "https://www.standardmedia.co.ke/rss/business.php",
            "name": "Standard Media - Business",
        },
        # The Star - WORKING (3 feeds verified)
        {
            "url": "https://www.the-star.co.ke/rss/news",
            "name": "The Star - News",
        },
        {
            "url": "https://www.the-star.co.ke/rss/business",
            "name": "The Star - Business",
        },
        {
            "url": "https://www.the-star.co.ke/rss/opinion",
            "name": "The Star - Opinion",
        },
        # Nation Africa - Using /feed path (more reliable than /rss)
        {
            "url": "https://nation.africa/kenya/feed",
            "name": "Nation Africa",
        },
        # Business Daily Africa - Using main feed
        {
            "url": "https://www.businessdailyafrica.com/feed",
            "name": "Business Daily Africa",
        },
        # The East African - Using main feed
        {
            "url": "https://www.theeastafrican.co.ke/feed",
            "name": "The East African",
        },
        # NTV Kenya - WORKING
        {
            "url": "https://www.ntvkenya.co.ke/feed/",
            "name": "NTV Kenya",
        },
        # Capital FM News - WORKING
        {
            "url": "https://www.capitalfm.co.ke/news/feed/",
            "name": "Capital FM News",
        },
        # Radio Citizen - Updated path
        {
            "url": "https://www.radiocitizen.digital/feed/",
            "name": "Radio Citizen",
        },
        # Alternative sources with better reliability
        {
            "url": "https://www.ke.undp.org/content/kenya/en/home.rss",
            "name": "UNDP Kenya",
        },
        {
            "url": "https://www.worldbank.org/en/country/kenya/rss",
            "name": "World Bank Kenya",
        },
    ]

    custom_settings = {
        "DOWNLOAD_DELAY": 2.0,
        "ROBOTSTXT_OBEY": False,  # RSS feeds often not in robots.txt
    }

    async def start(self):
        """Generate requests for each RSS feed (async version for Scrapy 2.13+)"""
        for feed in self.rss_feeds:
            self._check_shutdown()
            yield scrapy.Request(
                url=feed["url"],
                callback=self.parse_rss,
                errback=self.errback_feed,
                meta={"feed_name": feed["name"], "feed_url": feed["url"]},
                dont_filter=True,
            )

    def errback_feed(self, failure):
        """Handle failed feed fetches"""
        feed_name = failure.request.meta.get("feed_name", "Unknown")
        self.logger.error(f"Failed to fetch RSS feed {feed_name}: {failure.value}")
        self.error_count += 1
        self.consecutive_errors += 1

        if self.consecutive_errors >= self.max_consecutive_errors:
            self.logger.error(f"Too many consecutive feed errors. Stopping.")
            raise CloseSpider("Too many consecutive errors")

    def parse_rss(self, response):
        """Parse RSS feed using feedparser"""
        self._check_shutdown()

        feed_name = response.meta["feed_name"]
        feed_url = response.meta["feed_url"]

        self.logger.info(f"Parsing RSS feed: {feed_name}")
        self._reset_consecutive_errors()  # Reset on successful feed fetch

        try:
            # Parse feed
            feed = feedparser.parse(response.text)

            if not feed.entries:
                self.logger.warning(f"No entries found in feed: {feed_name}")
                return

            self.logger.info(f"Found {len(feed.entries)} articles in {feed_name}")

            # Process each entry
            for entry in feed.entries[:100]:  # Limit to 100 most recent
                self._check_shutdown()

                try:
                    # Extract publication date
                    pub_date = None
                    if hasattr(entry, "published"):
                        try:
                            pub_date = date_parser.parse(entry.published).isoformat()
                        except:
                            pub_date = entry.published

                    # Extract author
                    author = None
                    if hasattr(entry, "author"):
                        author = entry.author

                    # Get summary/description
                    summary = ""
                    if hasattr(entry, "summary"):
                        summary = entry.summary
                    elif hasattr(entry, "description"):
                        summary = entry.description

                    # Create RSS item
                    rss_item = RSSItem(
                        url=entry.link,
                        title=entry.title,
                        summary=summary,
                        published=pub_date,
                        author=author,
                        category="Kenyan News",
                        source_name=feed_name,
                        feed_url=feed_url,
                    )

                    # Now fetch the full article
                    yield scrapy.Request(
                        url=entry.link,
                        callback=self.parse_article,
                        meta={"rss_item": rss_item},
                        errback=self.errback_article,
                    )
                except Exception as e:
                    self.logger.warning(f"Error processing entry in {feed_name}: {e}")
                    self.error_count += 1
                    continue

        except Exception as e:
            self.logger.error(f"Error parsing feed {feed_name}: {e}")
            self.error_count += 1
            self.consecutive_errors += 1

    def parse_article(self, response):
        """Parse full article from URL using enhanced extractor"""
        self._check_shutdown()

        rss_item = response.meta["rss_item"]

        try:
            # Use enhanced article extractor
            extracted = self.article_extractor.extract(response.text, url=response.url)

            # Use extracted content, fallback to RSS summary if extraction failed
            content = extracted.get("text", "")
            if not content or len(content) < 100:
                content = rss_item.get("summary", "")
                self.logger.warning(
                    f"Extraction failed for {response.url}, using RSS summary"
                )

            # Use extracted title if better than RSS title
            title = extracted.get("title", "") or rss_item["title"]

            # Use extracted author if available
            author = extracted.get("author", "") or rss_item.get("author", "")

            # Use extracted date if available
            publication_date = extracted.get("date", "") or rss_item.get(
                "published", ""
            )

            # Use extracted description if available
            summary = extracted.get("description", "") or rss_item.get("summary", "")

            self._reset_consecutive_errors()  # Reset on successful article parse

            # Create document item
            yield DocumentItem(
                url=rss_item["url"],
                title=title,
                content=content,
                content_type="html",
                category="Kenyan News",
                source_name=rss_item["source_name"],
                author=author,
                publication_date=publication_date,
                summary=summary,
                raw_html=response.text,
                status_code=response.status,
            )
        except Exception as e:
            self.logger.error(f"Error parsing article {response.url}: {e}")
            self.error_count += 1
            self.consecutive_errors += 1

            # Still try to yield minimal item
            try:
                yield DocumentItem(
                    url=rss_item["url"],
                    title=rss_item["title"],
                    content=rss_item.get("summary", ""),
                    content_type="text",
                    category="Kenyan News",
                    source_name=rss_item["source_name"],
                    author=rss_item.get("author"),
                    publication_date=rss_item.get("published"),
                    summary=rss_item.get("summary"),
                    raw_html="",
                )
            except Exception:
                pass

    def errback_article(self, failure):
        """Handle failed article fetches"""
        self.logger.error(f"Failed to fetch article: {failure.request.url}")

        # Still yield the RSS item with summary only
        rss_item = failure.request.meta.get("rss_item")
        if rss_item:
            yield DocumentItem(
                url=rss_item["url"],
                title=rss_item["title"],
                content=rss_item.get("summary", ""),
                content_type="text",
                category="Kenyan News",
                source_name=rss_item["source_name"],
                author=rss_item.get("author"),
                publication_date=rss_item.get("published"),
                summary=rss_item.get("summary"),
                raw_html="",
            )
