"""
Fact Check Spider - Crawls Africa Check Kenya fact-checks
Provides verified claims, ratings, and evidence for fact-checking RAG queries
"""

import scrapy
import feedparser
from datetime import datetime
from dateutil import parser as date_parser
from scrapy import signals
from scrapy.exceptions import CloseSpider
from ..items import DocumentItem


class FactCheckSpider(scrapy.Spider):
    name = "fact_check"

    custom_settings = {
        "DOWNLOAD_DELAY": 2.0,
        "ROBOTSTXT_OBEY": False,
        "DOWNLOAD_TIMEOUT": 45,
        "RETRY_TIMES": 4,
        "CONCURRENT_REQUESTS": 8,
        "CLOSESPIDER_PAGECOUNT": 100,
    }

    def __init__(self, *args, **kwargs):
        super(FactCheckSpider, self).__init__(*args, **kwargs)
        self.error_count = 0
        self.success_count = 0
        self.max_consecutive_errors = 10
        self.consecutive_errors = 0

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(FactCheckSpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        crawler.signals.connect(spider.spider_error, signal=signals.spider_error)
        return spider

    def spider_closed(self, spider, reason):
        self.logger.info(
            f"Spider closed: {reason} — Success: {self.success_count}, Errors: {self.error_count}"
        )

    def spider_error(self, failure, response, spider):
        self.error_count += 1
        self.consecutive_errors += 1
        if self.consecutive_errors >= self.max_consecutive_errors:
            raise CloseSpider(f"Consecutive error limit: {self.consecutive_errors}")

    rss_feeds = [
        {
            "url": "https://africacheck.org/rss.xml",
            "name": "Africa Check - All Fact Checks",
        },
    ]

    kenya_keywords = [
        "kenya",
        "kenyan",
        "nairobi",
        "ruto",
        "odinga",
        "kenya's",
        "kenyan government",
        "kenyan shilling",
        "east africa",
        "eac",
    ]

    def start_requests(self):
        for feed in self.rss_feeds:
            yield scrapy.Request(
                url=feed["url"],
                callback=self.parse_rss,
                errback=self.errback_feed,
                meta={"feed_name": feed["name"]},
                dont_filter=True,
            )

    def errback_feed(self, failure):
        feed_name = failure.request.meta.get("feed_name", "Unknown")
        self.logger.error(f"Failed to fetch feed {feed_name}: {failure.value}")
        self.error_count += 1
        self.consecutive_errors += 1
        if self.consecutive_errors >= self.max_consecutive_errors:
            raise CloseSpider("Too many consecutive errors")

    def parse_rss(self, response):
        self.consecutive_errors = 0
        feed_name = response.meta["feed_name"]
        self.logger.info(f"Parsing fact-check feed: {feed_name}")

        feed = feedparser.parse(response.text)
        if not feed.entries:
            self.logger.warning(f"No entries in feed: {feed_name}")
            return

        self.logger.info(f"Found {len(feed.entries)} fact-checks in {feed_name}")

        for entry in feed.entries[:100]:
            title = entry.title.lower()
            summary = (
                getattr(entry, "summary", "").lower()
                if hasattr(entry, "summary")
                else ""
            )
            combined = f"{title} {summary}"

            has_kenya = any(kw in combined for kw in self.kenya_keywords)

            tags = []
            if hasattr(entry, "tags"):
                tags = [
                    t.get("term", "").lower() for t in entry.tags if hasattr(t, "get")
                ]

            is_kenya = has_kenya or any("kenya" in t for t in tags)

            if not is_kenya:
                continue

            pub_date = None
            if hasattr(entry, "published"):
                try:
                    pub_date = date_parser.parse(entry.published).isoformat()
                except Exception:
                    pub_date = entry.published

            yield scrapy.Request(
                url=entry.link,
                callback=self.parse_article,
                meta={
                    "title": entry.title,
                    "source_name": feed_name,
                    "pub_date": pub_date,
                    "summary": getattr(entry, "summary", ""),
                    "tags": tags,
                },
                errback=self.errback_article,
            )

    def parse_article(self, response):
        title = response.meta["title"]
        source_name = response.meta["source_name"]
        pub_date = response.meta.get("pub_date")
        rss_summary = response.meta.get("summary", "")

        extracted_title = response.css("h1::text").get()
        if extracted_title:
            title = extracted_title.strip()

        content_parts = response.css(
            "article p::text, .content p::text, main p::text, .field--name-body p::text"
        ).getall()
        content = "\n".join(p.strip() for p in content_parts if p.strip())

        if not content or len(content) < 100:
            content = rss_summary if rss_summary else ""

        author = response.css(
            ".author::text, .byline::text, .field--name-field-author::text"
        ).get()
        if author:
            author = author.strip()

        date_str = response.css(
            "time::attr(datetime), .date::text, .field--name-field-date-published::text"
        ).get()
        if date_str:
            pub_date = date_str.strip()
        elif not pub_date:
            pub_date = datetime.utcnow().isoformat()

        rating = response.css(
            ".rating-label::text, .fact-check-rating::text, .claim-rating::text"
        ).get()
        if rating:
            rating = rating.strip()

        claim = response.css(
            ".claim-text::text, .claim::text, .field--name-field-claim::text"
        ).get()
        if claim:
            claim = claim.strip()

        metadata_tags = ["fact-check"]
        if rating:
            metadata_tags.append(rating.lower().replace(" ", "-"))

        self.success_count += 1

        yield DocumentItem(
            url=response.url,
            title=title,
            content=content,
            content_type="html",
            category="Fact Check",
            source_name=source_name,
            author=author,
            publication_date=pub_date,
            summary=rss_summary,
            keywords=claim,
            metadata_tags=metadata_tags,
            language="en",
            raw_html=response.text,
            status_code=response.status,
        )

    def errback_article(self, failure):
        self.logger.error(f"Failed to fetch fact-check article: {failure.request.url}")
        self.error_count += 1
