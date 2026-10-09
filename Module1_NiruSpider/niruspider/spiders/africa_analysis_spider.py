"""
Africa Analysis Spider - Crawls The Conversation Africa for academic/policy analysis
Provides expert analysis on Kenya, East Africa, and African development topics
"""

import scrapy
import feedparser
from datetime import datetime
from dateutil import parser as date_parser
from scrapy import signals
from scrapy.exceptions import CloseSpider
from ..items import DocumentItem


class AfricaAnalysisSpider(scrapy.Spider):
    name = "africa_analysis"

    custom_settings = {
        "DOWNLOAD_DELAY": 2.0,
        "ROBOTSTXT_OBEY": False,
        "DOWNLOAD_TIMEOUT": 45,
        "RETRY_TIMES": 4,
        "CONCURRENT_REQUESTS": 8,
        "CLOSESPIDER_PAGECOUNT": 150,
    }

    def __init__(self, *args, **kwargs):
        super(AfricaAnalysisSpider, self).__init__(*args, **kwargs)
        self.error_count = 0
        self.success_count = 0
        self.max_consecutive_errors = 10
        self.consecutive_errors = 0

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(AfricaAnalysisSpider, cls).from_crawler(crawler, *args, **kwargs)
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
            "url": "https://theconversation.com/africa/feed",
            "name": "The Conversation - Africa",
        },
    ]

    kenya_keywords = [
        "kenya",
        "kenyan",
        "nairobi",
        "ruto",
        "odinga",
        "kenyatta",
        "kenya's",
        "kenyan government",
        "kenyan economy",
        "kenyan politics",
        "east africa",
        "eac",
        "east african",
    ]

    africa_broad_topics = [
        "africa",
        "african",
        "african union",
        "au",
        "democracy",
        "governance",
        "election",
        "constitution",
        "development",
        "economic",
        "trade",
        "investment",
        "climate",
        "environment",
        "energy",
        "agriculture",
        "health",
        "education",
        "technology",
        "innovation",
        "security",
        "peace",
        "conflict",
        "human rights",
        "gender",
        "inequality",
        "poverty",
        "food security",
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
        self.logger.info(f"Parsing analysis feed: {feed_name}")

        feed = feedparser.parse(response.text)
        if not feed.entries:
            self.logger.warning(f"No entries in feed: {feed_name}")
            return

        self.logger.info(f"Found {len(feed.entries)} articles in {feed_name}")

        for entry in feed.entries[:100]:
            title = entry.title.lower() if entry.title else ""
            summary = (
                getattr(entry, "summary", "").lower()
                if hasattr(entry, "summary")
                else ""
            )
            combined = f"{title} {summary}"

            is_kenya_relevant = any(kw in combined for kw in self.kenya_keywords)
            is_africa_broad = any(
                topic in combined for topic in self.africa_broad_topics
            )

            tags = []
            if hasattr(entry, "tags"):
                tags = [
                    t.get("term", "").lower() for t in entry.tags if hasattr(t, "get")
                ]
                is_kenya_relevant = is_kenya_relevant or any("kenya" in t for t in tags)
                is_africa_broad = is_africa_broad or any(
                    ("africa" in t or "kenya" in t) for t in tags
                )

            if not (is_kenya_relevant or is_africa_broad):
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
                    "source_name": (
                        f"{feed_name} - {getattr(entry, 'source', '')}"
                        if hasattr(entry, "source")
                        else feed_name
                    ),
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
            'article p::text, .content p::text, [itemprop="articleBody"] p::text, main p::text'
        ).getall()
        content = "\n".join(p.strip() for p in content_parts if p.strip())

        if not content or len(content) < 100:
            content = rss_summary

        author = response.css(
            '.author::text, .byline::text, [rel="author"]::text'
        ).get()
        if author:
            author = author.strip()

        date_str = response.css(
            'time::attr(datetime), meta[property="article:published_time"]::attr(content)'
        ).get()
        if date_str:
            pub_date = date_str.strip()
        elif not pub_date:
            pub_date = datetime.utcnow().isoformat()

        self.success_count += 1

        yield DocumentItem(
            url=response.url,
            title=title,
            content=content,
            content_type="html",
            category="Analysis",
            source_name=source_name,
            author=author,
            publication_date=pub_date,
            summary=rss_summary,
            language="en",
            raw_html=response.text,
            status_code=response.status,
        )

    def errback_article(self, failure):
        self.logger.error(f"Failed to fetch analysis article: {failure.request.url}")
        self.error_count += 1
