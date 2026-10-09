"""
Global Trends Spider - Fetches global news relevant to Kenya
Fetches international news that is relevant to Kenya (Kenya-specific or Africa policy context)
Enhanced with robust extraction and Kenya-specific filtering
"""

import scrapy
from scrapy.exceptions import CloseSpider
import feedparser
import re
from datetime import datetime
from dateutil import parser as date_parser
from ..items import RSSItem, DocumentItem
from ..extractors import ArticleExtractor


class GlobalTrendsSpider(scrapy.Spider):
    name = "global_trends"

    def __init__(self, *args, **kwargs):
        super(GlobalTrendsSpider, self).__init__(*args, **kwargs)
        self.article_extractor = ArticleExtractor()
        self.consecutive_errors = 0
        self.max_consecutive_errors = 10
        self.total_articles_fetched = 0
        self.total_articles_skipped = 0
        self.total_articles_filtered = 0

    # Global news, geopolitics, and policy RSS feeds
    rss_feeds = [
        # Geopolitics & International Affairs
        {
            "url": "https://www.reuters.com/world/rss",
            "name": "Reuters World News",
        },
        {
            "url": "https://feeds.bbci.co.uk/news/world/rss.xml",
            "name": "BBC World News",
        },
        {
            "url": "https://www.aljazeera.com/xml/rss/all.xml",
            "name": "Al Jazeera International",
        },
        {
            "url": "https://foreignpolicy.com/feed/",
            "name": "Foreign Policy",
        },
        # International Organizations
        {
            "url": "https://news.un.org/feed/subscribe/en/news/all/rss.xml",
            "name": "United Nations News",
        },
        {
            "url": "https://www.who.int/feeds/entity/mediacentre/news/en/rss.xml",
            "name": "World Health Organization",
        },
        {
            "url": "https://www.worldbank.org/en/news/rss",
            "name": "World Bank News",
        },
        {
            "url": "https://www.imf.org/en/News/RSS",
            "name": "IMF News",
        },
        {
            "url": "https://au.int/en/rss.xml",
            "name": "African Union",
        },
        # Technology & Innovation
        {
            "url": "https://www.reuters.com/technology/rss",
            "name": "Reuters Technology",
        },
        {
            "url": "https://techcrunch.com/feed/",
            "name": "TechCrunch",
        },
        {
            "url": "https://www.technologyreview.com/feed/",
            "name": "MIT Technology Review",
        },
        # Policy & Governance
        {
            "url": "https://www.economist.com/international/rss.xml",
            "name": "The Economist",
        },
        {
            "url": "https://www.brookings.edu/feed/",
            "name": "Brookings Institution",
        },
        {
            "url": "https://www.cfr.org/rss",
            "name": "Council on Foreign Relations",
        },
        # Climate & Development
        {
            "url": "https://unfccc.int/news/rss",
            "name": "UN Climate Change",
        },
        {
            "url": "https://www.undp.org/rss.xml",
            "name": "UNDP News",
        },
    ]

    # Kenya-specific keywords (must include Kenya prominently)
    kenya_keywords = [
        "kenya",
        "kenyan",
        "nairobi",
        "mombasa",
        "kisumu",
        "nakuru",
        "kenya's",
        "kenyan government",
        "kenyan parliament",
        "kenyan law",
        "kenyan economy",
        "kenyan policy",
        "kenyan constitution",
        "kenya and",
        "kenya's",
        "in kenya",
        "to kenya",
        "from kenya",
        "kenya-",
        "kenya–",
        "kenya—",  # Various dash types
        "eldoret",
        "thika",
        "malindi",
        "kitale",
        "garissa",
        "ruto",
        "william ruto",
        "raila odinga",
        "uhuru kenyatta",
    ]

    # East Africa Community keywords (Kenya-relevant regional context)
    eac_keywords = [
        "east africa",
        "east african",
        "eac",
        "east african community",
        "tanzania",
        "uganda",
        "rwanda",
        "burundi",
        "south sudan",
        "regional integration",
        "common market",
    ]

    # African Union keywords (must be combined with Kenya)
    au_keywords = [
        "african union",
        "au summit",
        "africa and kenya",
        "kenya and africa",
        "african countries",
        "african nations",
    ]

    # Exclude patterns - Only non-Kenya/non-Africa content
    exclude_patterns = [
        # Asia (except if Kenya is mentioned)
        r"\b(?:china|chinese|beijing)\b(?!.*kenya)",
        r"\b(?:russia|russian|moscow|putin)\b(?!.*kenya)",
        r"\b(?:ukraine|ukrainian|kyiv)\b(?!.*kenya)",
        r"\b(?:israel|israeli|palestine|palestinian|gaza|west bank)\b(?!.*kenya)",
        r"\b(?:iran|iranian|tehran)\b(?!.*kenya)",
        r"\b(?:north korea|south korea|seoul|pyongyang)\b(?!.*kenya)",
        r"\b(?:japan|japanese|tokyo)\b(?!.*kenya)",
        r"\b(?:india|indian|delhi|mumbai|bangalore)\b(?!.*kenya)",
        r"\b(?:pakistan|pakistani|islamabad|karachi)\b(?!.*kenya)",
        # Americas (except if Kenya is mentioned)
        r"\b(?:brazil|brazilian|brasilia|sao paulo)\b(?!.*kenya)",
        r"\b(?:mexico|mexican|mexico city)\b(?!.*kenya)",
        r"\b(?:argentina|argentine|buenos aires)\b(?!.*kenya)",
        # Europe (except if Kenya is mentioned)
        r"\b(?:france|french|paris)\b(?!.*kenya)",
        r"\b(?:germany|german|berlin)\b(?!.*kenya)",
        r"\b(?:spain|spanish|madrid)\b(?!.*kenya)",
        r"\b(?:italy|italian|rome)\b(?!.*kenya)",
    ]

    custom_settings = {
        "DOWNLOAD_DELAY": 2.0,
        "ROBOTSTXT_OBEY": False,
    }

    async def start(self):
        """Generate requests for each RSS feed (async version for Scrapy 2.13+)"""
        for feed in self.rss_feeds:
            yield scrapy.Request(
                url=feed["url"],
                callback=self.parse_rss,
                meta={"feed_name": feed["name"], "feed_url": feed["url"]},
                dont_filter=True,
            )

    def parse_rss(self, response):
        """Parse RSS feed using feedparser"""
        feed_name = response.meta["feed_name"]
        feed_url = response.meta["feed_url"]

        self.logger.info(f"Parsing RSS feed: {feed_name}")

        # Parse feed
        feed = feedparser.parse(response.text)

        if not feed.entries:
            self.logger.warning(f"No entries found in feed: {feed_name}")
            return

        self.logger.info(f"Found {len(feed.entries)} articles in {feed_name}")

        # Process each entry
        for entry in feed.entries[:50]:  # Limit to 50 most recent
            # Filter by keywords
            title = entry.title.lower()
            summary = getattr(entry, "summary", "").lower()
            combined_text = f"{title} {summary}"

            # STRICT KENYA FILTER: Must mention Kenya explicitly
            has_kenya_keyword = any(
                keyword in combined_text for keyword in self.kenya_keywords
            )

            # OR has East Africa context (EAC neighbors)
            has_eac_context = any(
                keyword in combined_text for keyword in self.eac_keywords
            )

            # OR has AU context with Kenya mention
            has_au_context = any(
                keyword in combined_text for keyword in self.au_keywords
            )

            # Skip if no Kenya/EAC/AU relevance at all
            if not (has_kenya_keyword or has_eac_context or has_au_context):
                continue

            # If it's EAC/AU context without explicit Kenya mention, it must be governance/policy related
            if not has_kenya_keyword:
                policy_terms = [
                    "policy",
                    "treaty",
                    "agreement",
                    "trade",
                    "integration",
                    "parliament",
                    "law",
                    "regulation",
                ]
                has_policy = any(term in combined_text for term in policy_terms)
                if not has_policy:
                    continue  # Skip - not policy-relevant

            # Exclude articles about other regions (even if they mention Kenya in passing)
            exclude_count = sum(
                1
                for pattern in self.exclude_patterns
                if re.search(pattern, combined_text, re.IGNORECASE)
            )
            kenya_count = sum(
                1 for keyword in self.kenya_keywords if keyword in combined_text
            )

            # If article mentions more excluded regions than Kenya keywords, skip it
            if exclude_count > kenya_count and kenya_count < 2:
                continue

            self.logger.info(f"Kenya-relevant article: {entry.title[:70]}...")

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
            summary_text = ""
            if hasattr(entry, "summary"):
                summary_text = entry.summary
            elif hasattr(entry, "description"):
                summary_text = entry.description

            # Create RSS item
            rss_item = RSSItem(
                url=entry.link,
                title=entry.title,
                summary=summary_text,
                published=pub_date,
                author=author,
                category="Global Trend",
                source_name=feed_name,
                feed_url=feed_url,
            )

            # Fetch the full article
            yield scrapy.Request(
                url=entry.link,
                callback=self.parse_article,
                meta={"rss_item": rss_item},
                errback=self.errback_article,
            )

    def parse_article(self, response):
        """Parse full article from URL using enhanced extractor"""
        rss_item = response.meta["rss_item"]

        # Use enhanced article extractor
        extracted = self.article_extractor.extract(response.text, url=response.url)

        # Use extracted content, fallback to RSS summary if extraction failed
        content = extracted.get("text", "")
        if not content or len(content) < 100:
            content = rss_item.get("summary", "")
            self.logger.warning(
                f"Extraction failed for {response.url}, using RSS summary"
            )

        # Final Kenya relevance check on full content
        content_lower = content.lower()

        # STRICT: Must have Kenya keyword in full content (not just title/summary)
        has_kenya_in_content = any(
            keyword in content_lower for keyword in self.kenya_keywords
        )

        # OR has EAC context (East Africa regional news)
        has_eac_in_content = any(
            keyword in content_lower for keyword in self.eac_keywords
        )

        # Count Kenya mentions vs excluded region mentions
        kenya_mention_count = sum(
            content_lower.count(keyword) for keyword in self.kenya_keywords
        )
        excluded_mention_count = sum(
            1
            for pattern in self.exclude_patterns
            if re.search(pattern, content_lower, re.IGNORECASE)
        )

        # Skip if no Kenya/EAC relevance at all
        if not has_kenya_in_content and not has_eac_in_content:
            self.logger.info(
                f"Skipping - no Kenya/EAC relevance: {rss_item.get('title', '')[:70]}"
            )
            return

        # Skip if mentions excluded regions more than Kenya
        if excluded_mention_count > kenya_mention_count and kenya_mention_count < 2:
            self.logger.info(
                f"Skipping - primarily about other regions: {rss_item.get('title', '')[:70]}"
            )
            return

        # For EAC content without explicit Kenya mention, must be policy/governance related
        if has_eac_in_content and not has_kenya_in_content:
            policy_terms = [
                "policy",
                "treaty",
                "agreement",
                "trade",
                "integration",
                "parliament",
                "law",
                "regulation",
                "governance",
            ]
            has_policy = any(term in content_lower for term in policy_terms)
            if not has_policy:
                self.logger.info(
                    f"Skipping EAC article without policy relevance: {rss_item.get('title', '')[:70]}"
                )
                return

        # Use extracted title if better than RSS title
        title = extracted.get("title", "") or rss_item["title"]

        # Use extracted author if available
        author = extracted.get("author", "") or rss_item.get("author", "")

        # Use extracted date if available
        publication_date = extracted.get("date", "") or rss_item.get("published", "")

        # Use extracted description if available
        summary = extracted.get("description", "") or rss_item.get("summary", "")

        # Extract keywords/tags if available
        keywords = response.css(
            'meta[name="keywords"]::attr(content)'
        ).get() or extracted.get("tags", [])
        if isinstance(keywords, list):
            keywords = ", ".join(keywords)

        yield DocumentItem(
            url=rss_item["url"],
            title=title,
            content=content,
            content_type="html",
            category="Global Trend",
            source_name=rss_item["source_name"],
            author=author,
            publication_date=publication_date,
            summary=summary,
            keywords=keywords,
            raw_html=response.text,
            status_code=response.status,
        )

    def errback_article(self, failure):
        """Handle failed article fetches"""
        self.logger.error(f"Failed to fetch article: {failure.request.url}")
        self.consecutive_errors += 1
        self.total_articles_skipped += 1

        if self.consecutive_errors >= self.max_consecutive_errors:
            self.logger.critical(
                f"{self.consecutive_errors} consecutive errors — aborting crawl"
            )
            raise CloseSpider(
                f"Too many consecutive errors ({self.consecutive_errors})"
            )

        # Still yield the RSS item with summary only
        rss_item = failure.request.meta.get("rss_item")
        if rss_item:
            yield DocumentItem(
                url=rss_item["url"],
                title=rss_item["title"],
                content=rss_item.get("summary", ""),
                content_type="text",
                category="Global Trend",
                source_name=rss_item["source_name"],
                author=rss_item.get("author"),
                publication_date=rss_item.get("published"),
                summary=rss_item.get("summary"),
                raw_html="",
            )
