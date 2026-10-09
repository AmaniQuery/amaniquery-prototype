"""
Parliament Spider - Crawls parliament.go.ke for Hansards and Bills
Enhanced with:
- Robust error handling with try/except in all parse methods
- Graceful shutdown on failures
- Request timeout protection
- Error counting and automatic shutdown on too many errors
- Signal handling for clean termination
"""

import scrapy
from datetime import datetime
import re
import signal
from loguru import logger
from scrapy.exceptions import CloseSpider
from ..items import DocumentItem
from ..extractors import ArticleExtractor


class ParliamentSpider(scrapy.Spider):
    name = "parliament"
    allowed_domains = ["parliament.go.ke"]

    # Error thresholds for graceful shutdown
    MAX_CONSECUTIVE_ERRORS = 10
    MAX_TOTAL_ERRORS = 50

    def __init__(self, *args, **kwargs):
        super(ParliamentSpider, self).__init__(*args, **kwargs)
        self.article_extractor = ArticleExtractor()
        self.max_pages = 50  # Limit pages to crawl
        self.pages_crawled = 0
        self.documents_found = 0
        self.consecutive_errors = 0
        self.total_errors = 0
        self._shutdown_requested = False

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful termination"""
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            logger.info("Signal handlers configured for graceful shutdown")
        except Exception as e:
            logger.warning(f"Could not setup signal handlers: {e}")

    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        signal_name = (
            signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        )
        logger.warning(
            f"Received signal {signal_name}, initiating graceful shutdown..."
        )
        self._shutdown_requested = True
        raise CloseSpider(f"Shutdown requested via signal {signal_name}")

    def _check_shutdown(self):
        """Check if shutdown was requested"""
        if self._shutdown_requested:
            raise CloseSpider("Shutdown requested")

    def _handle_error(self, error_msg: str, response_url: str = None):
        """Centralized error handling with automatic shutdown on too many errors"""
        self.consecutive_errors += 1
        self.total_errors += 1

        url_info = f" URL: {response_url}" if response_url else ""
        logger.error(f"Error in ParliamentSpider: {error_msg}{url_info}")
        logger.warning(
            f"Error count - consecutive: {self.consecutive_errors}, total: {self.total_errors}"
        )

        if self.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
            raise CloseSpider(
                f"Too many consecutive errors ({self.consecutive_errors}). Shutting down gracefully."
            )

        if self.total_errors >= self.MAX_TOTAL_ERRORS:
            raise CloseSpider(
                f"Too many total errors ({self.total_errors}). Shutting down gracefully."
            )

    def _reset_consecutive_errors(self):
        """Reset consecutive error count on successful operation"""
        self.consecutive_errors = 0

    start_urls = [
        "https://www.parliament.go.ke/the-national-assembly/house-business/hansard",
        "https://www.parliament.go.ke/the-national-assembly/house-business/bills",
        "https://www.parliament.go.ke/documents-publications",
        "https://www.parliament.go.ke/the-national-assembly/business/budget-documents",
        "https://www.parliament.go.ke/the-national-assembly/committees",
        "https://www.parliament.go.ke/the-senate/house-business/bills",
    ]

    custom_settings = {
        "DOWNLOAD_DELAY": 2.5,
        "RETRY_TIMES": 3,
        "RETRY_HTTP_CODES": [500, 502, 503, 504, 408, 429],
        "DOWNLOAD_TIMEOUT": 60,
        "CLOSESPIDER_TIMEOUT": 1800,  # 30 minutes max runtime
        "CLOSESPIDER_ERRORCOUNT": 50,  # Close after 50 errors
        "CONCURRENT_REQUESTS": 4,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
    }

    def parse(self, response):
        """Parse listing pages with robust error handling"""
        self._check_shutdown()

        try:
            self.logger.info(f"Parsing: {response.url}")
            self.pages_crawled += 1

            # Check page limit
            if self.pages_crawled > self.max_pages:
                self.logger.info(
                    f"Reached max pages limit ({self.max_pages}). Stopping."
                )
                return

            # Special handling for budget documents page
            if "budget-documents" in response.url:
                yield from self._safe_parse_budget_documents(response)
                return

            # Look for PDF links
            pdf_links = self._extract_pdf_links(response)

            for link in pdf_links:
                self._check_shutdown()
                yield from self._process_pdf_link(response, link)

            # Look for article/document links
            yield from self._process_doc_links(response)

            # Pagination
            yield from self._process_pagination(response)

            self._reset_consecutive_errors()

        except CloseSpider:
            raise
        except Exception as e:
            self._handle_error(str(e), response.url)

    def _extract_pdf_links(self, response):
        """Extract PDF links from page with deduplication"""
        try:
            pdf_links = response.css('a[href$=".pdf"]::attr(href)').getall()
            pdf_links.extend(response.css('a[href*=".pdf"]::attr(href)').getall())
            pdf_links.extend(
                response.css('a[href*="/sites/default/files/"]::attr(href)').getall()
            )

            # Remove duplicates while preserving order
            seen = set()
            unique_pdf_links = []
            for link in pdf_links:
                if link and link not in seen:
                    seen.add(link)
                    unique_pdf_links.append(link)

            return unique_pdf_links
        except Exception as e:
            self.logger.warning(f"Error extracting PDF links: {e}")
            return []

    def _process_pdf_link(self, response, link):
        """Process a single PDF link"""
        try:
            # Make absolute URL if relative
            if link.startswith("/"):
                link = response.urljoin(link)
            elif not link.startswith("http"):
                link = response.urljoin(link)

            # Extract title
            title = self._extract_link_title(response, link)

            self.documents_found += 1
            yield response.follow(
                link,
                callback=self.parse_pdf,
                errback=self.errback_handler,
                meta={
                    "title": title.strip() if title else "Parliamentary Document",
                    "dont_retry": False,
                },
            )
        except Exception as e:
            self.logger.warning(f"Error processing PDF link {link}: {e}")

    def _extract_link_title(self, response, link):
        """Extract title for a link"""
        try:
            filename = link.split("/")[-1]
            link_selector = f'a[href*="{filename}"]'
            link_element = response.css(link_selector)

            if link_element:
                title = link_element.css("::text").get()
                if title:
                    return title

            # Use filename as fallback
            if filename.endswith(".pdf"):
                filename = filename.replace(".pdf", "")
            return filename.replace("-", " ").replace("_", " ").title()
        except Exception:
            return "Parliamentary Document"

    def _process_doc_links(self, response):
        """Process document links"""
        try:
            doc_links = response.css(
                "article a::attr(href), .document a::attr(href)"
            ).getall()
            for link in doc_links:
                if link and not link.endswith(".pdf"):
                    yield response.follow(
                        link, callback=self.parse_document, errback=self.errback_handler
                    )
        except Exception as e:
            self.logger.warning(f"Error processing doc links: {e}")

    def _process_pagination(self, response):
        """Process pagination links"""
        try:
            next_page = response.css(
                'a.next::attr(href), a[rel="next"]::attr(href)'
            ).get()
            if next_page and self.pages_crawled < self.max_pages:
                yield response.follow(
                    next_page, callback=self.parse, errback=self.errback_handler
                )
        except Exception as e:
            self.logger.warning(f"Error processing pagination: {e}")

    def _safe_parse_budget_documents(self, response):
        """Parse budget documents with error handling"""
        try:
            yield from self.parse_budget_documents(response)
        except Exception as e:
            self._handle_error(f"Error in parse_budget_documents: {e}", response.url)

    def parse_budget_documents(self, response):
        """Special parser for budget documents page"""
        self._check_shutdown()
        self.logger.info(f"Parsing budget documents page: {response.url}")

        pdf_links = self._extract_pdf_links(response)
        self.logger.info(f"Found {len(pdf_links)} PDF links on budget documents page")

        for link in pdf_links:
            self._check_shutdown()

            try:
                # Make absolute URL
                if link.startswith("/"):
                    link = response.urljoin(link)
                elif not link.startswith("http"):
                    link = response.urljoin(link)

                # Extract title with budget context
                title = self._extract_budget_title(response, link)

                self.documents_found += 1
                yield response.follow(
                    link,
                    callback=self.parse_pdf,
                    errback=self.errback_handler,
                    meta={
                        "title": title.strip() if title else "Budget Document",
                        "is_budget_doc": True,
                        "dont_retry": False,
                    },
                )
            except Exception as e:
                self.logger.warning(f"Error processing budget PDF link {link}: {e}")

        self._reset_consecutive_errors()

    def _extract_budget_title(self, response, link):
        """Extract title for budget documents"""
        try:
            filename = link.split("/")[-1]
            link_selector = f'a[href*="{filename}"]'
            link_element = response.css(link_selector).first()

            title = None
            if link_element:
                title = link_element.css("::text").get()
                if not title:
                    title = link_element.xpath("../text()").get()

            if not title:
                if filename.endswith(".pdf"):
                    filename = filename.replace(".pdf", "")
                title = filename.replace("-", " ").replace("_", " ").replace("%20", " ")
                title = " ".join(word.capitalize() for word in title.split())

            # Add budget context
            if (
                title
                and "budget" not in title.lower()
                and "finance" not in title.lower()
            ):
                if "2025" in link or "2026" in link:
                    title = f"{title} (Budget 2025-2026)"

            return title
        except Exception:
            return "Budget Document"

    def parse_pdf(self, response):
        """Handle PDF documents with error handling"""
        self._check_shutdown()

        try:
            title = response.meta.get("title", "Untitled Parliamentary Document")
            is_budget_doc = response.meta.get("is_budget_doc", False)

            # Extract date from URL or title
            pub_date = self._extract_date(response.url, title)

            # Determine category
            category = "Parliament"
            if is_budget_doc or "budget" in title.lower() or "finance" in title.lower():
                category = "Parliament - Budget Documents"

            self._reset_consecutive_errors()

            yield DocumentItem(
                url=response.url,
                title=title,
                content="",
                content_type="pdf",
                category=category,
                source_name="Parliament of Kenya",
                publication_date=pub_date,
                raw_html="",
            )

        except CloseSpider:
            raise
        except Exception as e:
            self._handle_error(f"Error parsing PDF: {e}", response.url)

    def _extract_date(self, url: str, title: str) -> str:
        """Extract publication date from URL or title"""
        try:
            date_match = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", url)
            if date_match:
                return date_match.group(1)

            year_match = re.search(r"(202[4-9])", url + " " + title)
            if year_match:
                return year_match.group(1) + "-01-01"
        except Exception:
            pass
        return None

    def parse_document(self, response):
        """Parse HTML documents with robust error handling"""
        self._check_shutdown()

        try:
            # Use enhanced article extractor
            extracted = self.article_extractor.extract(response.text, url=response.url)

            # Extract content with fallbacks
            content = extracted.get("text", "")
            if not content or len(content) < 50:
                content_parts = response.css(
                    "article::text, .content::text, main::text"
                ).getall()
                if not content_parts:
                    content_parts = response.css("p::text").getall()
                content = "\n".join([t.strip() for t in content_parts if t.strip()])

            # Extract title with fallbacks
            title = extracted.get("title", "") or (
                response.css("h1::text").get()
                or response.css("title::text").get()
                or "Untitled Document"
            )
            title = title.strip()

            # Extract date
            pub_date = extracted.get("date", "")
            if not pub_date:
                date_text = response.css(
                    ".date::text, time::text, .published::text"
                ).get()
                if date_text:
                    date_match = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", date_text)
                    pub_date = date_match.group(1) if date_match else None

            self._reset_consecutive_errors()

            yield DocumentItem(
                url=response.url,
                title=title,
                content=content,
                content_type="html",
                category="Parliament",
                source_name="Parliament of Kenya",
                author=extracted.get("author", ""),
                publication_date=pub_date,
                summary=extracted.get("description", ""),
                raw_html=response.text,
                status_code=response.status,
            )

        except CloseSpider:
            raise
        except Exception as e:
            self._handle_error(f"Error parsing document: {e}", response.url)

    def errback_handler(self, failure):
        """Handle request failures"""
        request = failure.request
        self.logger.error(f"Request failed: {request.url}")
        self.logger.error(f"Failure type: {failure.type}")
        self.logger.error(f"Failure value: {failure.value}")

        self.total_errors += 1
        self.consecutive_errors += 1

        # Check for critical failures
        if self.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
            self.logger.error(f"Too many consecutive failures. Initiating shutdown.")
            raise CloseSpider(
                f"Too many consecutive errors ({self.consecutive_errors})"
            )

    def closed(self, reason):
        """Called when spider closes - log summary"""
        self.logger.info("=" * 60)
        self.logger.info("PARLIAMENT SPIDER SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Reason for closing: {reason}")
        self.logger.info(f"Pages crawled: {self.pages_crawled}")
        self.logger.info(f"Documents found: {self.documents_found}")
        self.logger.info(f"Total errors: {self.total_errors}")
        self.logger.info(f"Consecutive errors at close: {self.consecutive_errors}")
        self.logger.info("=" * 60)
