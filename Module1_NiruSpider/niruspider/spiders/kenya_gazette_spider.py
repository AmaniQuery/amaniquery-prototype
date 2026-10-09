"""
Kenya Gazette Spider - Crawls gazettes.africa for official Kenya Gazette notices
Covers: legal notices, appointments, tenders, new legislation, constitutional petitions
"""

import scrapy
from datetime import datetime
from scrapy import signals
from scrapy.exceptions import CloseSpider
from ..items import DocumentItem


class KenyaGazetteSpider(scrapy.Spider):
    name = "kenya_gazette"
    allowed_domains = ["gazettes.africa"]

    custom_settings = {
        "DOWNLOAD_DELAY": 2.0,
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_TIMEOUT": 60,
        "RETRY_TIMES": 5,
        "CONCURRENT_REQUESTS": 8,
        "CLOSESPIDER_PAGECOUNT": 200,
    }

    def __init__(self, *args, **kwargs):
        super(KenyaGazetteSpider, self).__init__(*args, **kwargs)
        self.error_count = 0
        self.success_count = 0
        self.max_consecutive_errors = 10
        self.consecutive_errors = 0

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(KenyaGazetteSpider, cls).from_crawler(crawler, *args, **kwargs)
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

    def start_requests(self):
        yield scrapy.Request(
            url="https://gazettes.africa/gazettes/ke",
            callback=self.parse_index,
            errback=self.errback_index,
        )

    def errback_index(self, failure):
        self.logger.error(f"Failed to fetch gazette index: {failure.value}")
        self.error_count += 1

    def parse_index(self, response):
        self.consecutive_errors = 0
        self.logger.info(f"Parsing gazettes index page: {response.url}")

        gazette_links = response.css(
            'a[href*="/gazettes/ke/"][href$="/"]::attr(href)'
        ).getall()
        gazette_links.extend(
            response.css(
                'a[href*="gazettes.africa/gazettes"]:not([href*="/page/"])::attr(href)'
            ).getall()
        )

        seen = set()
        for link in gazette_links:
            href = response.urljoin(link)
            if href in seen:
                continue
            seen.add(href)
            if "/page/" not in href:
                yield scrapy.Request(
                    url=href,
                    callback=self.parse_gazette_page,
                    errback=self.errback_page,
                )

        next_page = response.css(
            'a[rel="next"]::attr(href), .pagination a:contains("Next")::attr(href), a:contains("next")::attr(href)'
        ).get()
        if next_page:
            yield scrapy.Request(
                url=response.urljoin(next_page),
                callback=self.parse_index,
                errback=self.errback_index,
            )

    def errback_page(self, failure):
        self.logger.error(f"Failed to fetch gazette page: {failure.request.url}")
        self.error_count += 1
        self.consecutive_errors += 1
        if self.consecutive_errors >= self.max_consecutive_errors:
            raise CloseSpider(f"Consecutive error limit: {self.consecutive_errors}")

    def parse_gazette_page(self, response):
        self.consecutive_errors = 0
        self.logger.info(f"Parsing gazette: {response.url}")

        title = response.css("h1::text, .page-title::text, title::text").get()
        if not title:
            title = (
                response.url.split("/")[-2]
                if response.url.endswith("/")
                else response.url.split("/")[-1]
            )

        pub_date = None
        date_text = response.css(
            "time::attr(datetime), .date::text, .published::text"
        ).get()
        if date_text:
            pub_date = date_text.strip()

        content_parts = response.css(
            "article p::text, .content p::text, main p::text"
        ).getall()
        content = "\n".join(p.strip() for p in content_parts if p.strip())

        pdf_links = response.css('a[href$=".pdf"]::attr(href)').getall()
        pdf_path = None
        for pdf_link in pdf_links:
            full_pdf_url = response.urljoin(pdf_link)
            if "kenya" in full_pdf_url.lower() or "gazette" in full_pdf_url.lower():
                pdf_path = full_pdf_url
                break
        if not pdf_path and pdf_links:
            pdf_path = response.urljoin(pdf_links[0])

        self.success_count += 1

        yield DocumentItem(
            url=response.url,
            title=title,
            content=content,
            content_type="pdf" if pdf_path else "html",
            category="Kenya Gazette",
            source_name="Gazettes Africa",
            publication_date=pub_date,
            crawl_date=datetime.utcnow().isoformat(),
            author=None,
            summary=None,
            keywords=None,
            language="en",
            raw_html=response.text,
            status_code=response.status,
            pdf_path=pdf_path,
        )
