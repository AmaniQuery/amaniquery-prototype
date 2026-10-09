"""
Constitution Spider - Crawls constitution-related content for Kenya
"""

import scrapy
from datetime import datetime
from ..items import DocumentItem


class ConstitutionSpider(scrapy.Spider):
    name = "constitution"
    allowed_domains = [
        "kenyalaw.org",
        "parliament.go.ke",
        "constitutionnet.org",
    ]

    # Start URLs - Constitution sources
    start_urls = [
        # Kenya Law Reports - Constitution
        "http://kenyalaw.org/kl/index.php?id=398",  # Constitution main page
        "http://kenyalaw.org/lex/actview.xql?actid=Const2010",  # Constitution 2010
        # Parliament - Constitutional Bills
        "https://www.parliament.go.ke/the-national-assembly/house-business/bills?field_bill_category_tid=All&field_bill_status_value=All&keys=constitution",
        # ConstitutionNet - Reliable source for full constitution
        "https://constitutionnet.org/country/kenya",
        "http://constitutionnet.org/sites/default/files/final_constitution_of_kenya_2010.pdf",  # Direct PDF link
    ]

    custom_settings = {
        "DOWNLOAD_DELAY": 2.0,  # Be polite with government sites
        "ROBOTSTXT_OBEY": True,
    }

    def parse(self, response):
        """Parse constitution-related pages"""
        self.logger.info(f"Parsing constitution page: {response.url}")

        # Handle direct PDF links
        if response.url.endswith(".pdf"):
            yield from self.parse_constitution_document(response)
            return

        # Handle different sources
        if "kenyalaw.org" in response.url:
            yield from self.parse_kenya_law(response)
        elif "klrc.go.ke" in response.url:
            yield from self.parse_klrc(response)
        elif "katiba.go.ke" in response.url:
            yield from self.parse_katiba(response)
        elif "parliament.go.ke" in response.url:
            yield from self.parse_parliament_constitution(response)
        elif "constitutionnet.org" in response.url:
            yield from self.parse_constitution_net(response)
        else:
            yield from self.parse_generic_constitution(response)

    def parse_kenya_law(self, response):
        """Parse Kenya Law Reports constitution content"""
        # Constitution links
        constitution_links = response.css('a[href*="Const"]::attr(href)').getall()
        constitution_links.extend(
            response.css('a[href*="constitution"]::attr(href)').getall()
        )

        for link in constitution_links:
            if link.startswith("/"):
                link = f"http://kenyalaw.org{link}"
            yield response.follow(link, callback=self.parse_constitution_document)

        # Constitutional amendments
        amendment_links = response.css('a[href*="amendment"]::attr(href)').getall()
        for link in amendment_links:
            if link.startswith("/"):
                link = f"http://kenyalaw.org{link}"
            yield response.follow(link, callback=self.parse_constitution_document)

    def parse_klrc(self, response):
        """Parse Kenya Law Reform Commission constitution content"""
        # Constitution sections and articles
        content_links = response.css('a[href*="constitution"]::attr(href)').getall()
        content_links.extend(
            response.css('a[href*="katiba"]::attr(href)').getall()
        )  # Swahili for constitution

        for link in content_links:
            if link.startswith("/"):
                link = f"https://klrc.go.ke{link}"
            yield response.follow(link, callback=self.parse_constitution_document)

        # Analysis and reports
        analysis_links = response.css('a[href*="analysis"]::attr(href)').getall()
        analysis_links.extend(response.css('a[href*="report"]::attr(href)').getall())

        for link in analysis_links:
            if link.startswith("/"):
                link = f"https://klrc.go.ke{link}"
            yield response.follow(link, callback=self.parse_constitution_document)

    def parse_katiba(self, response):
        """Parse Katiba Institute content"""
        # Research and analysis
        research_links = response.css('a[href*="research"]::attr(href)').getall()
        research_links.extend(response.css('a[href*="analysis"]::attr(href)').getall())
        research_links.extend(
            response.css('a[href*="publication"]::attr(href)').getall()
        )

        for link in research_links:
            if link.startswith("/"):
                link = f"https://katiba.go.ke{link}"
            yield response.follow(link, callback=self.parse_constitution_document)

        # Court cases and judgments
        case_links = response.css('a[href*="case"]::attr(href)').getall()
        case_links.extend(response.css('a[href*="judgment"]::attr(href)').getall())

        for link in case_links:
            if link.startswith("/"):
                link = f"https://katiba.go.ke{link}"
            yield response.follow(link, callback=self.parse_constitution_document)

    def parse_parliament_constitution(self, response):
        """Parse Parliament constitution-related bills"""
        # Constitutional bills
        bill_links = response.css('a[href*="bill"]::attr(href)').getall()

        for link in bill_links:
            if link.startswith("/"):
                link = f"https://www.parliament.go.ke{link}"
            # Check if it's constitution-related
            if any(
                keyword in link.lower()
                for keyword in ["constitution", "katiba", "amendment"]
            ):
                yield response.follow(link, callback=self.parse_constitution_document)

    def parse_constitution_net(self, response):
        """Parse ConstitutionNet Kenya content"""
        # Kenya-specific content - filter out mailto links
        kenya_links = response.css('a[href*="kenya"]::attr(href)').getall()

        for link in kenya_links:
            if link.startswith("mailto:"):
                continue  # Skip email links
            if link.startswith("/"):
                link = f"https://constitutionnet.org{link}"
            yield response.follow(link, callback=self.parse_constitution_document)

    def parse_generic_constitution(self, response):
        """Parse generic constitution-related pages"""
        # Look for constitution-related content
        content_selectors = [
            "article",
            ".content",
            "#content",
            "main",
            ".post-content",
        ]

        for selector in content_selectors:
            content = response.css(selector)
            if content:
                title = content.css("h1::text, h2::text, title::text").get()
                if title and any(
                    keyword in title.lower()
                    for keyword in ["constitution", "katiba", "amendment"]
                ):
                    yield from self.parse_constitution_document(response)
                    break

    def parse_constitution_document(self, response):
        """Parse individual constitution document"""
        self.logger.info(f"Parsing constitution document: {response.url}")

        # Check if this is a PDF response
        content_type_header = (
            response.headers.get("Content-Type", b"").decode("utf-8").lower()
        )
        is_pdf = (
            response.url.endswith(".pdf")
            or "pdf" in response.url
            or "application/pdf" in content_type_header
        )

        if is_pdf:
            # For PDFs, let the pipeline handle the content extraction
            title = (
                "The Constitution of Kenya, 2010"
                if "final_constitution_of_kenya_2010.pdf" in response.url
                else "Constitution Document"
            )
            full_content = ""  # PDF content will be extracted by pipeline
            content_type = "pdf"
        else:
            # Special handling for known constitution PDF
            if "final_constitution_of_kenya_2010.pdf" in response.url:
                title = "The Constitution of Kenya, 2010"
            else:
                # Extract title
                title_selectors = [
                    "h1::text",
                    "h2::text",
                    "title::text",
                    ".page-title::text",
                    ".entry-title::text",
                ]

                title = None
                for selector in title_selectors:
                    title = response.css(selector).get()
                    if title:
                        title = title.strip()
                        break

                if not title:
                    title = "Constitution Document"

            # Extract content for HTML pages
            content_selectors = [
                "article p::text",
                ".content p::text",
                "#content p::text",
                "main p::text",
                ".post-content p::text",
                ".entry-content p::text",
                "p::text",  # Fallback
            ]

            content_parts = []
            for selector in content_selectors:
                parts = response.css(selector).getall()
                if parts:
                    content_parts.extend(parts)
                    break

            full_content = "\n".join(
                [part.strip() for part in content_parts if part.strip()]
            )
            content_type = "html"

        # Determine category and source
        source_name = self.get_source_name(response.url)
        category = self.categorize_document(title, response.url)

        yield DocumentItem(
            url=response.url,
            title=title,
            content=full_content,
            content_type=content_type,
            category=category,
            source_name=source_name,
            publication_date=self.extract_date(response),
            crawl_date=datetime.now().isoformat(),
            raw_html=response.text if not is_pdf else "",
            status_code=response.status,
            language="en",  # Most constitution docs are in English
        )

    def get_source_name(self, url):
        """Get source name from URL"""
        if "kenyalaw.org" in url:
            return "Kenya Law Reports"
        elif "klrc.go.ke" in url:
            return "Kenya Law Reform Commission"
        elif "katiba.go.ke" in url:
            return "Katiba Institute"
        elif "parliament.go.ke" in url:
            return "Parliament of Kenya"
        elif "constitutionnet.org" in url:
            return "ConstitutionNet International"
        else:
            return "Constitution Source"

    def categorize_document(self, title, url):
        """Categorize constitution document"""
        title_lower = title.lower()
        url_lower = url.lower()

        if any(keyword in title_lower for keyword in ["amendment", "amend"]):
            return "Constitutional Amendment"
        elif any(keyword in title_lower for keyword in ["judgment", "case", "court"]):
            return "Constitutional Court Decision"
        elif any(
            keyword in title_lower
            for keyword in ["analysis", "review", "interpretation"]
        ):
            return "Constitutional Analysis"
        elif any(keyword in title_lower for keyword in ["bill", "draft"]):
            return "Constitutional Bill"
        elif any(keyword in url_lower for keyword in ["katiba", "constitution"]):
            return "Kenyan Constitution"
        else:
            return "Constitutional Document"

    def extract_date(self, response):
        """Extract publication date from document"""
        # Skip date extraction for PDFs as they don't have HTML content
        content_type_header = (
            response.headers.get("Content-Type", b"").decode("utf-8").lower()
        )
        if (
            response.url.endswith(".pdf")
            or "pdf" in response.url
            or "application/pdf" in content_type_header
        ):
            return None

        # Look for date patterns in the page
        date_selectors = [
            ".date::text",
            ".published::text",
            ".post-date::text",
            "time::attr(datetime)",
            "time::text",
        ]

        for selector in date_selectors:
            date_text = response.css(selector).get()
            if date_text:
                # Try to parse the date
                try:
                    # Simple date extraction - could be enhanced
                    return date_text.strip()
                except:
                    continue

        return None
