"""
Kenya Law New Website Spider - Comprehensive crawler for new.kenyalaw.org
Crawls:
- Constitution of Kenya 2010
- Acts in Force (500+ chapters)
- Recent Legislation
- Subsidiary Legislation
- County Legislation
- Treaties
- Case Law (300,000+ judgments from all courts)
- Kenya Gazette (8,000+ gazettes from 1899-2025)
- Publications
- Cause Lists
- Blog Articles
"""

import scrapy
from datetime import datetime
from ..items import DocumentItem
import hashlib
import re
from urllib.parse import urljoin, urlparse, parse_qs


class KenyaLawNewSpider(scrapy.Spider):
    """
    Comprehensive spider for the new Kenya Law website (new.kenyalaw.org)
    Designed to crawl all legal resources with pagination support
    """

    name = "kenya_law"
    allowed_domains = ["new.kenyalaw.org", "kenyalaw.org"]

    def __init__(self, *args, **kwargs):
        super(KenyaLawNewSpider, self).__init__(*args, **kwargs)
        self.pages_crawled = 0
        self.max_pages = kwargs.get("max_pages", 10000)  # Configurable limit

    # Comprehensive start URLs covering all major sections
    start_urls = [
        # Constitution
        "https://new.kenyalaw.org/akn/ke/act/2010/constitution",
        # Legislation - All categories
        "https://new.kenyalaw.org/legislation/",  # Acts in force (paginated)
        "https://new.kenyalaw.org/legislation/recent",  # Recent legislation
        "https://new.kenyalaw.org/legislation/subsidiary",  # Subsidiary legislation
        "https://new.kenyalaw.org/legislation/uncommenced",  # Uncommenced legislation
        "https://new.kenyalaw.org/legislation/repealed",  # Repealed legislation
        "https://new.kenyalaw.org/legislation/all",  # All legislation
        "https://new.kenyalaw.org/legislation/counties",  # County legislation
        # Treaties
        "https://new.kenyalaw.org/taxonomy/collections/collections-treaties",
        # Case Law - All courts (paginated lists)
        "https://new.kenyalaw.org/judgments/all/",  # All courts
        # Superior Courts
        "https://new.kenyalaw.org/judgments/KESC/",  # Supreme Court
        "https://new.kenyalaw.org/judgments/KECA/",  # Court of Appeal
        "https://new.kenyalaw.org/judgments/KEHC/",  # High Court
        "https://new.kenyalaw.org/judgments/KEELRC/",  # Employment & Labour Relations
        "https://new.kenyalaw.org/judgments/KEELC/",  # Environment and Land
        "https://new.kenyalaw.org/judgments/KEIC/",  # Industrial Court
        # Subordinate Courts
        "https://new.kenyalaw.org/judgments/KEMC/",  # Magistrate's Court
        "https://new.kenyalaw.org/judgments/KEKC/",  # Kadhis Courts
        # Small Claims Court
        "https://new.kenyalaw.org/judgments/SCC/",
        # Tribunals
        "https://new.kenyalaw.org/judgments/court-class/civil-and-human-rights-tribunals/",
        "https://new.kenyalaw.org/judgments/court-class/commercial-tribunals/",
        "https://new.kenyalaw.org/judgments/court-class/environment-and-land-tribunals/",
        "https://new.kenyalaw.org/judgments/court-class/intellectual-property-tribunals/",
        "https://new.kenyalaw.org/judgments/court-class/tribunals/",
        # Regional and International Courts
        "https://new.kenyalaw.org/judgments/AfCHPR/",  # African Court on Human Rights
        "https://new.kenyalaw.org/judgments/CT/",  # Continental Court
        # Kenya Gazette - Comprehensive years coverage
        "https://new.kenyalaw.org/gazettes/",
        # Publications
        "https://new.kenyalaw.org/taxonomy/publications",
        "https://new.kenyalaw.org/taxonomy/collections",
        # Cause Lists
        "https://new.kenyalaw.org/causelists/",
        # Blog Articles
        "https://new.kenyalaw.org/articles/",
    ]

    custom_settings = {
        "DOWNLOAD_DELAY": 2,  # Polite crawling
        "RANDOMIZE_DOWNLOAD_DELAY": True,
        "CONCURRENT_REQUESTS": 4,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "ROBOTSTXT_OBEY": True,
        "USER_AGENT": "AmaniQuery/2.0 (Legal Research Bot; +https://amaniquery.org; contact@amaniquery.org)",
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 3,
        "RETRY_HTTP_CODES": [500, 502, 503, 504, 408, 429],
        "DOWNLOAD_TIMEOUT": 60,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1,
        "AUTOTHROTTLE_MAX_DELAY": 60,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 2.0,
        "HTTPCACHE_ENABLED": True,
        "HTTPCACHE_EXPIRATION_SECS": 86400,  # 24 hours
        "HTTPCACHE_DIR": "httpcache_new_kenyalaw",
        "DEPTH_LIMIT": 10,  # Prevent infinite loops
    }

    def parse(self, response):
        """Main parser - routes to appropriate handler based on URL"""
        url_path = urlparse(response.url).path

        self.logger.info(f"Parsing: {response.url}")

        # Route to appropriate parser based on URL pattern
        if "/akn/ke/act/2010/constitution" in response.url:
            yield from self.parse_constitution(response)

        elif "/legislation/" in response.url:
            yield from self.parse_legislation_list(response)

        elif "/akn/ke/act/" in response.url:
            yield from self.parse_act(response)

        elif "/judgments/" in response.url:
            yield from self.parse_judgments_list(response)

        elif "/gazettes/" in response.url:
            yield from self.parse_gazette_list(response)

        elif "/articles/" in response.url:
            yield from self.parse_articles_list(response)

        elif "/causelists/" in response.url:
            yield from self.parse_causelists(response)

        elif "/taxonomy/" in response.url:
            yield from self.parse_taxonomy(response)

        else:
            # Generic parser - extract links and content
            yield from self.parse_generic(response)

    def parse_constitution(self, response):
        """Parse the Constitution of Kenya 2010"""
        self.logger.info("Parsing Constitution of Kenya 2010")

        # Extract full text
        title = response.css("h1::text, title::text").get()
        if not title:
            title = "Constitution of Kenya, 2010"

        # Extract main content
        content_parts = response.css(
            "article ::text, main ::text, .document-content ::text"
        ).getall()
        full_text = "\n".join([t.strip() for t in content_parts if t.strip()])

        # Try to chunk by Articles
        article_pattern = r"(?:Article|ARTICLE)\s+(\d+)[.\s]+([^\n]+)"
        articles = list(re.finditer(article_pattern, full_text, re.IGNORECASE))

        if articles:
            # Parse as chunked articles
            total_chunks = len(articles)
            self.logger.info(f"Found {total_chunks} articles in Constitution")

            for idx, match in enumerate(articles):
                article_num = match.group(1)
                article_title = match.group(2).strip()

                # Extract article content (text between this and next article)
                start_pos = match.end()
                end_pos = (
                    articles[idx + 1].start()
                    if idx + 1 < len(articles)
                    else len(full_text)
                )
                article_content = full_text[start_pos:end_pos].strip()

                doc_id = hashlib.md5(
                    f"constitution_article_{article_num}".encode()
                ).hexdigest()

                yield DocumentItem(
                    doc_id=doc_id,
                    url=response.url,
                    title=f"Constitution of Kenya, 2010 - Article {article_num}: {article_title}",
                    content=f"Article {article_num}: {article_title}\n\n{article_content}",
                    content_type="html",
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    doc_type="constitution",
                    category="Kenyan Law",
                    source_name="Kenya Law",
                    publication_date="2010-08-27",
                    date_enacted="2010-08-27",
                    article_number=int(article_num),
                    crawl_date=datetime.now().isoformat(),
                    metadata_tags=["constitution", "article", "foundational"],
                    raw_html=response.text[:10000],  # Truncate for storage
                )
        else:
            # Parse as single document
            doc_id = hashlib.md5(response.url.encode()).hexdigest()

            yield DocumentItem(
                doc_id=doc_id,
                url=response.url,
                title=title,
                content=full_text,
                content_type="html",
                chunk_index=0,
                total_chunks=1,
                doc_type="constitution",
                category="Kenyan Law",
                source_name="Kenya Law",
                publication_date="2010-08-27",
                date_enacted="2010-08-27",
                crawl_date=datetime.now().isoformat(),
                metadata_tags=["constitution", "foundational"],
                raw_html=response.text[:10000],
            )

    def parse_legislation_list(self, response):
        """Parse legislation listing pages and follow individual act links"""
        # Extract all act links from the table
        act_links = response.css("table tr td a::attr(href)").getall()

        self.logger.info(f"Found {len(act_links)} legislation links on page")

        for link in act_links:
            full_url = response.urljoin(link)
            if "/akn/ke/act/" in full_url:
                yield response.follow(full_url, callback=self.parse_act)

        # Handle pagination
        next_page = response.css(
            'a:contains("Next")::attr(href), .pagination a.next::attr(href)'
        ).get()
        if next_page and self.pages_crawled < self.max_pages:
            self.pages_crawled += 1
            yield response.follow(next_page, callback=self.parse_legislation_list)

        # Handle numbered pagination links
        page_links = response.css(".pagination a::attr(href)").getall()
        for page_link in page_links:
            if self.pages_crawled < self.max_pages:
                self.pages_crawled += 1
                yield response.follow(page_link, callback=self.parse_legislation_list)

    def parse_act(self, response):
        """Parse individual Act/Bill with section-level chunking"""
        self.logger.info(f"Parsing Act: {response.url}")

        # Extract title
        title = response.css("h1::text, h2.title::text, title::text").get()
        if title:
            title = title.strip()
        else:
            # Extract from URL pattern
            url_parts = response.url.split("/")
            title = (
                url_parts[-1].replace("-", " ").title() if url_parts else "Untitled Act"
            )

        # Extract act metadata from title
        act_match = re.search(
            r"(?:Act|CAP)[.\s]+(?:No[.\s]+)?(\d+)\s+of\s+(\d{4})", title, re.IGNORECASE
        )
        act_number = act_match.group(0) if act_match else None
        year = act_match.group(2) if act_match else None

        # Extract full text
        content_parts = response.css(
            "article ::text, main ::text, .document-content ::text, .content ::text"
        ).getall()
        full_text = "\n".join([t.strip() for t in content_parts if t.strip()])

        # Try to chunk by Sections
        section_pattern = r"(?:Section|SECTION|§)\s+(\d+[A-Z]?(?:\([a-z]\))?)[.\s]+"
        sections = list(re.finditer(section_pattern, full_text, re.IGNORECASE))

        if sections and len(sections) > 1:
            # Parse as chunked sections
            total_chunks = len(sections)
            self.logger.info(f"Found {total_chunks} sections in {title}")

            for idx, match in enumerate(sections):
                section_num = match.group(1)

                # Extract section content
                start_pos = match.end()
                end_pos = (
                    sections[idx + 1].start()
                    if idx + 1 < len(sections)
                    else len(full_text)
                )
                section_content = full_text[start_pos:end_pos].strip()

                # Get section title (first line of content)
                section_lines = section_content.split("\n")
                section_title = section_lines[0][:100] if section_lines else ""

                doc_id = hashlib.md5(
                    f"{response.url}_section_{section_num}".encode()
                ).hexdigest()

                yield DocumentItem(
                    doc_id=doc_id,
                    url=response.url,
                    title=f"{title} - Section {section_num}",
                    content=f"Section {section_num}\n\n{section_content}",
                    content_type="html",
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    doc_type="act",
                    category="Kenyan Law",
                    source_name="Kenya Law",
                    publication_date=year,
                    date_enacted=year,
                    crawl_date=datetime.now().isoformat(),
                    act_number=act_number,
                    section_number=section_num,
                    clause_text=section_content[:500],
                    metadata_tags=["act", "legislation", "section"],
                    raw_html=response.text[:5000],
                )
        else:
            # Parse as single document
            doc_id = hashlib.md5(response.url.encode()).hexdigest()

            yield DocumentItem(
                doc_id=doc_id,
                url=response.url,
                title=title,
                content=full_text,
                content_type="html",
                chunk_index=0,
                total_chunks=1,
                doc_type="act",
                category="Kenyan Law",
                source_name="Kenya Law",
                publication_date=year,
                date_enacted=year,
                crawl_date=datetime.now().isoformat(),
                act_number=act_number,
                metadata_tags=["act", "legislation"],
                raw_html=response.text[:10000],
            )

    def parse_judgments_list(self, response):
        """Parse court judgments listing pages"""
        # Extract judgment links
        judgment_links = response.css(
            "table tbody tr td a::attr(href), .judgment-list a::attr(href)"
        ).getall()

        self.logger.info(f"Found {len(judgment_links)} judgment links")

        for link in judgment_links:
            full_url = response.urljoin(link)
            # Follow judgment links
            if any(
                court in full_url
                for court in [
                    "/KESC/",
                    "/KECA/",
                    "/KEHC/",
                    "/KEELRC/",
                    "/KEELC/",
                    "/KEIC/",
                    "/KEMC/",
                    "/KEKC/",
                    "/SCC/",
                ]
            ):
                yield response.follow(full_url, callback=self.parse_judgment)

        # Handle pagination
        next_page = response.css(
            'a:contains("Next")::attr(href), .pagination a.next::attr(href)'
        ).get()
        if next_page and self.pages_crawled < self.max_pages:
            self.pages_crawled += 1
            yield response.follow(next_page, callback=self.parse_judgments_list)

        # Page numbers
        page_links = response.css(".pagination a::attr(href)").getall()
        for page_link in page_links[:20]:  # Limit concurrent pages
            if self.pages_crawled < self.max_pages:
                self.pages_crawled += 1
                yield response.follow(page_link, callback=self.parse_judgments_list)

    def parse_judgment(self, response):
        """Parse individual court judgment"""
        self.logger.info(f"Parsing Judgment: {response.url}")

        # Extract title/citation
        title = response.css("h1::text, h2.case-title::text, title::text").get()
        if title:
            title = title.strip()
        else:
            title = "Court Judgment"

        # Extract case citation
        citation_match = re.search(r"\[(\d{4})\]\s+\w+\s+\d+", title)
        citation = citation_match.group(0) if citation_match else None

        # Extract court from URL
        url_parts = urlparse(response.url).path.split("/")
        court_code = None
        for part in url_parts:
            if part in [
                "KESC",
                "KECA",
                "KEHC",
                "KEELRC",
                "KEELC",
                "KEIC",
                "KEMC",
                "KEKC",
                "SCC",
            ]:
                court_code = part
                break

        court_names = {
            "KESC": "Supreme Court",
            "KECA": "Court of Appeal",
            "KEHC": "High Court",
            "KEELRC": "Employment and Labour Relations Court",
            "KEELC": "Environment and Land Court",
            "KEIC": "Industrial Court",
            "KEMC": "Magistrate's Court",
            "KEKC": "Kadhis Court",
            "SCC": "Small Claims Court",
        }
        court_name = court_names.get(court_code, "Unknown Court")

        # Extract date
        date_str = response.css(".judgment-date::text, .date::text").get()
        publication_date = self._parse_date(date_str) if date_str else None

        # Extract full judgment text
        content_parts = response.css(
            "article ::text, main ::text, .judgment-content ::text, .document-content ::text"
        ).getall()
        full_text = "\n".join([t.strip() for t in content_parts if t.strip()])

        doc_id = hashlib.md5(response.url.encode()).hexdigest()

        yield DocumentItem(
            doc_id=doc_id,
            url=response.url,
            title=title,
            content=full_text,
            content_type="html",
            chunk_index=0,
            total_chunks=1,
            doc_type="judgment",
            category=f"Case Law - {court_name}",
            source_name="Kenya Law",
            publication_date=publication_date,
            crawl_date=datetime.now().isoformat(),
            act_number=citation,  # Store citation in act_number field
            metadata_tags=[
                "judgment",
                "case_law",
                court_code.lower() if court_code else "court",
            ],
            raw_html=response.text[:10000],
        )

    def parse_gazette_list(self, response):
        """Parse Kenya Gazette listing pages"""
        # Extract gazette links by year
        gazette_links = response.css('a[href*="/gazettes/"]::attr(href)').getall()

        for link in gazette_links:
            full_url = response.urljoin(link)
            # Follow year pages
            if re.search(r"/gazettes/\d{4}", full_url):
                yield response.follow(full_url, callback=self.parse_gazette_year)
            # Follow individual gazette documents
            elif re.search(r"/gazettes/\d{4}/\d+", full_url):
                yield response.follow(full_url, callback=self.parse_gazette_document)

    def parse_gazette_year(self, response):
        """Parse gazette year page (e.g., gazettes/2025)"""
        # Extract individual gazette links
        gazette_doc_links = response.css('a[href*="/gazettes/"]::attr(href)').getall()

        for link in gazette_doc_links:
            full_url = response.urljoin(link)
            if re.search(r"/gazettes/\d{4}/\d+", full_url):
                yield response.follow(full_url, callback=self.parse_gazette_document)

    def parse_gazette_document(self, response):
        """Parse individual Kenya Gazette document"""
        self.logger.info(f"Parsing Gazette: {response.url}")

        title = response.css("h1::text, title::text").get()
        if title:
            title = title.strip()
        else:
            title = "Kenya Gazette Notice"

        # Extract year from URL
        year_match = re.search(r"/gazettes/(\d{4})/", response.url)
        year = year_match.group(1) if year_match else None

        # Extract content
        content_parts = response.css(
            "article ::text, main ::text, .gazette-content ::text"
        ).getall()
        full_text = "\n".join([t.strip() for t in content_parts if t.strip()])

        doc_id = hashlib.md5(response.url.encode()).hexdigest()

        yield DocumentItem(
            doc_id=doc_id,
            url=response.url,
            title=title,
            content=full_text,
            content_type="html",
            chunk_index=0,
            total_chunks=1,
            doc_type="gazette",
            category="Kenya Gazette",
            source_name="Kenya Law",
            publication_date=year,
            crawl_date=datetime.now().isoformat(),
            metadata_tags=["gazette", "official_notice"],
            raw_html=response.text[:10000],
        )

    def parse_articles_list(self, response):
        """Parse blog articles listing"""
        article_links = response.css(
            "article a::attr(href), .article-list a::attr(href)"
        ).getall()

        for link in article_links:
            full_url = response.urljoin(link)
            if "/articles/" in full_url:
                yield response.follow(full_url, callback=self.parse_article)

        # Pagination
        next_page = response.css('a:contains("Next")::attr(href)').get()
        if next_page and self.pages_crawled < self.max_pages:
            self.pages_crawled += 1
            yield response.follow(next_page, callback=self.parse_articles_list)

    def parse_article(self, response):
        """Parse individual blog article"""
        title = response.css("h1::text, article h2::text").get()
        if title:
            title = title.strip()
        else:
            title = "Kenya Law Article"

        date_str = response.css(
            ".article-date::text, .published::text, time::text"
        ).get()
        publication_date = self._parse_date(date_str) if date_str else None

        content_parts = response.css("article ::text, .article-content ::text").getall()
        full_text = "\n".join([t.strip() for t in content_parts if t.strip()])

        doc_id = hashlib.md5(response.url.encode()).hexdigest()

        yield DocumentItem(
            doc_id=doc_id,
            url=response.url,
            title=title,
            content=full_text,
            content_type="html",
            chunk_index=0,
            total_chunks=1,
            doc_type="article",
            category="Kenya Law Blog",
            source_name="Kenya Law",
            publication_date=publication_date,
            crawl_date=datetime.now().isoformat(),
            metadata_tags=["article", "blog", "legal_commentary"],
            raw_html=response.text[:10000],
        )

    def parse_causelists(self, response):
        """Parse cause lists"""
        causelist_links = response.css('a[href*="/causelists/"]::attr(href)').getall()

        for link in causelist_links:
            full_url = response.urljoin(link)
            yield response.follow(full_url, callback=self.parse_causelist_document)

    def parse_causelist_document(self, response):
        """Parse individual cause list document"""
        title = response.css("h1::text").get() or "Cause List"

        content_parts = response.css("article ::text, main ::text").getall()
        full_text = "\n".join([t.strip() for t in content_parts if t.strip()])

        doc_id = hashlib.md5(response.url.encode()).hexdigest()

        yield DocumentItem(
            doc_id=doc_id,
            url=response.url,
            title=title,
            content=full_text,
            content_type="html",
            doc_type="causelist",
            category="Cause Lists",
            source_name="Kenya Law",
            crawl_date=datetime.now().isoformat(),
            metadata_tags=["causelist", "court_schedule"],
        )

    def parse_taxonomy(self, response):
        """Parse taxonomy/collection pages (publications, treaties, etc.)"""
        # Extract document links
        doc_links = response.css(
            'a[href*="/akn/"], a[href*="/taxonomy/"]::attr(href)'
        ).getall()

        for link in doc_links:
            full_url = response.urljoin(link)
            yield response.follow(full_url, callback=self.parse_generic)

        # Pagination
        next_page = response.css('a:contains("Next")::attr(href)').get()
        if next_page:
            yield response.follow(next_page, callback=self.parse_taxonomy)

    def parse_generic(self, response):
        """Generic parser for other document types"""
        title = response.css("h1::text, title::text").get() or "Kenya Law Document"

        content_parts = response.css(
            "article ::text, main ::text, .content ::text"
        ).getall()
        full_text = "\n".join([t.strip() for t in content_parts if t.strip()])

        if len(full_text) < 100:
            # Skip if no meaningful content
            return

        doc_id = hashlib.md5(response.url.encode()).hexdigest()

        # Determine doc type from URL
        doc_type = "document"
        if "/treaty/" in response.url or "/treaties/" in response.url:
            doc_type = "treaty"
        elif "/publication/" in response.url:
            doc_type = "publication"

        yield DocumentItem(
            doc_id=doc_id,
            url=response.url,
            title=title.strip(),
            content=full_text,
            content_type="html",
            doc_type=doc_type,
            category="Kenyan Law",
            source_name="Kenya Law",
            crawl_date=datetime.now().isoformat(),
            metadata_tags=["legal_resource", doc_type],
            raw_html=response.text[:10000],
        )

    def _parse_date(self, date_str):
        """Parse various date formats to ISO format"""
        if not date_str:
            return None

        from dateutil import parser

        try:
            # Clean up the date string
            date_str = date_str.strip()
            dt = parser.parse(date_str, fuzzy=True)
            return dt.date().isoformat()
        except:
            # Try extracting year at minimum
            year_match = re.search(r"(\d{4})", date_str)
            if year_match:
                return year_match.group(1)
            return None
