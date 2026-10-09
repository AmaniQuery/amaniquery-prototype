"""
NiruSpider - Scrapy-based data items
"""

import scrapy


class DocumentItem(scrapy.Item):
    """Base item for all crawled documents with comprehensive metadata"""

    # ========================================================================
    # CORE FIELDS
    # ========================================================================
    doc_id = scrapy.Field()  # Unique hash ID
    url = scrapy.Field()
    title = scrapy.Field()
    content = scrapy.Field()  # Full text or chunk
    summary = scrapy.Field()  # Brief summary/description
    content_type = scrapy.Field()  # html, pdf, text
    chunk_index = scrapy.Field()  # Position in document (0 for full doc)
    total_chunks = scrapy.Field()  # Total chunks in document

    # ========================================================================
    # DOCUMENT METADATA
    # ========================================================================
    doc_type = (
        scrapy.Field()
    )  # "act", "bill", "hansard", "judgment", "constitution", "county_law"
    category = scrapy.Field()  # Kenyan Law, Parliament, Kenyan News, Global Trend
    source_name = scrapy.Field()
    author = scrapy.Field()
    publication_date = scrapy.Field()  # ISO format: 2024-06-25
    crawl_date = scrapy.Field()

    # ========================================================================
    # LEGAL METADATA (for Acts, Bills, Constitution)
    # ========================================================================
    bill_stage = (
        scrapy.Field()
    )  # "Introduced", "Committee", "Third Reading", "Assented"
    act_number = scrapy.Field()  # e.g., "Act No. 7 of 2023"
    section_number = scrapy.Field()  # e.g., "3(b)"
    article_number = scrapy.Field()  # e.g., 201 (for Constitution)
    clause_text = scrapy.Field()  # Verbatim statutory text
    date_enacted = scrapy.Field()  # When became law (ISO format)
    date_passed = scrapy.Field()  # When Parliament passed (ISO format)

    # ========================================================================
    # PARLIAMENTARY METADATA (for Hansard)
    # ========================================================================
    speaking_mp = scrapy.Field()  # MP name
    constituency = scrapy.Field()  # MP constituency
    committee = scrapy.Field()  # Parliamentary committee
    session_date = scrapy.Field()  # Session date (ISO format)
    hansard_page = scrapy.Field()  # Page number

    # ========================================================================
    # COUNTY METADATA
    # ========================================================================
    county_name = scrapy.Field()  # e.g., "Nairobi"
    ward = scrapy.Field()  # Ward name

    # ========================================================================
    # DATA/STATISTICS
    # ========================================================================
    has_tables = scrapy.Field()  # Boolean: contains data tables
    tables = scrapy.Field()  # List of extracted tables
    statistics = scrapy.Field()  # Dict of key statistics
    voting_record = scrapy.Field()  # Dict with vote tallies

    # ========================================================================
    # ENHANCEMENT FIELDS
    # ========================================================================
    summary_plain_english = scrapy.Field()  # LLM-generated plain English summary
    metadata_tags = scrapy.Field()  # List: ["explainer", "summary", "analysis"]
    keywords = scrapy.Field()  # Extracted keywords
    language = scrapy.Field()  # Detected language
    quality_score = scrapy.Field()  # Quality score from QualityScoringPipeline (0-1)
    quality_breakdown = scrapy.Field()  # Detailed quality scoring breakdown

    # ========================================================================
    # TECHNICAL FIELDS
    # ========================================================================
    raw_html = scrapy.Field()
    pdf_path = scrapy.Field()
    status_code = scrapy.Field()


class RSSItem(scrapy.Item):
    """Item for RSS feed entries"""

    url = scrapy.Field()
    title = scrapy.Field()
    summary = scrapy.Field()
    published = scrapy.Field()
    author = scrapy.Field()
    category = scrapy.Field()
    source_name = scrapy.Field()
    feed_url = scrapy.Field()

    # To be populated after fetching full article
    full_content = scrapy.Field()
    crawl_date = scrapy.Field()
