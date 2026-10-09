"""
NiruSpider Settings
"""

import os
from pathlib import Path

BOT_NAME = "niruspider"

SPIDER_MODULES = ["niruspider.spiders"]
NEWSPIDER_MODULE = "niruspider.spiders"

# Crawl responsibly
ROBOTSTXT_OBEY = True
USER_AGENT = "AmaniQuery/1.0 (+https://github.com/amaniquery; contact@amaniquery.ke)"

# Performance settings - Balanced for polite, efficient crawling
CONCURRENT_REQUESTS = 24  # Reduced from 64 to avoid triggering rate limits
CONCURRENT_REQUESTS_PER_DOMAIN = 8  # Reduced from 16 to be polite to sources
DOWNLOAD_DELAY = 1.0  # Increased from 0.5 for more polite crawling
RANDOMIZE_DOWNLOAD_DELAY = True

# AutoThrottle (dynamic delay adjustment) - Polite, adaptive
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1.0
AUTOTHROTTLE_MAX_DELAY = 10
AUTOTHROTTLE_TARGET_CONCURRENCY = 8.0
AUTOTHROTTLE_DEBUG = False  # Set to True for debugging

# Caching (for development)
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 86400  # 24 hours
HTTPCACHE_DIR = "httpcache"

# Retry settings - Enhanced for reliability
RETRY_TIMES = 5  # Increased from 3 for better resilience
RETRY_HTTP_CODES = [
    500,
    502,
    503,
    504,
    408,
    429,
]  # 403 removed — Forbidden means blocked, retrying is futile
RETRY_PRIORITY_ADJUST = -1  # Lower priority for retries

# Download timeout
DOWNLOAD_TIMEOUT = 45  # Increased from 30 for slower sources

# Enable pipelines
ITEM_PIPELINES = {
    "niruspider.pipelines.DeduplicationPipeline": 50,  # Run first to filter duplicates
    "niruspider.pipelines.DataValidationPipeline": 100,
    "niruspider.pipelines.QualityScoringPipeline": 110,  # Enabled - filters low-quality articles at min_quality_score=0.3
    "niruspider.pipelines.VectorStorePipeline": 150,  # Disabled - processing done separately via process_all.py
    "niruspider.pipelines.PDFDownloadPipeline": 200,
    "niruspider.pipelines.FileStoragePipeline": 300,
}

# Quality scoring settings
MIN_QUALITY_SCORE = 0.3  # Lowered from 0.6 to allow more articles through

# Output directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
LOG_DATEFORMAT = "%Y-%m-%d %H:%M:%S"

# PDF handling
FILES_STORE = str(RAW_DATA_PATH / "pdfs")
MEDIA_ALLOW_REDIRECTS = True

# Request headers
DEFAULT_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Middlewares
DOWNLOADER_MIDDLEWARES = {
    "niruspider.middlewares.PoliteDelayMiddleware": 543,
}

# Rate limiting settings
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DEFAULT_RATE_LIMIT = 2.0  # Default requests per second per domain

# Feed exports
FEEDS = {
    str(RAW_DATA_PATH / "%(name)s_%(time)s.jsonl"): {
        "format": "jsonlines",
        "encoding": "utf8",
        "store_empty": False,
        "overwrite": False,
    }
}
