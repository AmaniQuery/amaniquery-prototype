"""
NiruSpider v2.0 - Scrapy Settings
==================================
"""

import os
from dotenv import load_dotenv

load_dotenv()

BOT_NAME = "niruspider"
SPIDER_MODULES = ["niruspider"]
NEWSPIDER_MODULE = "niruspider"

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests
CONCURRENT_REQUESTS = 4
CONCURRENT_REQUESTS_PER_DOMAIN = 2

# Configure a delay for requests for the same website
DOWNLOAD_DELAY = 2
RANDOMIZE_DOWNLOAD_DELAY = True

# Disable cookies (enabled by default)
COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
TELNETCONSOLE_ENABLED = False

# Override the default request headers:
DEFAULT_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Enable or disable spider middlewares
SPIDER_MIDDLEWARES = {
    "scrapy.spidermiddlewares.httperror.HttpErrorMiddleware": 50,
}

# Enable or disable downloader middlewares
DOWNLOADER_MIDDLEWARES = {
    "scrapy.downloadermiddlewares.useragent.UserAgentMiddleware": None,
    "scrapy.downloadermiddlewares.retry.RetryMiddleware": 90,
}

# Enable or disable extensions
EXTENSIONS = {
    "scrapy.extensions.telnet.TelnetConsole": None,
}

# Configure item pipelines
ITEM_PIPELINES = {
    "niruspider.pipelines.LLMSummaryPipeline": 300,
    "niruspider.pipelines.VectorDBPipeline": 400,
}

# Disable reactor verification for Windows compatibility
TWISTED_REACTOR = "twisted.internet.selectreactor.SelectReactor"

# Enable HTTP caching
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 86400  # 24 hours
HTTPCACHE_DIR = "httpcache"
HTTPCache_IGNORE_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408, 429]

# AutoThrottle settings
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 2
AUTOTHROTTLE_MAX_DELAY = 10
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0

# Retry settings
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408, 429]

# Log settings
LOG_LEVEL = "INFO"
LOG_FILE = "niruspider.log"

# Feed exports
FEED_EXPORT_ENCODING = "utf-8"
FEED_EXPORT_INDENT = 2

# LLM Configuration (for summary generation)
LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER", "gemini"
)  # Options: 'gemini', 'openai', 'groq'
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_API_KEY = os.getenv("GEMINI_API_KEY")  # Load from environment
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 200))

# Vector DB Configuration
VECTOR_DB = os.getenv("VECTOR_DB", "qdrant")  # Options: 'weaviate', 'qdrant'
VECTOR_DB_URL = os.getenv("QDRANT_URL")  # Cloud-based, no localhost fallback
VECTOR_DB_COLLECTION = os.getenv("VECTOR_DB_COLLECTION", "amaniquery_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
