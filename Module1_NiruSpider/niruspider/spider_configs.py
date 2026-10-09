"""
Spider-specific configuration settings
Allows per-spider optimization of concurrency, delays, and retry logic
"""

# Spider-specific settings override
SPIDER_CONFIGS = {
    "news_rss": {
        "CONCURRENT_REQUESTS": 30,  # Reduced from 100 — many RSS sources rate-limit
        "CONCURRENT_REQUESTS_PER_DOMAIN": 8,
        "DOWNLOAD_DELAY": 1.0,
        "RETRY_TIMES": 3,
        "DOWNLOAD_TIMEOUT": 30,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 8.0,
    },
    "global_trends": {
        "CONCURRENT_REQUESTS": 24,  # Reduced from 80 — international sources are aggressive with rate limits
        "CONCURRENT_REQUESTS_PER_DOMAIN": 6,
        "DOWNLOAD_DELAY": 1.5,
        "RETRY_TIMES": 4,
        "DOWNLOAD_TIMEOUT": 45,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 6.0,
    },
    "parliament": {
        "CONCURRENT_REQUESTS": 16,  # Government site - be polite
        "CONCURRENT_REQUESTS_PER_DOMAIN": 4,
        "DOWNLOAD_DELAY": 2.0,  # More conservative for official site
        "RETRY_TIMES": 7,  # Government sites can be flaky
        "DOWNLOAD_TIMEOUT": 60,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 4.0,
    },
    "kenya_law": {
        "CONCURRENT_REQUESTS": 16,  # Government site - be polite
        "CONCURRENT_REQUESTS_PER_DOMAIN": 4,
        "DOWNLOAD_DELAY": 2.5,
        "RETRY_TIMES": 8,  # Kenya Law can be very slow/flaky
        "DOWNLOAD_TIMEOUT": 90,  # Very slow sometimes
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 3.0,
    },
    "constitution": {
        "CONCURRENT_REQUESTS": 12,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 3,
        "DOWNLOAD_DELAY": 2.5,
        "RETRY_TIMES": 6,
        "DOWNLOAD_TIMEOUT": 60,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 3.0,
    },
    "parliament_videos": {
        "CONCURRENT_REQUESTS": 8,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "DOWNLOAD_DELAY": 3.0,  # YouTube - be very polite
        "RETRY_TIMES": 5,
        "DOWNLOAD_TIMEOUT": 60,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 2.0,
    },
    "kenya_gazette": {
        "CONCURRENT_REQUESTS": 8,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 3,
        "DOWNLOAD_DELAY": 2.0,
        "RETRY_TIMES": 5,
        "DOWNLOAD_TIMEOUT": 60,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 3.0,
    },
    "fact_check": {
        "CONCURRENT_REQUESTS": 10,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 4,
        "DOWNLOAD_DELAY": 1.5,
        "RETRY_TIMES": 4,
        "DOWNLOAD_TIMEOUT": 45,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 4.0,
    },
    "africa_analysis": {
        "CONCURRENT_REQUESTS": 10,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 4,
        "DOWNLOAD_DELAY": 1.5,
        "RETRY_TIMES": 4,
        "DOWNLOAD_TIMEOUT": 45,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 4.0,
    },
}


def get_spider_config(spider_name):
    """
    Get configuration for a specific spider

    Args:
        spider_name: Name of the spider

    Returns:
        dict: Configuration dictionary or empty dict if not found
    """
    return SPIDER_CONFIGS.get(spider_name, {})


def apply_spider_config(spider):
    """
    Apply spider-specific configuration to a spider instance

    Args:
        spider: Spider instance to configure
    """
    config = get_spider_config(spider.name)

    if not config:
        return

    # Apply custom settings from config
    if not hasattr(spider, "custom_settings"):
        spider.custom_settings = {}

    spider.custom_settings.update(config)
    spider.logger.info(f"Applied custom config for {spider.name}: {config}")
