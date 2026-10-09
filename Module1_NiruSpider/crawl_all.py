"""
Run all spiders sequentially
"""

import os
import sys
from pathlib import Path

# Handle reactor installation properly for cross-platform compatibility
import platform

try:
    import asyncio
    from twisted.internet import asyncioreactor

    if "twisted.internet.reactor" not in sys.modules:
        asyncioreactor.install()
except Exception as e:
    print(f"Warning: Could not install AsyncioSelectorReactor: {e}")
    # Fallback to default reactor if AsyncioSelectorReactor fails
    pass

# Now import Scrapy after reactor handling
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

# Monkey patch reactor verification to avoid Windows compatibility issues
from scrapy.utils.reactor import verify_installed_reactor


def patched_verify_installed_reactor(reactor_class):
    """Skip reactor verification on Windows to avoid compatibility issues"""
    pass


verify_installed_reactor.__code__ = patched_verify_installed_reactor.__code__

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from niruspider.spiders.kenya_law_new_spider import KenyaLawNewSpider
from niruspider.spiders.parliament_spider import ParliamentSpider
from niruspider.spiders.news_rss_spider import NewsRSSSpider
from niruspider.spiders.global_trends_spider import GlobalTrendsSpider
from niruspider.spiders.parliament_video_spider import ParliamentVideoSpider


def main():
    """Run all spiders"""
    print("=" * 60)
    print("[START] Starting AmaniQuery Data Crawl")
    print("=" * 60)

    # Get Scrapy settings
    settings = get_project_settings()

    # Create crawler process
    process = CrawlerProcess(settings)

    # Add all spiders
    spiders = [
        ("Kenya Law", KenyaLawNewSpider),
        ("Parliament", ParliamentSpider),
        ("Parliament Videos", ParliamentVideoSpider),
        ("Kenyan News (RSS)", NewsRSSSpider),
        ("Global Trends (RSS)", GlobalTrendsSpider),
    ]

    for name, spider in spiders:
        print(f"\n[QUEUE] Queuing spider: {name}")
        process.crawl(spider)

    print("\n[START] Starting crawl process...\n")

    # Start crawling (blocking)
    process.start()

    print("\n" + "=" * 60)
    print("[DONE] Crawl complete!")
    print("[INFO] Data saved to: ../data/raw/")
    print("=" * 60)


if __name__ == "__main__":
    main()
