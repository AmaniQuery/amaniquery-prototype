"""
Test script for Kenya Law New Spider
Quick test to verify the spider works correctly
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from niruspider.spiders.kenya_law_new_spider import KenyaLawNewSpider


def test_spider():
    """Test the Kenya Law New spider with limited pages"""
    print("=" * 80)
    print("[TEST] Testing Kenya Law New Spider")
    print("=" * 80)
    print("\nThis will crawl a limited number of pages to test functionality.")
    print("Full crawl can be run with: python crawl_spider.py kenya_law_new_spider\n")

    # Get Scrapy settings
    settings = get_project_settings()

    # Override settings for testing
    settings.set("CLOSESPIDER_PAGECOUNT", 50)  # Stop after 50 pages
    settings.set("CONCURRENT_REQUESTS", 2)
    settings.set("DOWNLOAD_DELAY", 1)

    # Create crawler process
    process = CrawlerProcess(settings)

    # Add spider with limited pages
    process.crawl(KenyaLawNewSpider, max_pages=50)

    print("[START] Starting test crawl...\n")

    # Start crawling (blocking)
    process.start()

    print("\n" + "=" * 80)
    print("[OK] Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_spider()
