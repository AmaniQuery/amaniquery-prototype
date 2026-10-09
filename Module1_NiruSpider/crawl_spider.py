"""
Run individual spider with timeout protection and graceful shutdown
"""

import os
import sys
import signal
import threading
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

# Default timeout: 30 minutes
DEFAULT_TIMEOUT = 1800


class SpiderRunner:
    """Spider runner with timeout and graceful shutdown support"""

    def __init__(self, spider_name: str, timeout: int = DEFAULT_TIMEOUT):
        self.spider_name = spider_name
        self.timeout = timeout
        self.process = None
        self.timed_out = False
        self._shutdown_event = threading.Event()

    def _timeout_handler(self):
        """Handle timeout - stop the crawler gracefully"""
        if not self._shutdown_event.is_set():
            print(
                f"\n[TIMEOUT] Timeout reached ({self.timeout}s). Stopping spider gracefully..."
            )
            self.timed_out = True
            if self.process:
                try:
                    # Try to stop the reactor gracefully
                    from twisted.internet import reactor

                    if reactor.running:
                        reactor.callFromThread(reactor.stop)
                except Exception as e:
                    print(f"Error stopping reactor: {e}")

    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        signal_name = (
            signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        )
        print(f"\n[STOP] Received {signal_name}. Stopping spider gracefully...")
        self._shutdown_event.set()
        if self.process:
            try:
                from twisted.internet import reactor

                if reactor.running:
                    reactor.callFromThread(reactor.stop)
            except Exception as e:
                print(f"Error handling signal: {e}")

    def run(self) -> bool:
        """Run the spider with timeout protection"""
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Map spider names to classes
        spider_mapping = {
            "kenya_law": (
                "Kenya Law",
                "niruspider.spiders.kenya_law_new_spider.KenyaLawNewSpider",
            ),
            "parliament": (
                "Parliament",
                "niruspider.spiders.parliament_spider.ParliamentSpider",
            ),
            "news_rss": (
                "Kenyan News (RSS)",
                "niruspider.spiders.news_rss_spider.NewsRSSSpider",
            ),
            "global_trends": (
                "Global Trends (RSS)",
                "niruspider.spiders.global_trends_spider.GlobalTrendsSpider",
            ),
            "parliament_videos": (
                "Parliament Videos",
                "niruspider.spiders.parliament_video_spider.ParliamentVideoSpider",
            ),
        }

        if self.spider_name not in spider_mapping:
            print(f"[ERROR] Unknown spider: {self.spider_name}")
            return False

        display_name, spider_class_path = spider_mapping[self.spider_name]

        try:
            # Import the spider class dynamically
            module_path, class_name = spider_class_path.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            spider_class = getattr(module, class_name)

            # Get Scrapy settings
            settings = get_project_settings()

            # Add close on timeout
            settings.set("CLOSESPIDER_TIMEOUT", self.timeout)

            # Create crawler process
            self.process = CrawlerProcess(settings)

            # Setup timeout timer
            timer = threading.Timer(self.timeout, self._timeout_handler)
            timer.daemon = True
            timer.start()

            print(f"[START] Starting spider: {self.spider_name}")
            print(
                f"[INFO] Timeout set to: {self.timeout}s ({self.timeout // 60} minutes)"
            )
            print(f"[CRAWL] Starting crawl for {display_name}...\n")

            # Add spider to process
            self.process.crawl(spider_class)

            # Start crawling (blocking)
            self.process.start()

            # Cancel timer if we finished before timeout
            timer.cancel()
            self._shutdown_event.set()

            if self.timed_out:
                print(f"\n[WARNING] {display_name} crawl stopped due to timeout!")
                return False
            else:
                print(f"\n[SUCCESS] {display_name} crawl complete!")
                return True

        except KeyboardInterrupt:
            print(f"\n[STOP] {display_name} crawl interrupted by user")
            return False
        except Exception as e:
            print(f"[ERROR] Error running spider {self.spider_name}: {e}")
            import traceback

            traceback.print_exc()
            return False
        finally:
            self._shutdown_event.set()


def run_spider(spider_name: str, timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Run a specific spider with timeout"""
    runner = SpiderRunner(spider_name, timeout)
    return runner.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AmaniQuery spiders")
    parser.add_argument("spider_name", help="Name of the spider to run")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds (default: {DEFAULT_TIMEOUT}s = {DEFAULT_TIMEOUT // 60} minutes)",
    )

    args = parser.parse_args()

    if args.spider_name == "help":
        print("Available spiders:")
        print("  - kenya_law (new.kenyalaw.org - comprehensive legal database)")
        print("  - parliament (parliament.go.ke - Hansards, Bills)")
        print("  - news_rss (Kenyan news RSS feeds)")
        print("  - global_trends (Global news RSS feeds)")
        print("  - parliament_videos (YouTube - Parliament videos)")
        sys.exit(0)

    success = run_spider(args.spider_name, args.timeout)
    sys.exit(0 if success else 1)
