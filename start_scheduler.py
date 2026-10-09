#!/usr/bin/env python
"""
Start the AmaniQuery Crawler Scheduler

This script starts the automatic crawler scheduling service that keeps
the vector store up to date with new legal documents and news.

Usage:
    # Start with APScheduler (no Redis required)
    python start_scheduler.py

    # Start with Celery (requires Redis)
    python start_scheduler.py --celery

    # Show status
    python start_scheduler.py --status

Schedule Overview:
    - News RSS: Every 4 hours (Kenyan news)
    - Global Trends: Every 6 hours (international news)
    - Parliament: Every 12 hours (Hansards, Bills)
    - Parliament Videos: Daily at 3 AM
    - Kenya Law: Every 2 days (comprehensive legal database)
    - Vector Store: Auto-updated after crawls
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []

    # Core dependencies
    try:
        import scrapy
    except ImportError:
        missing.append("scrapy")

    try:
        import loguru
    except ImportError:
        missing.append("loguru")

    return missing


def check_apscheduler():
    """Check if APScheduler is available"""
    try:
        import apscheduler

        return True
    except ImportError:
        return False


def check_celery():
    """Check if Celery and Redis are available"""
    try:
        import celery
        import redis

        # Try to connect to Redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = redis.from_url(redis_url)
        r.ping()
        return True
    except ImportError:
        return False
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="AmaniQuery Crawler Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--celery", action="store_true", help="Use Celery backend (requires Redis)"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show scheduler status and exit"
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--save-config", type=str, help="Save default configuration to file and exit"
    )
    parser.add_argument(
        "--trigger",
        type=str,
        choices=[
            "kenya_law",
            "global_trends",
            "parliament",
            "parliament_videos",
            "news_rss",
        ],
        help="Manually trigger a specific crawler",
    )

    args = parser.parse_args()

    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"[ERROR] Missing dependencies: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        sys.exit(1)

    from loguru import logger

    # Import scheduler components
    from Module1_NiruSpider.scheduler.scheduler_service import (
        SchedulerService,
        SchedulerConfig,
        CrawlerType,
    )

    # Save default config if requested
    if args.save_config:
        config = SchedulerConfig.default()
        config.save(Path(args.save_config))
        print(f"[OK] Default configuration saved to: {args.save_config}")
        return

    # Determine backend
    if args.celery:
        if not check_celery():
            print("[ERROR] Celery backend requires Redis. Please ensure:")
            print("   1. Redis is installed and running")
            print("   2. REDIS_URL environment variable is set")
            print(
                "   Or use the default APScheduler backend: python start_scheduler.py"
            )
            sys.exit(1)
        backend = "celery"
        print("[START] Using Celery backend (distributed)")
    else:
        if not check_apscheduler():
            print("[ERROR] APScheduler not installed")
            print("   Install with: pip install apscheduler")
            sys.exit(1)
        backend = "apscheduler"
        print("[START] Using APScheduler backend (standalone)")

    # Load configuration
    if args.config:
        config = SchedulerConfig.load(Path(args.config))
        logger.info(f"Loaded configuration from: {args.config}")
    else:
        config = SchedulerConfig.default()

    # Create service
    service = SchedulerService(config=config, backend=backend)

    # Handle different modes
    if args.status:
        if service.start():
            import time
            import json

            time.sleep(2)
            status = service.get_status()
            print("\n[STATUS] Scheduler Status:")
            print(json.dumps(status, indent=2))
            service.stop()
        return

    if args.trigger:
        print(f"[TRIGGER] Triggering crawler: {args.trigger}")
        if service.start():
            crawler_type = CrawlerType(args.trigger)
            service.trigger_crawler(crawler_type)
            print(f"[OK] Crawler {args.trigger} triggered")

            # Wait for it to start
            import time

            time.sleep(5)

            # Show status
            status = service.get_status()
            running = status.get("running_crawlers", [])
            print(f"[STATUS] Running crawlers: {running}")

            # Keep running until crawler finishes
            print("\nPress Ctrl+C to exit (crawler will continue in background)")
            try:
                while True:
                    time.sleep(10)
                    status = service.get_status()
                    running = status.get("running_crawlers", [])
                    if not running:
                        print("[OK] Crawler completed")
                        break
            except KeyboardInterrupt:
                pass
            finally:
                service.stop()
        return

    # Run scheduler
    print("\n" + "=" * 60)
    print("AmaniQuery Crawler Scheduler")
    print("=" * 60)
    print("\n[SCHEDULE] Schedule Overview:")
    for name, schedule in config.schedules.items():
        if schedule.enabled:
            print(f"   - {name}: Every {schedule.interval_hours}h")
        else:
            print(f"   - {name}: DISABLED")
    print("\n" + "=" * 60)
    print("Press Ctrl+C to stop the scheduler")
    print("=" * 60 + "\n")

    try:
        service.run_forever()
    except Exception as e:
        logger.error(f"Scheduler error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
