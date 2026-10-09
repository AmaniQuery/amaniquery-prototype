import time
import schedule
from ..common.config import settings
from ..common.storage import storage
from .nitter_client import NitterClient
from .playwright_fallback import TwitterPlaywrightScraper

nitter = NitterClient()
playwright_scraper = TwitterPlaywrightScraper()


def job():
    print("Starting Twitter scrape job...")
    for keyword in settings.TWITTER_KEYWORDS:
        print(f"Scraping keyword: {keyword}")

        # 1. Try Nitter
        tweets = nitter.search(keyword)

        # 2. Fallback to Playwright
        if not tweets:
            tweets = playwright_scraper.search(keyword)

        if tweets:
            print(f"Found {len(tweets)} tweets for {keyword}")
            for tweet in tweets:
                # Save to storage (MinIO + Redis)
                storage.save_raw_data("twitter", tweet)
        else:
            print(f"No tweets found for {keyword}")


def main():
    # Schedule job every 10 minutes
    schedule.every(10).minutes.do(job)

    # Run once immediately
    job()

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
