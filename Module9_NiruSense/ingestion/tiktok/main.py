import time
import schedule
from ..common.config import settings
from ..common.storage import storage
from .scraper import TikTokScraper

scraper = TikTokScraper()


def job():
    print("Starting TikTok scrape job...")
    for hashtag in settings.TIKTOK_HASHTAGS:
        videos = scraper.scrape_hashtag(hashtag)

        if videos:
            print(f"Found {len(videos)} videos for #{hashtag}")
            for video in videos:
                storage.save_raw_data("tiktok", video)
        else:
            print(f"No videos found for #{hashtag}")


def main():
    # Schedule job every 30 minutes
    schedule.every(30).minutes.do(job)

    # Run once immediately
    job()

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
