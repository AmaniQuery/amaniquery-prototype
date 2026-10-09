import time
import schedule
from ..common.storage import storage
from .rss_fetcher import RSSFetcher
from .fulltext_extractor import FullTextExtractor

rss = RSSFetcher()
extractor = FullTextExtractor()


def job():
    print("Starting News scrape job...")
    articles = rss.fetch_feeds()
    print(f"Found {len(articles)} articles from RSS feeds")

    for article in articles:
        # Check if we already have this article (deduplication logic would go here)
        # For now, we just process everything.

        print(f"Processing: {article['title']}")
        full_content = extractor.extract(article["link"])

        if full_content:
            # Merge RSS metadata with full content
            enriched_article = {**article, **full_content, "platform": "news"}
            storage.save_raw_data("news", enriched_article)
        else:
            print(f"Failed to extract content for {article['link']}")


def main():
    # Schedule job every 1 hour
    schedule.every(1).hours.do(job)

    # Run once immediately
    job()

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
