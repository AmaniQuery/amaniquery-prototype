import feedparser
from ..common.config import settings


class RSSFetcher:
    def fetch_feeds(self):
        articles = []
        for url in settings.NEWS_RSS_FEEDS:
            print(f"Fetching RSS feed: {url}")
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    articles.append(
                        {
                            "title": entry.get("title", ""),
                            "link": entry.get("link", ""),
                            "published": entry.get("published", ""),
                            "summary": entry.get("summary", ""),
                            "source": feed.feed.get("title", url),
                        }
                    )
            except Exception as e:
                print(f"Error fetching RSS {url}: {e}")

        return articles
