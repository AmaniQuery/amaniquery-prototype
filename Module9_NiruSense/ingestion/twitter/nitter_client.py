import requests
import random
import time
from ..common.config import settings


class NitterClient:
    def __init__(self):
        self.instances = [
            "https://nitter.net",
            "https://nitter.cz",
            "https://nitter.it",
            "https://nitter.privacydev.net",
            "https://nitter.poast.org",
            "https://nitter.lucabased.xyz",
        ]
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        ]

    def get_random_instance(self):
        return random.choice(self.instances)

    def search(self, query: str, limit: int = 20):
        """
        Searches for tweets using a random Nitter instance.
        """
        retries = 3
        for _ in range(retries):
            instance = self.get_random_instance()
            url = f"{instance}/search"
            params = {"f": "tweets", "q": query, "limit": limit}
            headers = {"User-Agent": random.choice(self.user_agents)}

            try:
                print(f"Trying Nitter instance: {instance}")
                response = requests.get(url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    return self._parse_nitter_html(response.text, instance)
                else:
                    print(f"Instance {instance} failed with {response.status_code}")
            except Exception as e:
                print(f"Error connecting to {instance}: {e}")

            time.sleep(1)

        return None  # All retries failed

    def _parse_nitter_html(self, html: str, instance_url: str):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        tweets = []

        timeline_items = soup.find_all(class_="timeline-item")

        for item in timeline_items:
            try:
                # Skip if it's a "Load more" button or empty
                if "show-more" in item.get("class", []):
                    continue

                tweet_content = item.find(class_="tweet-content")
                if not tweet_content:
                    continue

                text = tweet_content.get_text(strip=True)

                # Extract ID and Link
                link_tag = item.find(class_="tweet-link")
                tweet_link = link_tag["href"] if link_tag else ""
                tweet_id = (
                    tweet_link.split("/")[-1].split("#")[0] if tweet_link else "unknown"
                )

                # Extract Author
                fullname_tag = item.find(class_="fullname")
                username_tag = item.find(class_="username")
                author = fullname_tag.get_text(strip=True) if fullname_tag else ""
                username = username_tag.get_text(strip=True) if username_tag else ""

                # Extract Date
                date_tag = item.find(class_="tweet-date")
                date_str = (
                    date_tag.find("a")["title"]
                    if date_tag and date_tag.find("a")
                    else ""
                )

                # Extract Stats
                stats = item.find(class_="tweet-stats")
                likes = 0
                retweets = 0
                if stats:
                    stat_icons = stats.find_all(class_="icon-container")
                    for stat in stat_icons:
                        icon = stat.find(class_="icon-heart")
                        if icon:
                            likes = self._parse_stat(stat.get_text(strip=True))
                        icon = stat.find(class_="icon-retweet")
                        if icon:
                            retweets = self._parse_stat(stat.get_text(strip=True))

                tweets.append(
                    {
                        "id": tweet_id,
                        "text": text,
                        "author": author,
                        "username": username,
                        "date": date_str,
                        "url": f"{instance_url}{tweet_link}",
                        "likes": likes,
                        "retweets": retweets,
                        "platform": "twitter_nitter",
                    }
                )
            except Exception as e:
                print(f"Error parsing tweet item: {e}")
                continue

        return tweets

    def _parse_stat(self, stat_str: str) -> int:
        if not stat_str:
            return 0
        stat_str = stat_str.replace(",", "")
        if "K" in stat_str:
            return int(float(stat_str.replace("K", "")) * 1000)
        if "M" in stat_str:
            return int(float(stat_str.replace("M", "")) * 1000000)
        try:
            return int(stat_str)
        except:
            return 0
