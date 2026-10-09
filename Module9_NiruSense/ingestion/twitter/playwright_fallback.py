from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync
import time
import random


class TwitterPlaywrightScraper:
    def search(self, query: str):
        print("Falling back to Playwright for Twitter...")
        tweets = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
            page = context.new_page()
            stealth_sync(page)

            try:
                # Twitter search URL
                url = f"https://twitter.com/search?q={query}&src=typed_query&f=live"
                page.goto(url, timeout=30000)
                page.wait_for_selector('article[data-testid="tweet"]', timeout=10000)

                # Scroll a bit to load more
                for _ in range(3):
                    page.evaluate("window.scrollBy(0, 1000)")
                    time.sleep(2)

                # Extract tweets
                tweet_elements = page.query_selector_all('article[data-testid="tweet"]')
                for element in tweet_elements:
                    text_el = element.query_selector('div[data-testid="tweetText"]')
                    if text_el:
                        tweets.append(
                            {
                                "text": text_el.inner_text(),
                                "platform": "twitter_playwright",
                                "id": str(
                                    random.randint(100000, 999999)
                                ),  # Mock ID as extracting real ID is complex via DOM
                            }
                        )
            except Exception as e:
                print(f"Playwright error: {e}")
            finally:
                browser.close()

        return tweets
