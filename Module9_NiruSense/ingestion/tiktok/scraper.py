from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync
import time
import random


class TikTokScraper:
    def scrape_hashtag(self, hashtag: str, limit: int = 10):
        print(f"Scraping TikTok hashtag: {hashtag}")
        videos = []
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True
            )  # Headless=False often better for TikTok but trying True first
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                viewport={"width": 1280, "height": 720},
            )
            page = context.new_page()
            stealth_sync(page)

            try:
                url = f"https://www.tiktok.com/tag/{hashtag}"
                page.goto(url, timeout=60000)
                page.wait_for_load_state("networkidle")

                # Scroll to load videos
                for _ in range(3):
                    page.evaluate("window.scrollBy(0, 1000)")
                    time.sleep(random.uniform(2, 4))

                # Extract video elements
                # Note: Selectors change often. This is a best-effort selector.
                video_elements = page.query_selector_all(
                    'div[data-e2e="search_video-item"]'
                )

                # If tag page structure is different (grid view)
                if not video_elements:
                    video_elements = page.query_selector_all(
                        'div[class*="DivItemContainer"]'
                    )

                print(f"Found {len(video_elements)} potential videos")

                for element in video_elements[:limit]:
                    try:
                        desc_el = element.query_selector(
                            'div[data-e2e="search_video-desc"]'
                        )
                        author_el = element.query_selector(
                            'div[data-e2e="search_video-author-uniqueid"]'
                        )
                        link_el = element.query_selector("a")

                        text = desc_el.inner_text() if desc_el else ""
                        author = author_el.inner_text() if author_el else ""
                        link = link_el.get_attribute("href") if link_el else ""

                        if text or link:
                            videos.append(
                                {
                                    "text": text,
                                    "author": author,
                                    "url": link,
                                    "platform": "tiktok",
                                    "hashtag": hashtag,
                                }
                            )
                    except Exception as e:
                        print(f"Error extracting video: {e}")

            except Exception as e:
                print(f"TikTok Playwright error: {e}")
            finally:
                browser.close()

        return videos
