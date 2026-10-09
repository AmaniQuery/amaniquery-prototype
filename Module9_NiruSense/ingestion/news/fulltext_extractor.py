from newspaper import Article


class FullTextExtractor:
    def extract(self, url: str):
        try:
            article = Article(url)
            article.download()
            article.parse()

            return {
                "text": article.text,
                "authors": article.authors,
                "publish_date": (
                    str(article.publish_date) if article.publish_date else None
                ),
                "top_image": article.top_image,
            }
        except Exception as e:
            print(f"Error extracting full text from {url}: {e}")
            return None
