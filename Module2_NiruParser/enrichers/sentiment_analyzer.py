"""
Sentiment Analyzer - Extract sentiment from news articles
Uses TextBlob for simple, effective sentiment analysis
"""

from typing import Dict, Optional
from loguru import logger

try:
    from textblob import TextBlob

    TEXTBLOB_AVAILABLE = True
except ImportError:
    logger.warning("TextBlob not installed. Sentiment analysis will be disabled.")
    TEXTBLOB_AVAILABLE = False


class SentimentAnalyzer:
    """Analyze sentiment of text using TextBlob"""

    def __init__(self):
        self.enabled = TEXTBLOB_AVAILABLE

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text

        Args:
            text: Text to analyze

        Returns:
            {
                "polarity": float (-1.0 to 1.0),
                "subjectivity": float (0.0 to 1.0),
                "sentiment_label": str ("positive", "negative", "neutral")
            }
        """
        if not self.enabled:
            return {"polarity": 0.0, "subjectivity": 0.0, "sentiment_label": "neutral"}

        if not text or len(text.strip()) < 10:
            return {"polarity": 0.0, "subjectivity": 0.0, "sentiment_label": "neutral"}

        try:
            # Analyze with TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1.0 to 1.0
            subjectivity = blob.sentiment.subjectivity  # 0.0 to 1.0

            # Determine label
            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"

            return {
                "polarity": round(polarity, 3),
                "subjectivity": round(subjectivity, 3),
                "sentiment_label": label,
            }

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"polarity": 0.0, "subjectivity": 0.0, "sentiment_label": "neutral"}

    def analyze_batch(self, texts: list) -> list:
        """Analyze sentiment for multiple texts"""
        return [self.analyze(text) for text in texts]

    def get_aggregate_sentiment(self, texts: list) -> Dict[str, any]:
        """
        Get aggregate sentiment from multiple texts

        Returns:
            {
                "average_polarity": float,
                "average_subjectivity": float,
                "sentiment_distribution": {
                    "positive": int,
                    "negative": int,
                    "neutral": int
                },
                "sentiment_percentages": {
                    "positive": float,
                    "negative": float,
                    "neutral": float
                }
            }
        """
        if not texts:
            return self._empty_aggregate()

        sentiments = self.analyze_batch(texts)

        # Calculate averages
        avg_polarity = sum(s["polarity"] for s in sentiments) / len(sentiments)
        avg_subjectivity = sum(s["subjectivity"] for s in sentiments) / len(sentiments)

        # Count labels
        label_counts = {
            "positive": sum(
                1 for s in sentiments if s["sentiment_label"] == "positive"
            ),
            "negative": sum(
                1 for s in sentiments if s["sentiment_label"] == "negative"
            ),
            "neutral": sum(1 for s in sentiments if s["sentiment_label"] == "neutral"),
        }

        # Calculate percentages
        total = len(sentiments)
        percentages = {
            "positive": round((label_counts["positive"] / total) * 100, 1),
            "negative": round((label_counts["negative"] / total) * 100, 1),
            "neutral": round((label_counts["neutral"] / total) * 100, 1),
        }

        return {
            "average_polarity": round(avg_polarity, 3),
            "average_subjectivity": round(avg_subjectivity, 3),
            "sentiment_distribution": label_counts,
            "sentiment_percentages": percentages,
            "total_articles": total,
        }

    def _empty_aggregate(self) -> Dict:
        """Return empty aggregate result"""
        return {
            "average_polarity": 0.0,
            "average_subjectivity": 0.0,
            "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
            "sentiment_percentages": {"positive": 0.0, "negative": 0.0, "neutral": 0.0},
            "total_articles": 0,
        }
