"""
Example: Public Sentiment Gauge

This example demonstrates how to use the sentiment analysis feature to track
public sentiment on topics from news coverage.
"""

import requests
import json

API_URL = "http://localhost:8000"


def check_sentiment(topic: str, category: str = None, days: int = 30):
    """Query the sentiment endpoint"""
    params = {"topic": topic, "days": days}
    if category:
        params["category"] = category

    response = requests.get(f"{API_URL}/sentiment", params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def print_sentiment_summary(result):
    """Print a formatted sentiment summary"""
    if not result:
        return

    print("\n" + "=" * 70)
    print(f"📊 SENTIMENT ANALYSIS: {result['topic']}")
    print("=" * 70)

    print(f"\n📅 Time Period: Last {result['time_period_days']} days")
    print(f"📰 Articles Analyzed: {result['total_articles']}")

    if result.get("category_filter"):
        print(f"🔍 Category Filter: {result['category_filter']}")

    print("\n📈 SENTIMENT BREAKDOWN:")
    percentages = result["sentiment_percentages"]
    distribution = result["sentiment_distribution"]

    # Visual bar representation
    bar_length = 50

    # Positive
    positive_bar = "█" * int(percentages["positive"] / 100 * bar_length)
    print(
        f"  ✅ Positive: {percentages['positive']}% ({distribution['positive']} articles)"
    )
    print(f"     [{positive_bar:<{bar_length}}]")

    # Neutral
    neutral_bar = "█" * int(percentages["neutral"] / 100 * bar_length)
    print(
        f"  ⚪ Neutral:  {percentages['neutral']}% ({distribution['neutral']} articles)"
    )
    print(f"     [{neutral_bar:<{bar_length}}]")

    # Negative
    negative_bar = "█" * int(percentages["negative"] / 100 * bar_length)
    print(
        f"  ❌ Negative: {percentages['negative']}% ({distribution['negative']} articles)"
    )
    print(f"     [{negative_bar:<{bar_length}}]")

    print(f"\n📊 SCORES:")
    print(f"  Polarity: {result['average_polarity']:.3f} (range: -1.0 to 1.0)")
    print(f"  Subjectivity: {result['average_subjectivity']:.3f} (range: 0.0 to 1.0)")

    # Interpretation
    polarity = result["average_polarity"]
    if polarity > 0.1:
        tone = "Generally POSITIVE 😊"
    elif polarity < -0.1:
        tone = "Generally NEGATIVE 😟"
    else:
        tone = "Generally NEUTRAL 😐"

    print(f"\n💡 Overall Tone: {tone}")
    print("=" * 70)


def main():
    """Run sentiment analysis examples"""
    print("\n🎯 PUBLIC SENTIMENT GAUGE - Examples\n")

    # Example 1: Finance Bill
    print("\n1️⃣  Analyzing: Finance Bill (Last 30 days)")
    result = check_sentiment("Finance Bill", days=30)
    print_sentiment_summary(result)

    # Example 2: Housing Levy (Kenyan News only)
    print("\n2️⃣  Analyzing: Housing Levy (Kenyan News, Last 60 days)")
    result = check_sentiment("Housing Levy", category="Kenyan News", days=60)
    print_sentiment_summary(result)

    # Example 3: Climate Policy (Global only)
    print("\n3️⃣  Analyzing: Climate Policy (Global Trends, Last 90 days)")
    result = check_sentiment("Climate Change", category="Global Trend", days=90)
    print_sentiment_summary(result)

    # Example 4: Healthcare
    print("\n4️⃣  Analyzing: Healthcare Reform (Last 30 days)")
    result = check_sentiment("Healthcare Reform", days=30)
    print_sentiment_summary(result)

    # Example 5: Economic Policy
    print("\n5️⃣  Analyzing: Economic Policy (Last 45 days)")
    result = check_sentiment("Economic Policy", days=45)
    print_sentiment_summary(result)

    print("\n" + "=" * 70)
    print("✅ Sentiment analysis examples complete!")
    print("=" * 70)
    print("\nUSE CASES:")
    print("  • Track public reaction to legislation")
    print("  • Monitor news tone on policies")
    print("  • Identify controversial topics (high negative sentiment)")
    print("  • Compare Kenyan vs Global coverage sentiment")
    print("=" * 70)


if __name__ == "__main__":
    main()
