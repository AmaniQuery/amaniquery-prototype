"""
Example: Using AmaniQuery API
"""

import requests
import json


def query_amaniquery(question, category=None, top_k=5):
    """Query the AmaniQuery API"""

    url = "http://localhost:8000/query"

    payload = {
        "query": question,
        "top_k": top_k,
        "include_sources": True,
    }

    if category:
        payload["category"] = category

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Status {response.status_code}: {response.text}"}


def main():
    """Example queries"""

    print("=" * 80)
    print("🇰🇪 AmaniQuery - Example Queries")
    print("=" * 80)

    # Example 1: Constitutional query
    print("\n📜 Query 1: Constitutional Law")
    print("-" * 80)

    result = query_amaniquery(
        "What does the Kenyan Constitution say about freedom of expression?",
        category="Kenyan Law",
    )

    print(f"\n✨ Answer:\n{result['answer']}\n")
    print(f"⏱️  Query time: {result['query_time']:.2f}s")
    print(f"📚 Sources used: {result['retrieved_chunks']}")

    if result.get("sources"):
        print("\n📎 Sources:")
        for i, source in enumerate(result["sources"][:3], 1):
            print(f"\n  {i}. {source['title']}")
            print(f"     {source['source_name']} | {source['category']}")
            print(f"     {source['url'][:80]}...")

    # Example 2: Parliamentary query
    print("\n" + "=" * 80)
    print("\n🏛️  Query 2: Parliamentary Proceedings")
    print("-" * 80)

    result = query_amaniquery(
        "What are the recent debates in Parliament about finance?",
        category="Parliament",
    )

    print(f"\n✨ Answer:\n{result['answer']}\n")
    print(f"⏱️  Query time: {result['query_time']:.2f}s")

    # Example 3: News query
    print("\n" + "=" * 80)
    print("\n📰 Query 3: Current News")
    print("-" * 80)

    result = query_amaniquery(
        "What is the latest news about technology policy in Kenya?",
        category="Kenyan News",
        top_k=10,
    )

    print(f"\n✨ Answer:\n{result['answer']}\n")
    print(f"⏱️  Query time: {result['query_time']:.2f}s")

    # Example 4: General query (all categories)
    print("\n" + "=" * 80)
    print("\n🔍 Query 4: General Search")
    print("-" * 80)

    result = query_amaniquery(
        "How is Kenya addressing artificial intelligence regulation?", top_k=8
    )

    print(f"\n✨ Answer:\n{result['answer']}\n")
    print(f"⏱️  Query time: {result['query_time']:.2f}s")

    if result.get("sources"):
        print("\n📊 Sources by category:")
        categories = {}
        for source in result["sources"]:
            cat = source["category"]
            categories[cat] = categories.get(cat, 0) + 1

        for cat, count in categories.items():
            print(f"   - {cat}: {count}")

    print("\n" + "=" * 80)
    print("✅ Examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API")
        print("   Make sure the API server is running:")
        print("   python -m Module4_NiruAPI.api")
    except Exception as e:
        print(f"❌ Error: {e}")
