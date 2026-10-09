"""
Example: Using the Social Media Sharing Feature
"""

import requests
import json

API_BASE = "http://localhost:8000"


def example_1_query_and_share():
    """Example 1: Query AmaniQuery and share to Twitter"""
    print("=" * 80)
    print("Example 1: Query and Share to Twitter")
    print("=" * 80)

    # Step 1: Query AmaniQuery
    print("\n1️⃣ Querying AmaniQuery...")
    query_response = requests.post(
        f"{API_BASE}/query",
        json={
            "query": "What does the Kenyan Constitution say about freedom of expression?",
            "top_k": 3,
            "category": "Kenyan Law",
        },
    )

    result = query_response.json()
    print(f"✅ Got answer ({len(result['answer'])} chars)")
    print(f"📚 {len(result['sources'])} sources")

    # Step 2: Format for Twitter
    print("\n2️⃣ Formatting for Twitter...")
    share_response = requests.post(
        f"{API_BASE}/share/format",
        json={
            "answer": result["answer"],
            "sources": result["sources"],
            "platform": "twitter",
            "query": "What does the Kenyan Constitution say about freedom of expression?",
            "include_hashtags": True,
        },
    )

    formatted = share_response.json()

    if formatted["metadata"].get("format") == "thread":
        print(
            f"✅ Created Twitter thread ({formatted['metadata']['tweet_count']} tweets)"
        )
        print("\n📱 Thread Preview:")
        for i, tweet in enumerate(formatted["content"], 1):
            print(f"\n--- Tweet {i} ---")
            print(tweet)
            print(f"({len(tweet)} chars)")
    else:
        print(f"✅ Created single tweet ({formatted['character_count']} chars)")
        print("\n📱 Tweet Preview:")
        print(formatted["content"])

    # Step 3: Generate share link
    print("\n3️⃣ Generating share link...")
    link_response = requests.post(
        f"{API_BASE}/share/generate-link",
        json={"platform": "twitter", "content": formatted["content"]},
    )

    link = link_response.json()
    print(f"✅ Share URL: {link['share_url'][:80]}...")
    print(f"💡 {link['instructions']}")


def example_2_multi_platform_preview():
    """Example 2: Preview on all platforms"""
    print("\n" + "=" * 80)
    print("Example 2: Multi-Platform Preview")
    print("=" * 80)

    # Query first
    print("\n1️⃣ Querying AmaniQuery...")
    query_response = requests.post(
        f"{API_BASE}/query",
        json={
            "query": "Recent parliamentary debates on technology policy",
            "top_k": 5,
            "category": "Parliament",
        },
    )

    result = query_response.json()

    # Preview all platforms
    print("\n2️⃣ Generating previews for all platforms...")
    preview_response = requests.post(
        f"{API_BASE}/share/preview",
        json={
            "answer": result["answer"],
            "sources": result["sources"],
            "query": "Recent parliamentary debates on technology policy",
        },
    )

    previews = preview_response.json()

    # Twitter preview
    print("\n" + "─" * 80)
    print("🐦 TWITTER/X")
    print("─" * 80)
    twitter = previews["twitter"]
    if twitter.get("metadata", {}).get("format") == "thread":
        print(f"Format: Thread ({twitter['metadata']['tweet_count']} tweets)")
        print("\nFirst tweet:")
        print(twitter["content"][0])
    else:
        print("Format: Single tweet")
        print(twitter["content"])

    # LinkedIn preview
    print("\n" + "─" * 80)
    print("💼 LINKEDIN")
    print("─" * 80)
    linkedin = previews["linkedin"]
    print(f"Characters: {linkedin['character_count']}")
    print("\nPreview (first 300 chars):")
    print(linkedin["content"][:300] + "...")

    # Facebook preview
    print("\n" + "─" * 80)
    print("📘 FACEBOOK")
    print("─" * 80)
    facebook = previews["facebook"]
    print(f"Word count: {facebook['metadata']['word_count']}")
    print("\nPreview (first 300 chars):")
    print(facebook["content"][:300] + "...")


def example_3_linkedin_professional():
    """Example 3: LinkedIn professional post"""
    print("\n" + "=" * 80)
    print("Example 3: LinkedIn Professional Post")
    print("=" * 80)

    # Query about legal topic
    print("\n1️⃣ Querying about legal topic...")
    query_response = requests.post(
        f"{API_BASE}/query",
        json={
            "query": "Kenya's data protection and privacy laws",
            "top_k": 5,
            "category": "Kenyan Law",
        },
    )

    result = query_response.json()

    # Format for LinkedIn
    print("\n2️⃣ Formatting for LinkedIn...")
    share_response = requests.post(
        f"{API_BASE}/share/format",
        json={
            "answer": result["answer"],
            "sources": result["sources"],
            "platform": "linkedin",
            "query": "Kenya's data protection and privacy laws",
            "include_hashtags": True,
        },
    )

    formatted = share_response.json()

    print(f"\n✅ LinkedIn Post ({formatted['character_count']} chars)")
    print(f"📊 Optimal length: {formatted['metadata']['optimal_length']}")
    print(f"🏷️  Hashtags: {len(formatted['hashtags'])}")

    print("\n" + "─" * 80)
    print("FULL POST:")
    print("─" * 80)
    print(formatted["content"])
    print("─" * 80)


def example_4_get_platforms():
    """Example 4: Get supported platforms"""
    print("\n" + "=" * 80)
    print("Example 4: Supported Platforms")
    print("=" * 80)

    response = requests.get(f"{API_BASE}/share/platforms")
    platforms = response.json()

    print("\n📱 Supported Platforms:\n")
    for platform in platforms["platforms"]:
        print(f"• {platform['display_name']}")
        print(f"  Name: {platform['name']}")
        print(f"  Char Limit: {platform['char_limit'] or 'None'}")
        print(f"  Threads: {'Yes' if platform['supports_threads'] else 'No'}")
        print()


def main():
    """Run all examples"""
    try:
        print("\n" + "=" * 80)
        print("🔗 AmaniQuery Social Media Sharing Examples")
        print("=" * 80)

        # Run examples
        example_1_query_and_share()
        example_2_multi_platform_preview()
        example_3_linkedin_professional()
        example_4_get_platforms()

        print("\n" + "=" * 80)
        print("✅ All examples completed!")
        print("=" * 80)
        print("\n💡 Tips:")
        print("  • Use Twitter threads for longer answers")
        print("  • LinkedIn works best for professional/legal content")
        print("  • Facebook is great for engaging, accessible posts")
        print("  • Always preview before sharing!")
        print("\n" + "=" * 80)

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("   Make sure the API server is running:")
        print("   python -m Module4_NiruAPI.api")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
