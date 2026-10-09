"""
Example: Constitutional Alignment Analysis
Demonstrates the dual-retrieval RAG workflow for bill-constitution comparison
"""

import requests
import json

API_BASE = "http://localhost:8000"


def example_1_finance_bill_housing_levy():
    """
    Example 1: Analyze Finance Bill housing levy against Constitution
    """
    print("=" * 80)
    print("Example 1: Finance Bill Housing Levy - Constitutional Alignment")
    print("=" * 80)

    query = "How does the new Finance Bill's housing levy align with the constitution?"

    print(f"\n📝 Query: {query}\n")

    response = requests.post(
        f"{API_BASE}/alignment-check",
        json={
            "query": query,
            "bill_top_k": 3,
            "constitution_top_k": 3,
            "temperature": 0.3,  # Lower temperature for more factual analysis
            "max_tokens": 2000,
        },
    )

    if response.status_code == 200:
        result = response.json()

        print("🏛️ CONSTITUTIONAL ALIGNMENT ANALYSIS")
        print("=" * 80)
        print(result["analysis"])
        print("\n" + "=" * 80)

        # Show metadata
        metadata = result["metadata"]
        print("\n📊 Analysis Metadata:")
        print(f"  • Bill Identified: {metadata['bill_name']}")
        print(f"  • Legal Concepts: {', '.join(metadata['legal_concepts'])}")
        print(f"  • Analysis Type: {metadata['analysis_type']}")
        print(f"  • Bill Chunks: {metadata['bill_chunks_count']}")
        print(f"  • Constitution Chunks: {metadata['constitution_chunks_count']}")
        print(f"  • Query Time: {result.get('query_time', 0):.2f}s")

        # Show Bill context
        print("\n📜 Bill Context Used:")
        for i, ctx in enumerate(result["bill_context"], 1):
            print(f"\n  [{i}] {ctx['title']}")
            print(f"      Clause {ctx['clause_number']}: {ctx['subject']}")
            print(f"      {ctx['text'][:150]}...")

        # Show Constitution context
        print("\n📖 Constitutional Articles Referenced:")
        for i, ctx in enumerate(result["constitution_context"], 1):
            print(f"\n  [{i}] Article {ctx['article_number']}: {ctx['article_title']}")
            print(f"      {ctx['text'][:150]}...")

    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)


def example_2_quick_check():
    """
    Example 2: Quick alignment check
    """
    print("\n" + "=" * 80)
    print("Example 2: Quick Alignment Check")
    print("=" * 80)

    response = requests.post(
        f"{API_BASE}/alignment-quick-check",
        params={
            "bill_name": "Finance Bill 2025",
            "constitutional_topic": "taxation and revenue",
        },
    )

    if response.status_code == 200:
        result = response.json()

        print("\n⚡ QUICK ALIGNMENT CHECK")
        print("=" * 80)
        print(result["analysis"])
        print("=" * 80)
    else:
        print(f"❌ Error: {response.status_code}")


def example_3_data_protection():
    """
    Example 3: Data Protection Act vs Constitutional Privacy Rights
    """
    print("\n" + "=" * 80)
    print("Example 3: Data Protection Act - Privacy Rights")
    print("=" * 80)

    query = "Does the Data Protection Act comply with constitutional privacy rights?"

    response = requests.post(
        f"{API_BASE}/alignment-check",
        json={"query": query, "bill_top_k": 4, "constitution_top_k": 4},
    )

    if response.status_code == 200:
        result = response.json()
        print("\n" + result["analysis"])
    else:
        print(f"❌ Error: {response.status_code}")


def example_4_taxation_analysis():
    """
    Example 4: Complex taxation query
    """
    print("\n" + "=" * 80)
    print("Example 4: Taxation Measures Constitutional Analysis")
    print("=" * 80)

    query = """
    What does the Constitution say about the new taxation measures 
    in the Finance Bill, particularly regarding the right to property 
    and economic rights?
    """

    response = requests.post(f"{API_BASE}/alignment-check", json={"query": query})

    if response.status_code == 200:
        result = response.json()

        print("\n📊 Multi-Concept Analysis:")
        print(f"Concepts identified: {', '.join(result['metadata']['legal_concepts'])}")
        print("\n" + "=" * 80)
        print(result["analysis"])
        print("=" * 80)


def example_5_compare_responses():
    """
    Example 5: Compare regular query vs alignment analysis
    """
    print("\n" + "=" * 80)
    print("Example 5: Regular Query vs Alignment Analysis")
    print("=" * 80)

    question = "Finance Bill housing levy and constitution"

    # Regular query
    print("\n1️⃣ REGULAR QUERY:")
    print("-" * 80)
    regular = requests.post(f"{API_BASE}/query", json={"query": question})
    if regular.status_code == 200:
        print(regular.json()["answer"][:500] + "...")

    # Alignment analysis
    print("\n2️⃣ ALIGNMENT ANALYSIS:")
    print("-" * 80)
    alignment = requests.post(f"{API_BASE}/alignment-check", json={"query": question})
    if alignment.status_code == 200:
        print(alignment.json()["analysis"][:500] + "...")

    print("\n💡 Notice the difference:")
    print("   • Regular: General answer from mixed sources")
    print("   • Alignment: Structured comparison with Bill vs Constitution")


def main():
    """Run all examples"""
    try:
        print("\n" + "=" * 80)
        print("🏛️ Constitutional Alignment Module - Examples")
        print("=" * 80)
        print("\nThese examples demonstrate AmaniQuery's unique dual-retrieval")
        print("RAG workflow for constitutional alignment analysis.")
        print("\n" + "=" * 80)

        # Run examples
        example_1_finance_bill_housing_levy()

        input("\n\nPress Enter to continue to Example 2...")
        example_2_quick_check()

        input("\n\nPress Enter to continue to Example 3...")
        example_3_data_protection()

        input("\n\nPress Enter to continue to Example 4...")
        example_4_taxation_analysis()

        input("\n\nPress Enter to continue to Example 5...")
        example_5_compare_responses()

        print("\n" + "=" * 80)
        print("✅ All examples completed!")
        print("=" * 80)

        print("\n💡 Key Takeaways:")
        print("   • Dual-retrieval: Separate searches for Bill + Constitution")
        print("   • Structured prompts: Forces LLM to compare, not judge")
        print("   • Granular metadata: Article numbers, clause numbers, subjects")
        print("   • Citation-heavy: Every claim backed by source")
        print("   • Objective analysis: Highlights alignment & tensions")
        print("\n" + "=" * 80)

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("   Make sure the API server is running:")
        print("   python -m Module4_NiruAPI.api")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
