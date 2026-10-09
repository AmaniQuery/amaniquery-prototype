"""
Example: InfoSMS Gateway Usage

This example demonstrates how to use the SMS gateway for querying AmaniQuery
via SMS (feature phone accessibility - "kabambe").
"""

import requests
import json

API_URL = "http://localhost:8000"


def preview_sms_response(query: str, language: str = "en"):
    """Preview what SMS response would be sent"""
    response = requests.get(
        f"{API_URL}/sms-query", params={"query": query, "language": language}
    )

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None


def send_test_sms(phone_number: str, message: str):
    """Send a test SMS (requires Africa's Talking setup)"""
    response = requests.post(
        f"{API_URL}/sms-send", params={"phone_number": phone_number, "message": message}
    )

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def print_sms_preview(result):
    """Print formatted SMS preview"""
    if not result:
        return

    print("\n" + "=" * 70)
    print(f"📱 SMS QUERY PREVIEW")
    print("=" * 70)

    print(f"\n❓ Query: {result['query']}")
    print(f"🌍 Language: {'English' if result['language'] == 'en' else 'Swahili'}")
    print(f"📊 Query Type: {result['query_type']}")

    print(f"\n💬 SMS RESPONSE ({result['character_count']}/160 characters):")
    print("─" * 70)
    print(f"  {result['response']}")
    print("─" * 70)

    status = "✅ FITS" if result["within_sms_limit"] else "❌ TOO LONG"
    print(f"\nStatus: {status} in single SMS")

    if result.get("sources"):
        print(f"\n📚 Sources Used ({len(result['sources'])}):")
        for i, source in enumerate(result["sources"], 1):
            print(f"  {i}. {source['source_name']}: {source['title']}")

    print("=" * 70)


def main():
    """Run SMS gateway examples"""
    print("\n📱 InfoSMS GATEWAY - Examples\n")
    print("This feature enables SMS-based queries for feature phone users (kabambe)")
    print("Maximum response: 160 characters per SMS\n")

    # Example 1: Legal question (English)
    print("\n1️⃣  Legal Question (English)")
    result = preview_sms_response("What is the Finance Bill about?", language="en")
    print_sms_preview(result)

    # Example 2: Parliament question (English)
    print("\n2️⃣  Parliament Question (English)")
    result = preview_sms_response("Who is the speaker of parliament?", language="en")
    print_sms_preview(result)

    # Example 3: News question (English)
    print("\n3️⃣  News Question (English)")
    result = preview_sms_response("Latest news on housing", language="en")
    print_sms_preview(result)

    # Example 4: Constitutional query (English)
    print("\n4️⃣  Constitutional Query (English)")
    result = preview_sms_response("Constitution Article 10", language="en")
    print_sms_preview(result)

    # Example 5: Swahili query
    print("\n5️⃣  Swahili Query")
    result = preview_sms_response("Muswada wa Fedha ni kuhusu nini?", language="sw")
    print_sms_preview(result)

    # Example 6: Short factual query
    print("\n6️⃣  Short Factual Query")
    result = preview_sms_response("How many MPs in Kenya?", language="en")
    print_sms_preview(result)

    print("\n" + "=" * 70)
    print("📱 SMS GATEWAY FEATURES")
    print("=" * 70)
    print("\n✅ Capabilities:")
    print("  • 160-character concise responses")
    print("  • English and Swahili support")
    print("  • Automatic query type detection")
    print("  • Smart context filtering")
    print("  • Feature phone accessibility (kabambe)")
    print("  • Africa's Talking integration")

    print("\n📋 Use Cases:")
    print("  • Rural areas with limited internet")
    print("  • Quick legal/constitutional lookups")
    print("  • News summaries on the go")
    print("  • Parliament information access")
    print("  • Democratizing information access")

    print("\n🔧 Setup Instructions:")
    print("  1. Sign up at https://africastalking.com")
    print("  2. Get API credentials (username + API key)")
    print("  3. Set environment variables:")
    print("     - AT_USERNAME=your_username")
    print("     - AT_API_KEY=your_api_key")
    print("  4. Configure webhook URL in Africa's Talking dashboard")
    print("  5. Webhook endpoint: POST /sms-webhook")

    print("\n💰 Cost:")
    print("  • ~KES 0.80 per SMS in Kenya")
    print("  • Sandbox mode available for testing")

    print("\n📊 Example User Flow:")
    print("  1. User sends: 'Finance Bill' to shortcode")
    print("  2. Africa's Talking → POST /sms-webhook")
    print("  3. AmaniQuery RAG pipeline processes query")
    print("  4. Response sent back via SMS in <2 seconds")
    print("  5. User receives: 'Finance Bill 2025 raises revenue...'")

    print("\n" + "=" * 70)
    print("✅ SMS Gateway examples complete!")
    print("=" * 70)

    # Optionally test sending SMS (uncomment and add your phone number)
    # print("\n📤 Testing SMS Send...")
    # result = send_test_sms("+254712345678", "Test message from AmaniQuery!")
    # if result:
    #     print(f"✅ SMS sent! Message ID: {result.get('message_id')}")


if __name__ == "__main__":
    main()
