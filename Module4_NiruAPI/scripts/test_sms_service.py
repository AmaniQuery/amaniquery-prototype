#!/usr/bin/env python3
"""
Test script for Africa's Talking SMS Service
Tests SMS sending, phone formatting, account balance, and more
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Module4_NiruAPI.sms_service import AfricasTalkingSMSService
from loguru import logger
import argparse
from datetime import datetime


def test_service_initialization():
    """Test service initialization"""
    print("\n" + "=" * 60)
    print("Test 1: Service Initialization")
    print("=" * 60)

    try:
        service = AfricasTalkingSMSService()

        print(f"✓ Username: {service.username}")
        print(
            f"✓ API Key: {'*' * (len(service.api_key) - 4) + service.api_key[-4:] if service.api_key else 'Not set'}"
        )
        print(f"✓ Service Available: {service.available}")

        if not service.available:
            print(
                "❌ Service not available - check credentials or install africastalking package"
            )
            return False, None

        print("✅ Service initialized successfully")
        return True, service

    except Exception as e:
        print(f"❌ Error initializing service: {e}")
        return False, None


def test_phone_formatting(service: AfricasTalkingSMSService):
    """Test phone number formatting"""
    print("\n" + "=" * 60)
    print("Test 2: Phone Number Formatting")
    print("=" * 60)

    test_cases = [
        ("+254712345678", "+254712345678"),  # Already formatted
        ("254712345678", "+254712345678"),  # Missing +
        ("0712345678", "+254712345678"),  # Local format
        ("712345678", "+254712345678"),  # Without prefix
        ("1 234 567 890", "+2541234567890"),  # With spaces
        ("(254) 712-345-678", "+254712345678"),  # With special chars
    ]

    all_passed = True
    for input_phone, expected in test_cases:
        result = service.format_kenyan_phone(input_phone)
        status = "✅" if result == expected else "❌"
        print(
            f"{status} Input: {input_phone:20} -> Output: {result:20} (Expected: {expected})"
        )
        if result != expected:
            all_passed = False

    return all_passed


def test_account_balance(service: AfricasTalkingSMSService):
    """Test account balance retrieval"""
    print("\n" + "=" * 60)
    print("Test 3: Account Balance")
    print("=" * 60)

    try:
        balance_info = service.get_account_balance()

        if "error" in balance_info:
            print(f"❌ Error fetching balance: {balance_info['error']}")
            return False

        balance = balance_info.get("balance", "N/A")
        currency = balance_info.get("currency", "KES")

        print(f"✓ Balance: {balance} {currency}")
        print("✅ Account balance retrieved successfully")
        return True

    except Exception as e:
        print(f"❌ Error fetching balance: {e}")
        return False


def test_sms_parsing(service: AfricasTalkingSMSService):
    """Test incoming SMS parsing"""
    print("\n" + "=" * 60)
    print("Test 4: Incoming SMS Parsing")
    print("=" * 60)

    test_webhook = {
        "from": "+254712345678",
        "to": "+254800000000",
        "text": "Hello, this is a test message",
        "date": "2024-01-01T12:00:00Z",
        "id": "test-message-id-123",
        "linkId": "test-link-id-456",
        "networkCode": "63902",
    }

    try:
        parsed = service.parse_incoming_sms(test_webhook)

        print(f"✓ From: {parsed.get('from')}")
        print(f"✓ To: {parsed.get('to')}")
        print(f"✓ Text: {parsed.get('text')}")
        print(f"✓ Date: {parsed.get('date')}")
        print(f"✓ ID: {parsed.get('id')}")
        print(f"✓ Link ID: {parsed.get('linkId')}")
        print(f"✓ Network Code: {parsed.get('networkCode')}")

        if "error" in parsed:
            print(f"❌ Error parsing: {parsed['error']}")
            return False

        print("✅ SMS parsing works correctly")
        return True

    except Exception as e:
        print(f"❌ Error parsing SMS: {e}")
        return False


def test_send_sms(
    service: AfricasTalkingSMSService, phone_number: str, message: str = None
):
    """Test sending SMS"""
    print("\n" + "=" * 60)
    print("Test 5: Send SMS")
    print("=" * 60)

    if not message:
        message = (
            f"Test SMS from AmaniQuery - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    # Format phone number
    formatted_phone = service.format_kenyan_phone(phone_number)
    print(f"✓ Original phone: {phone_number}")
    print(f"✓ Formatted phone: {formatted_phone}")
    print(f"✓ Message: {message}")
    print(f"✓ Message length: {len(message)} characters")

    try:
        result = service.send_sms(formatted_phone, message)

        if result.get("success"):
            print("✅ SMS sent successfully!")
            print(f"  - Message ID: {result.get('message_id')}")
            print(f"  - Cost: {result.get('cost')}")
            print(f"  - Phone: {result.get('phone_number')}")
            return True
        else:
            print(f"❌ Failed to send SMS: {result.get('error')}")
            if "response" in result:
                print(f"  Response: {result['response']}")
            return False

    except Exception as e:
        print(f"❌ Error sending SMS: {e}")
        logger.exception(e)
        return False


def test_multi_part_sms(service: AfricasTalkingSMSService, phone_number: str):
    """Test sending multi-part SMS"""
    print("\n" + "=" * 60)
    print("Test 6: Multi-Part SMS")
    print("=" * 60)

    messages = [
        "Part 1: This is the first part of a multi-part message.",
        "Part 2: This is the second part.",
        "Part 3: This is the final part.",
    ]

    formatted_phone = service.format_kenyan_phone(phone_number)
    print(f"✓ Phone: {formatted_phone}")
    print(f"✓ Number of parts: {len(messages)}")

    try:
        result = service.send_multi_part_sms(formatted_phone, messages)

        if result.get("success"):
            print("✅ All parts sent successfully!")
            print(
                f"  - Parts sent: {result.get('parts_sent')}/{result.get('total_parts')}"
            )
            for part_result in result.get("results", []):
                status = "✅" if part_result.get("success") else "❌"
                print(
                    f"  {status} Part {part_result.get('part')}: {part_result.get('message_id', 'N/A')}"
                )
            return True
        else:
            print(f"❌ Failed to send all parts")
            print(
                f"  - Parts sent: {result.get('parts_sent')}/{result.get('total_parts')}"
            )
            return False

    except Exception as e:
        print(f"❌ Error sending multi-part SMS: {e}")
        logger.exception(e)
        return False


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test Africa's Talking SMS Service")
    parser.add_argument(
        "--phone",
        type=str,
        help="Phone number to send test SMS to (format: +254XXXXXXXXX or 0712345678)",
    )
    parser.add_argument(
        "--message",
        type=str,
        help="Custom message to send (default: auto-generated test message)",
    )
    parser.add_argument(
        "--skip-send",
        action="store_true",
        help="Skip actual SMS sending tests (to avoid charges)",
    )
    parser.add_argument(
        "--skip-multipart", action="store_true", help="Skip multi-part SMS test"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Africa's Talking SMS Service Test Suite")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check environment variables
    print("\nEnvironment Check:")
    at_username = os.getenv("AT_USERNAME")
    at_api_key = os.getenv("AT_API_KEY")

    if at_username:
        print(f"✓ AT_USERNAME: {at_username}")
    else:
        print("⚠️  AT_USERNAME not set (will use default: sandbox)")

    if at_api_key:
        print(f"✓ AT_API_KEY: {'*' * (len(at_api_key) - 4) + at_api_key[-4:]}")
    else:
        print("❌ AT_API_KEY not set!")
        print("   Please set it: export AT_API_KEY='your_api_key'")
        return

    # Run tests
    results = {}

    # Test 1: Initialization
    init_success, service = test_service_initialization()
    results["initialization"] = init_success

    if not init_success or not service:
        print("\n❌ Cannot continue tests - service initialization failed")
        return

    # Test 2: Phone formatting
    results["phone_formatting"] = test_phone_formatting(service)

    # Test 3: Account balance
    results["account_balance"] = test_account_balance(service)

    # Test 4: SMS parsing
    results["sms_parsing"] = test_sms_parsing(service)

    # Test 5: Send SMS (optional)
    if not args.skip_send:
        if args.phone:
            results["send_sms"] = test_send_sms(service, args.phone, args.message)
        else:
            print("\n" + "=" * 60)
            print("Test 5: Send SMS - SKIPPED")
            print("=" * 60)
            print("⚠️  No phone number provided. Use --phone to test SMS sending")
            print("   Example: python test_sms_service.py --phone +254712345678")
            results["send_sms"] = None
    else:
        print("\n" + "=" * 60)
        print("Test 5: Send SMS - SKIPPED (--skip-send flag)")
        print("=" * 60)
        results["send_sms"] = None

    # Test 6: Multi-part SMS (optional)
    if not args.skip_multipart and not args.skip_send:
        if args.phone:
            results["multipart_sms"] = test_multi_part_sms(service, args.phone)
        else:
            print("\n" + "=" * 60)
            print("Test 6: Multi-Part SMS - SKIPPED")
            print("=" * 60)
            print("⚠️  No phone number provided")
            results["multipart_sms"] = None
    else:
        print("\n" + "=" * 60)
        print("Test 6: Multi-Part SMS - SKIPPED")
        print("=" * 60)
        results["multipart_sms"] = None

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⏭️  SKIPPED"
        print(f"{status}: {test_name.replace('_', ' ').title()}")

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")


if __name__ == "__main__":
    main()
