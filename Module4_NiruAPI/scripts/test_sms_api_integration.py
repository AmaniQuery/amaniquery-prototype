#!/usr/bin/env python3
"""
Test script for SMS Service Integration through FastAPI
Tests the complete SMS pipeline: query processing + SMS sending
"""

import os
import sys
import requests
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
import argparse
from datetime import datetime


class SMSAPITester:
    """Test SMS service through FastAPI endpoints"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API tester

        Args:
            base_url: FastAPI server base URL
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )

    def test_sms_query_preview(self, query: str, language: str = "en") -> Dict:
        """
        Test SMS query preview endpoint

        Args:
            query: Query to test
            language: Response language ('en' or 'sw')

        Returns:
            API response dictionary
        """
        print("\n" + "=" * 60)
        print("Test: SMS Query Preview")
        print("=" * 60)
        print(f"Query: {query}")
        print(f"Language: {language}")

        try:
            url = f"{self.base_url}/sms-query"
            params = {"query": query, "language": language}

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            result = response.json()

            print(f"✅ Query processed successfully")
            print(f"  Response: {result.get('response', 'N/A')}")
            print(f"  Character count: {result.get('character_count', 0)}")
            print(f"  Within SMS limit (160): {result.get('within_sms_limit', False)}")
            print(f"  Query type: {result.get('query_type', 'N/A')}")

            sources = result.get("sources", [])
            if sources:
                print(f"  Sources: {len(sources)}")
                for i, source in enumerate(sources[:3], 1):
                    print(f"    {i}. {source.get('title', 'N/A')}")

            return {"success": True, "data": result}

        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to API at {self.base_url}")
            print("   Make sure the FastAPI server is running")
            return {"success": False, "error": "Connection error"}
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error: {e}")
            if e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"   Detail: {error_detail}")
                except:
                    print(f"   Response: {e.response.text}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.exception(e)
            return {"success": False, "error": str(e)}

    def test_send_sms(self, phone_number: str, message: str) -> Dict:
        """
        Test manual SMS sending endpoint

        Args:
            phone_number: Recipient phone number
            message: SMS message text

        Returns:
            API response dictionary
        """
        print("\n" + "=" * 60)
        print("Test: Send SMS via API")
        print("=" * 60)
        print(f"Phone: {phone_number}")
        print(f"Message: {message}")
        print(f"Message length: {len(message)} characters")

        try:
            url = f"{self.base_url}/sms-send"
            params = {"phone_number": phone_number, "message": message}

            response = self.session.post(url, params=params, timeout=30)
            response.raise_for_status()
            result = response.json()

            if result.get("status") == "success":
                print("✅ SMS sent successfully via API")
                print(f"  Message ID: {result.get('message_id', 'N/A')}")
                print(f"  Cost: {result.get('cost', 'N/A')}")
                print(f"  Formatted phone: {result.get('phone_number', 'N/A')}")
                return {"success": True, "data": result}
            else:
                print(f"❌ Failed to send SMS: {result.get('detail', 'Unknown error')}")
                return {
                    "success": False,
                    "error": result.get("detail", "Unknown error"),
                }

        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to API at {self.base_url}")
            print("   Make sure the FastAPI server is running")
            return {"success": False, "error": "Connection error"}
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error: {e}")
            if e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"   Detail: {error_detail.get('detail', error_detail)}")
                except:
                    print(f"   Response: {e.response.text}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.exception(e)
            return {"success": False, "error": str(e)}

    def test_full_sms_flow(
        self, phone_number: str, query: str, language: str = "en"
    ) -> Dict:
        """
        Test complete SMS flow: query processing + SMS sending

        Args:
            phone_number: Recipient phone number
            query: User query
            language: Response language

        Returns:
            Combined test results
        """
        print("\n" + "=" * 60)
        print("Test: Full SMS Flow (Query + Send)")
        print("=" * 60)
        print(f"Phone: {phone_number}")
        print(f"Query: {query}")
        print(f"Language: {language}")

        results = {}

        # Step 1: Preview query response
        print("\n📋 Step 1: Preview Query Response")
        preview_result = self.test_sms_query_preview(query, language)
        results["preview"] = preview_result

        if not preview_result.get("success"):
            print("\n❌ Cannot proceed - query preview failed")
            return results

        # Get the response text
        response_text = preview_result["data"].get("response", "")

        if not response_text:
            print("\n⚠️  No response generated, skipping SMS send")
            return results

        # Step 2: Send SMS with the generated response
        print("\n📤 Step 2: Send SMS with Generated Response")
        send_result = self.test_send_sms(phone_number, response_text)
        results["send"] = send_result

        # Summary
        print("\n" + "=" * 60)
        print("Full Flow Summary")
        print("=" * 60)

        if preview_result.get("success") and send_result.get("success"):
            print("✅ Complete flow successful!")
            print(f"  Query processed: ✅")
            print(f"  SMS sent: ✅")
            print(f"  Message ID: {send_result['data'].get('message_id', 'N/A')}")
        elif preview_result.get("success"):
            print("⚠️  Query processed but SMS sending failed")
            print(f"  Query processed: ✅")
            print(f"  SMS sent: ❌")
        else:
            print("❌ Flow failed at query processing stage")

        return results


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test SMS Service through FastAPI")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="FastAPI server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--phone",
        type=str,
        help="Phone number for SMS tests (format: +254XXXXXXXXX or 0712345678)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What is the Finance Bill?",
        help="Query to test (default: 'What is the Finance Bill?')",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "sw"],
        help="Response language (default: en)",
    )
    parser.add_argument(
        "--test-mode",
        type=str,
        choices=["preview", "send", "full"],
        default="full",
        help="Test mode: preview (query only), send (SMS only), full (both)",
    )
    parser.add_argument(
        "--message",
        type=str,
        help="Custom message for send test (overrides query response)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("SMS Service API Integration Test")
    print("=" * 60)
    print(f"API URL: {args.url}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tester = SMSAPITester(base_url=args.url)

    results = {}

    # Test 1: Query Preview
    if args.test_mode in ["preview", "full"]:
        results["preview"] = tester.test_sms_query_preview(args.query, args.language)

    # Test 2: Send SMS
    if args.test_mode in ["send", "full"]:
        if not args.phone:
            print("\n" + "=" * 60)
            print("Test: Send SMS - SKIPPED")
            print("=" * 60)
            print("⚠️  No phone number provided. Use --phone to test SMS sending")
            print(
                "   Example: python test_sms_api_integration.py --phone 0796572619 --test-mode send"
            )
            results["send"] = None
        else:
            if args.message:
                # Use custom message
                results["send"] = tester.test_send_sms(args.phone, args.message)
            elif args.test_mode == "send":
                # Send mode without query - use default test message
                test_message = f"Test SMS from AmaniQuery API - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                results["send"] = tester.test_send_sms(args.phone, test_message)
            else:
                # Full mode - will be handled in test_full_sms_flow
                pass

    # Test 3: Full Flow (if in full mode and phone provided)
    if args.test_mode == "full" and args.phone:
        results["full_flow"] = tester.test_full_sms_flow(
            phone_number=args.phone, query=args.query, language=args.language
        )

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, result in results.items():
        if isinstance(result, dict):
            if result.get("success"):
                status = "✅ PASSED"
            elif result is None:
                status = "⏭️  SKIPPED"
            else:
                status = "❌ FAILED"
        else:
            status = "⏭️  SKIPPED"
        print(f"{status}: {test_name.replace('_', ' ').title()}")

    passed = sum(
        1 for r in results.values() if isinstance(r, dict) and r.get("success")
    )
    failed = sum(
        1
        for r in results.values()
        if isinstance(r, dict) and not r.get("success") and r is not None
    )
    skipped = sum(
        1
        for r in results.values()
        if r is None or (isinstance(r, dict) and r.get("success") is None)
    )

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0 and passed > 0:
        print("\n🎉 All tests passed!")
    elif failed > 0:
        print(f"\n⚠️  {failed} test(s) failed")
    else:
        print("\n⚠️  No tests were run")


if __name__ == "__main__":
    main()
