"""
Africa's Talking SMS Service Integration
Handles sending and receiving SMS messages
"""

from typing import Optional, Dict
from loguru import logger
import os
import time
import ssl
import urllib3

# Disable SSL warnings if we're using unverified context (for development)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class AfricasTalkingSMSService:
    """Service for Africa's Talking SMS integration"""

    def __init__(
        self, username: str = None, api_key: str = None, test_mode: bool = None
    ):
        """
        Initialize Africa's Talking SMS service

        Args:
            username: Africa's Talking username (default from env)
            api_key: Africa's Talking API key (default from env)
            test_mode: Enable test mode (simulates sending without API calls)
                      If None, reads from env variable SMS_TEST_MODE
        """
        self.username = username or os.getenv("AT_USERNAME", "sandbox")
        self.api_key = api_key or os.getenv("AT_API_KEY", "")

        # Check for test mode
        if test_mode is None:
            test_mode = os.getenv("SMS_TEST_MODE", "false").lower() in (
                "true",
                "1",
                "yes",
            )
        self.test_mode = test_mode

        # Check for SSL verification bypass (development only)
        self.verify_ssl = os.getenv("SMS_VERIFY_SSL", "true").lower() not in (
            "false",
            "0",
            "no",
        )
        if not self.verify_ssl:
            logger.warning(
                "[WARN]  SSL verification is DISABLED - This is insecure and should only be used in development!"
            )
            # When SSL verification is disabled, use direct API by default for better control
            self.use_direct_api = True
        else:
            self.use_direct_api = False

        if self.test_mode:
            logger.info("SMS service running in TEST MODE - SMS will be simulated")
            self.available = True
            return

        # Configure SSL before initializing SDK
        self._configure_ssl()

        # Initialize Africa's Talking SDK (skip if using direct API)
        if not self.use_direct_api:
            try:
                import africastalking

                self.africastalking = africastalking
                self.africastalking.initialize(self.username, self.api_key)
                self.sms = self.africastalking.SMS
                self.available = True
                logger.info(
                    f"Africa's Talking SDK initialized for user: {self.username}"
                )
            except ImportError:
                logger.warning(
                    "africastalking package not installed. Using direct API calls instead."
                )
                self.use_direct_api = True
                self.available = True  # Try direct API
            except Exception as e:
                logger.warning(f"Failed to initialize Africa's Talking SDK: {e}")
                logger.info("Will use direct API calls as fallback")
                self.use_direct_api = True
                self.available = True  # Try direct API
        else:
            # Using direct API by default (SSL verification disabled)
            logger.info("Using direct API calls (SSL verification disabled)")
            self.available = True

    def _configure_ssl(self):
        """Configure SSL settings to handle various network environments"""
        try:
            # Try to update certifi if available
            try:
                import certifi

                os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
                os.environ["SSL_CERT_FILE"] = certifi.where()
                logger.debug(f"Using certifi certificates: {certifi.where()}")
            except ImportError:
                logger.debug("certifi not available, using system certificates")

            # Configure SSL context for better compatibility
            if not self.verify_ssl:
                # Create unverified context (development only)
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                logger.warning("SSL verification disabled - using unverified context")
        except Exception as e:
            logger.warning(f"Could not configure SSL settings: {e}")

    def _send_sms_direct_api(self, phone_number: str, message: str) -> Dict:
        """
        Send SMS using direct API calls (fallback when SDK fails)

        Args:
            phone_number: Recipient phone number
            message: SMS message text

        Returns:
            Dictionary with status and details
        """
        try:
            import requests
            import base64

            # Determine API endpoint (sandbox or production)
            if "sandbox" in self.username.lower():
                base_url = "https://api.sandbox.africastalking.com"
            else:
                base_url = "https://api.africastalking.com"

            url = f"{base_url}/version1/messaging"

            # Prepare authentication
            auth_string = f"{self.username}:{self.api_key}"
            auth_bytes = auth_string.encode("ascii")
            auth_b64 = base64.b64encode(auth_bytes).decode("ascii")

            headers = {
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "apiKey": self.api_key,
                "Authorization": f"Basic {auth_b64}",
            }

            # Prepare payload (Africa's Talking API format)
            payload = {
                "username": self.username,
                "to": phone_number,
                "message": message,
            }
            # Note: "from" (sender ID) is optional and can be added if needed

            # Make request with SSL configuration
            session = requests.Session()
            if not self.verify_ssl:
                session.verify = False

            response = session.post(url, data=payload, headers=headers, timeout=30)
            response.raise_for_status()

            result = response.json()

            # Parse response
            if result.get("SMSMessageData", {}).get("Recipients"):
                recipient = result["SMSMessageData"]["Recipients"][0]

                if recipient.get("status") == "Success":
                    return {
                        "success": True,
                        "message_id": recipient.get("messageId"),
                        "cost": recipient.get("cost"),
                        "phone_number": phone_number,
                        "message": message,
                    }
                else:
                    return {
                        "success": False,
                        "error": recipient.get("status") or recipient.get("statusCode"),
                        "phone_number": phone_number,
                    }
            else:
                return {
                    "success": False,
                    "error": "No recipients in response",
                    "response": result,
                }

        except ImportError:
            return {
                "success": False,
                "error": "requests library not available for direct API calls",
            }
        except Exception as e:
            logger.error(f"Error in direct API call: {e}")
            return {"success": False, "error": str(e), "phone_number": phone_number}

    def send_sms(self, phone_number: str, message: str, max_retries: int = 3) -> Dict:
        """
        Send SMS to a phone number with retry logic and SSL fixes

        Args:
            phone_number: Recipient phone number (format: +254XXXXXXXXX)
            message: SMS message text (max 160 chars for single SMS)
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            Dictionary with status and details
        """
        if not self.available:
            return {
                "success": False,
                "error": "SMS service not available",
                "message": "SMS service not initialized",
            }

        # Validate phone number format
        if not phone_number.startswith("+"):
            phone_number = f"+{phone_number}"

        # Test mode - simulate sending
        if self.test_mode:
            logger.info(
                f"[TEST MODE] Simulated SMS to {phone_number}: {message[:50]}..."
            )
            return {
                "success": True,
                "message_id": f"test-{int(time.time())}",
                "cost": "0.00",
                "phone_number": phone_number,
                "message": message,
                "test_mode": True,
            }

        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(max_retries):
            try:
                # Try SDK first, fallback to direct API if needed
                if self.use_direct_api or attempt > 0:
                    logger.info(
                        f"Using direct API call (attempt {attempt + 1}/{max_retries})"
                    )
                    result = self._send_sms_direct_api(phone_number, message)
                    if result.get("success"):
                        return result
                    # If direct API also fails, try SDK on next retry
                    if attempt < max_retries - 1:
                        self.use_direct_api = False
                else:
                    # Try SDK (only if SDK is available)
                    if not hasattr(self, "sms") or self.sms is None:
                        # SDK not available, use direct API
                        logger.info("SDK not available, using direct API")
                        result = self._send_sms_direct_api(phone_number, message)
                        if result.get("success"):
                            return result
                        # Continue to retry logic if it fails
                        raise Exception(result.get("error", "Direct API call failed"))

                    response = self.sms.send(message, [phone_number])
                    logger.info(f"SMS sent to {phone_number}: {response}")

                    # Parse response
                    if response.get("SMSMessageData", {}).get("Recipients"):
                        recipient = response["SMSMessageData"]["Recipients"][0]

                        if recipient.get("status") == "Success":
                            return {
                                "success": True,
                                "message_id": recipient.get("messageId"),
                                "cost": recipient.get("cost"),
                                "phone_number": phone_number,
                                "message": message,
                            }
                        else:
                            return {
                                "success": False,
                                "error": recipient.get("status"),
                                "phone_number": phone_number,
                            }
                    else:
                        return {
                            "success": False,
                            "error": "No recipients in response",
                            "response": response,
                        }

            except Exception as e:
                last_error = e
                error_str = str(e)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {error_str}"
                )

                # Detect SSL/network errors
                is_ssl_error = any(
                    keyword in error_str
                    for keyword in [
                        "SSL",
                        "Connection",
                        "WRONG_VERSION_NUMBER",
                        "SSLError",
                        "HTTPSConnectionPool",
                    ]
                )

                if is_ssl_error:
                    logger.info(
                        "SSL error detected, will try direct API with SSL fixes on next attempt"
                    )
                    self.use_direct_api = True

                    # On first SSL error, try disabling SSL verification if allowed
                    if attempt == 0 and self.verify_ssl:
                        logger.warning(
                            "SSL error on first attempt. Consider setting SMS_VERIFY_SSL=false for development"
                        )

                # Don't retry on last attempt
                if attempt < max_retries - 1:
                    wait_time = (2**attempt) * 1  # Exponential backoff: 1s, 2s, 4s
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} attempts failed")

        # All retries exhausted
        error_msg = str(last_error) if last_error else "Unknown error"

        # Provide helpful error message
        if "SSL" in error_msg or "Connection" in error_msg:
            error_msg += (
                "\n\nTroubleshooting:\n"
                "1. Set SMS_VERIFY_SSL=false to disable SSL verification (development only)\n"
                "2. Update certifi: pip install --upgrade certifi\n"
                "3. Check proxy/firewall settings\n"
                "4. Try from a different network"
            )

        return {
            "success": False,
            "error": error_msg,
            "phone_number": phone_number,
            "attempts": max_retries,
        }

    def send_multi_part_sms(self, phone_number: str, messages: list) -> Dict:
        """
        Send multiple SMS messages in sequence

        Args:
            phone_number: Recipient phone number
            messages: List of message strings

        Returns:
            Dictionary with status and details
        """
        results = []

        for i, message in enumerate(messages):
            result = self.send_sms(phone_number, message)
            results.append({"part": i + 1, "total": len(messages), **result})

            if not result.get("success"):
                logger.error(f"Failed to send part {i+1}/{len(messages)}")
                break

        all_success = all(r.get("success") for r in results)

        return {
            "success": all_success,
            "parts_sent": len(results),
            "total_parts": len(messages),
            "results": results,
            "phone_number": phone_number,
        }

    def parse_incoming_sms(self, webhook_data: Dict) -> Dict:
        """
        Parse incoming SMS webhook data from Africa's Talking

        Args:
            webhook_data: Webhook POST data from Africa's Talking

        Returns:
            Parsed SMS data
        """
        try:
            return {
                "from": webhook_data.get("from"),
                "to": webhook_data.get("to"),
                "text": webhook_data.get("text", "").strip(),
                "date": webhook_data.get("date"),
                "id": webhook_data.get("id"),
                "linkId": webhook_data.get("linkId"),
                "networkCode": webhook_data.get("networkCode"),
            }
        except Exception as e:
            logger.error(f"Error parsing incoming SMS: {e}")
            return {"error": str(e), "raw_data": webhook_data}

    def validate_webhook_signature(self, request_data: str, signature: str) -> bool:
        """
        Validate Africa's Talking webhook HMAC-SHA256 signature.

        Args:
            request_data: Raw request body as string
            signature: X-Africastalking-Signature header value

        Returns:
            True if signature is valid
        """
        import hashlib
        import hmac

        webhook_token = os.getenv("AFRICASTALKING_WEBHOOK_TOKEN") or os.getenv(
            "AFRICASTALKING_API_KEY"
        )
        if not webhook_token:
            logger.warning(
                "AFRICASTALKING_WEBHOOK_TOKEN not set — skipping signature validation"
            )
            return True
        if not signature:
            logger.warning("Empty webhook signature — rejecting")
            return False
        expected = hmac.new(
            webhook_token.encode(),
            request_data.encode(),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature.lower())

    def format_kenyan_phone(self, phone_number: str) -> str:
        """
        Format phone number to Kenya format (+254...)

        Args:
            phone_number: Phone number in various formats

        Returns:
            Formatted phone number with +254 prefix
        """
        # Remove spaces and special characters
        phone = (
            phone_number.replace(" ", "")
            .replace("-", "")
            .replace("(", "")
            .replace(")", "")
        )

        # Handle different formats
        if phone.startswith("+254"):
            return phone
        elif phone.startswith("254"):
            return f"+{phone}"
        elif phone.startswith("0"):
            return f"+254{phone[1:]}"
        elif phone.startswith("7") or phone.startswith("1"):
            return f"+254{phone}"
        else:
            return phone  # Return as-is if unrecognized

    def get_account_balance(self) -> Dict:
        """Get Africa's Talking account balance"""
        if not self.available:
            return {"error": "Service not available"}

        try:
            application = self.africastalking.Application
            response = application.fetch_application_data()

            return {
                "balance": response.get("UserData", {}).get("balance"),
                "currency": "KES",  # Kenya Shillings
            }
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {"error": str(e)}
