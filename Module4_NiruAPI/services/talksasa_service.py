"""
Talksasa SMS/WhatsApp Notification Service
Handles sending SMS and WhatsApp messages through Talksasa bulk SMS API
"""

import os
import requests
from typing import Optional, Dict
from loguru import logger


class TalksasaNotificationService:
    """Service for Talksasa SMS/WhatsApp integration"""

    def __init__(
        self, api_token: Optional[str] = None, sender_id: Optional[str] = None
    ):
        """
        Initialize Talksasa notification service

        Args:
            api_token: Talksasa API token (Bearer token)
            sender_id: Sender ID for messages
        """
        self.api_token = api_token or os.getenv("TALKSASA_API_TOKEN")
        self.sender_id = sender_id or os.getenv("TALKSASA_SENDER_ID", "TALKSASA")
        self.base_url = "https://bulksms.talksasa.com/api/v3"
        self.available = bool(self.api_token)

        if not self.available:
            logger.warning(
                "Talksasa API token not configured. Notification features disabled."
            )

    def format_phone_number(self, phone_number: str) -> str:
        """
        Format phone number to include country code if missing
        Assumes Kenyan numbers if no country code (+254)
        """
        phone = phone_number.strip().replace(" ", "").replace("-", "")

        # If starts with 0, replace with +254
        if phone.startswith("0"):
            phone = "+254" + phone[1:]
        # If doesn't start with +, add +254
        elif not phone.startswith("+"):
            phone = "+254" + phone

        # Remove + for API (API expects numbers without +)
        if phone.startswith("+"):
            phone = phone[1:]

        return phone

    def send_sms(
        self, recipient: str, message: str, sender_id: Optional[str] = None
    ) -> Dict:
        """
        Send SMS message via Talksasa API

        Args:
            recipient: Phone number to send to
            message: SMS message text
            sender_id: Sender ID (defaults to configured sender_id)

        Returns:
            Dict with status and response data
        """
        if not self.available:
            return {
                "status": "error",
                "message": "SMS service not available - API token not configured",
            }

        try:
            recipient = self.format_phone_number(recipient)
            sender = sender_id or self.sender_id

            url = f"{self.base_url}/sms/send"
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            payload = {
                "recipient": recipient,
                "sender_id": sender,
                "type": "plain",
                "message": message,
            }

            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            result = response.json()

            if result.get("status") == "success":
                logger.info(f"[OK] SMS sent to {recipient}")
                return {"status": "success", "data": result.get("data")}
            else:
                logger.error(f"Failed to send SMS: {result.get('message')}")
                return {
                    "status": "error",
                    "message": result.get("message", "Unknown error"),
                }

        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending SMS: {e}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error sending SMS: {e}")
            return {"status": "error", "message": str(e)}

    def send_whatsapp(
        self, recipient: str, message: str, sender_id: Optional[str] = None
    ) -> Dict:
        """
        Send WhatsApp message via Talksasa API

        Args:
            recipient: Phone number to send to
            message: WhatsApp message text
            sender_id: Sender ID (defaults to configured sender_id)

        Returns:
            Dict with status and response data
        """
        if not self.available:
            return {
                "status": "error",
                "message": "WhatsApp service not available - API token not configured",
            }

        try:
            recipient = self.format_phone_number(recipient)
            sender = sender_id or self.sender_id

            url = f"{self.base_url}/sms/send"
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            payload = {
                "recipient": recipient,
                "sender_id": sender,
                "type": "whatsapp",
                "message": message,
            }

            response = requests.post(url, json=payload, headers=headers, timeout=10)

            # Handle 403 Forbidden specifically (check before raise_for_status)
            if response.status_code == 403:
                try:
                    error_data = response.json()
                    error_message = error_data.get("message", "Forbidden")
                except:
                    error_message = response.text or "Forbidden"

                logger.error(f"WhatsApp 403 Forbidden: {error_message}")
                logger.error(f"Response status: {response.status_code}")
                logger.error(f"Response headers: {dict(response.headers)}")

                return {
                    "status": "error",
                    "message": f"WhatsApp access forbidden (403): {error_message}",
                    "error_code": 403,
                    "details": "This usually means: 1) Your account doesn't have WhatsApp enabled, 2) Your API token lacks WhatsApp permissions, or 3) WhatsApp service is not available for your account. Please contact Talksasa support or use SMS instead.",
                    "response_body": error_message,
                }

            # Check for other error status codes before raising
            if not response.ok:
                try:
                    error_data = response.json()
                    error_message = error_data.get(
                        "message", f"HTTP {response.status_code}"
                    )
                except:
                    error_message = response.text or f"HTTP {response.status_code}"

                logger.error(f"WhatsApp HTTP {response.status_code}: {error_message}")
                return {
                    "status": "error",
                    "message": f"HTTP {response.status_code}: {error_message}",
                    "error_code": response.status_code,
                    "response_body": error_message,
                }

            result = response.json()

            if result.get("status") == "success":
                logger.info(f"[OK] WhatsApp sent to {recipient}")
                return {"status": "success", "data": result.get("data")}
            else:
                logger.error(f"Failed to send WhatsApp: {result.get('message')}")
                return {
                    "status": "error",
                    "message": result.get("message", "Unknown error"),
                }

        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors that weren't caught above (shouldn't happen now, but keep as fallback)
            status_code = e.response.status_code if e.response else None
            try:
                error_data = e.response.json() if e.response else {}
                error_message = error_data.get("message", str(e))
            except:
                error_message = e.response.text if e.response else str(e)

            logger.error(
                f"HTTP error sending WhatsApp (status {status_code}): {error_message}"
            )

            # Special handling for 403
            if status_code == 403:
                return {
                    "status": "error",
                    "message": f"WhatsApp access forbidden (403): {error_message}",
                    "error_code": 403,
                    "details": "This usually means: 1) Your account doesn't have WhatsApp enabled, 2) Your API token lacks WhatsApp permissions, or 3) WhatsApp service is not available for your account. Please contact Talksasa support or use SMS instead.",
                    "response_body": error_message,
                }

            return {
                "status": "error",
                "message": f"HTTP {status_code}: {error_message}",
                "error_code": status_code,
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error sending WhatsApp: {e}")
            return {"status": "error", "message": f"Request failed: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error sending WhatsApp: {e}")
            return {"status": "error", "message": str(e)}

    def send_notification(
        self, recipient: str, message: str, notification_type: str = "whatsapp"
    ) -> Dict:
        """
        Send notification (WhatsApp with SMS fallback)

        Args:
            recipient: Phone number to send to
            message: Message text
            notification_type: 'whatsapp', 'sms', or 'both'

        Returns:
            Dict with status and response data
        """
        if notification_type == "sms":
            return self.send_sms(recipient, message)
        elif notification_type == "whatsapp":
            # Try WhatsApp first, fallback to SMS on error
            result = self.send_whatsapp(recipient, message)
            if result.get("status") == "error":
                error_code = result.get("error_code")
                # If it's a 403 (forbidden), log it but still fallback to SMS
                if error_code == 403:
                    logger.warning(
                        f"WhatsApp access forbidden (403) for {recipient}. This usually means WhatsApp is not enabled for your account. Falling back to SMS."
                    )
                else:
                    logger.info(
                        f"WhatsApp failed (error code: {error_code}), falling back to SMS for {recipient}"
                    )
                return self.send_sms(recipient, message)
            return result
        elif notification_type == "both":
            # Send both
            whatsapp_result = self.send_whatsapp(recipient, message)
            sms_result = self.send_sms(recipient, message)
            # Return success if at least one succeeded
            if (
                whatsapp_result.get("status") == "success"
                or sms_result.get("status") == "success"
            ):
                return {
                    "status": "success",
                    "whatsapp": whatsapp_result,
                    "sms": sms_result,
                }
            return {
                "status": "error",
                "message": "Both WhatsApp and SMS failed",
                "whatsapp": whatsapp_result,
                "sms": sms_result,
            }
        else:
            return {
                "status": "error",
                "message": f"Invalid notification type: {notification_type}",
            }
