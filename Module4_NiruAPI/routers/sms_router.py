"""
SMS Router - SMS webhook and gateway endpoints for AmaniQuery
"""

import os
import time
from fastapi import APIRouter, HTTPException, Request, Form, Depends
from loguru import logger

router = APIRouter(tags=["SMS Gateway"])


# =============================================================================
# DEPENDENCIES - Module-level globals set by main app
# =============================================================================

sms_pipeline = None
sms_service = None

# Simple in-memory rate limiter for SMS sending
_sms_rate_limit_store: dict = {}
_SMS_RATE_LIMIT = int(os.getenv("SMS_RATE_LIMIT_PER_MINUTE", "5"))


async def sms_rate_limiter(request: Request):
    """Rate limit SMS sending per IP (5 per minute default)"""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    minute_ago = now - 60
    _sms_rate_limit_store[client_ip] = [
        t for t in _sms_rate_limit_store.get(client_ip, []) if t > minute_ago
    ]
    if len(_sms_rate_limit_store[client_ip]) >= _SMS_RATE_LIMIT:
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded. Try again later."
        )
    _sms_rate_limit_store[client_ip].append(now)
    return True


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/sms-webhook")
async def sms_webhook(
    request: Request,
    from_: str = Form(..., alias="from"),
    to: str = Form(...),
    text: str = Form(...),
    date: str = Form(None),
    id_: str = Form(None, alias="id"),
    linkId: str = Form(None),
    networkCode: str = Form(None),
):
    """
    Africa's Talking SMS Webhook

    Receives incoming SMS messages and sends intelligent responses.
    This endpoint is called by Africa's Talking when an SMS is received.

    **How it works:**
    1. User sends SMS to your Africa's Talking shortcode/number
    2. Africa's Talking forwards the SMS to this webhook
    3. AmaniQuery processes the query using RAG pipeline
    4. Response is sent back via SMS (max 160 characters)

    **Example SMS queries:**
    - "What is the Finance Bill about?"
    - "Latest news on housing"
    - "Constitution Article 10"

    **Setup:**
    1. Sign up at https://africastalking.com
    2. Get API key and username
    3. Set environment variables: AT_USERNAME, AT_API_KEY
    4. Configure webhook URL in Africa's Talking dashboard
    """
    if sms_pipeline is None or sms_service is None:
        logger.error("SMS services not initialized")
        return {"status": "error", "message": "SMS service unavailable"}

    try:
        # Parse incoming SMS
        phone_number = sms_service.format_kenyan_phone(from_)
        query_text = text.strip()

        logger.info(f"📱 Incoming SMS from {phone_number}: {query_text}")

        # Detect language (basic detection)
        language = (
            "sw"
            if any(
                word in query_text.lower()
                for word in ["nini", "habari", "tafadhali", "je"]
            )
            else "en"
        )

        # Process query through SMS-optimized RAG
        result = sms_pipeline.process_sms_query(
            query=query_text, language=language, phone_number=phone_number
        )

        response_text = result["response"]

        # Send SMS response
        if sms_service.available:
            send_result = sms_service.send_sms(phone_number, response_text)

            if send_result.get("success"):
                logger.info(f"✓ SMS sent to {phone_number}")
                return {
                    "status": "success",
                    "message": "Response sent",
                    "response_text": response_text,
                    "query_type": result.get("query_type"),
                    "message_id": send_result.get("message_id"),
                }
            else:
                logger.error(f"Failed to send SMS: {send_result.get('error')}")
                return {
                    "status": "error",
                    "message": "Failed to send response",
                    "error": send_result.get("error"),
                }
        else:
            logger.warning(f"SMS service unavailable. Would send: {response_text}")
            return {
                "status": "success",
                "message": "Query processed (SMS sending disabled)",
                "response_text": response_text,
                "query_type": result.get("query_type"),
            }

    except Exception as e:
        logger.error(f"Error handling SMS webhook: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/sms-send")
async def send_sms_manual(
    phone_number: str, message: str, request: Request, _=Depends(sms_rate_limiter)
):
    """
    Send SMS manually (for testing)

    **Parameters:**
    - phone_number: Recipient phone number (+254XXXXXXXXX)
    - message: SMS message text (max 160 characters recommended)

    **Example:**
    ```
    POST /sms-send
    {
        "phone_number": "+254712345678",
        "message": "Finance Bill 2025 aims to raise revenue through new taxes."
    }
    ```
    """
    if sms_service is None:
        raise HTTPException(
            status_code=503,
            detail="SMS service not initialized. Please restart the FastAPI server.",
        )

    if not sms_service.available:
        error_detail = "SMS service not available"
        if hasattr(sms_service, "test_mode") and sms_service.test_mode:
            error_detail += " (test mode is enabled)"
        elif hasattr(sms_service, "use_direct_api") and sms_service.use_direct_api:
            error_detail += " (using direct API fallback)"
        else:
            error_detail += ". Check AT_USERNAME and AT_API_KEY environment variables."

        raise HTTPException(status_code=503, detail=error_detail)

    try:
        formatted_phone = sms_service.format_kenyan_phone(phone_number)
        result = sms_service.send_sms(formatted_phone, message)

        if result.get("success"):
            return {
                "status": "success",
                "phone_number": formatted_phone,
                "message": message,
                "message_id": result.get("message_id"),
                "cost": result.get("cost"),
            }
        else:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"Failed to send SMS: {error_msg}")
            if "SSL" in str(error_msg) or "Connection" in str(error_msg):
                raise HTTPException(
                    status_code=503, detail=f"Network error: {error_msg}"
                )
            raise HTTPException(
                status_code=500, detail=f"Failed to send SMS: {error_msg}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending manual SMS: {e}")
        error_str = str(e)
        if "SSL" in error_str or "Connection" in error_str:
            raise HTTPException(status_code=503, detail=f"Network error: {error_str}")
        raise HTTPException(status_code=500, detail=f"Internal error: {error_str}")


@router.get("/sms-query")
async def sms_query_preview(query: str, language: str = "en"):
    """
    Preview SMS response without sending

    Test what response would be sent via SMS for a given query.
    Useful for testing before deploying webhook.

    **Parameters:**
    - query: Question to ask
    - language: Response language ('en' or 'sw')
    """
    if sms_pipeline is None:
        raise HTTPException(status_code=503, detail="SMS pipeline not initialized")

    try:
        result = sms_pipeline.process_sms_query(query=query, language=language)

        return {
            "query": query,
            "response": result["response"],
            "character_count": len(result["response"]),
            "within_sms_limit": len(result["response"]) <= 160,
            "query_type": result.get("query_type"),
            "sources": result.get("sources", []),
            "language": language,
        }

    except Exception as e:
        logger.error(f"Error previewing SMS query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
