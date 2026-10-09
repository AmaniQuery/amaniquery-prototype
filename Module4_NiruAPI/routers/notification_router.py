"""
Notification Subscription API Router
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from pydantic import BaseModel
from loguru import logger
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Module3_NiruDB.notification_models import (
    SubscriptionCreate,
    SubscriptionUpdate,
    SubscriptionResponse,
)
from Module4_NiruAPI.services.notification_service import NotificationService
from Module4_NiruAPI.services.news_service import NewsService

router = APIRouter(prefix="/api/v1/notifications", tags=["notifications"])

# Global notification service instance (will be initialized in api.py)
notification_service: Optional[NotificationService] = None
news_service: Optional[NewsService] = None


def get_notification_service() -> NotificationService:
    """Dependency to get notification service"""
    if notification_service is None:
        raise HTTPException(
            status_code=503, detail="Notification service not initialized"
        )
    return notification_service


def get_news_service() -> NewsService:
    """Dependency to get news service"""
    global news_service
    if news_service is None:
        news_service = NewsService()
    return news_service


@router.post("/subscribe", response_model=SubscriptionResponse)
async def subscribe(request: SubscriptionCreate):
    """Subscribe to news notifications"""
    try:
        service = get_notification_service()
        subscription = service.subscribe(request)
        return subscription
    except Exception as e:
        logger.error(f"Error subscribing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unsubscribe")
async def unsubscribe(phone_number: str):
    """Unsubscribe from news notifications"""
    try:
        service = get_notification_service()
        success = service.unsubscribe(phone_number)
        if not success:
            raise HTTPException(status_code=404, detail="Subscription not found")
        return {"status": "success", "message": "Unsubscribed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unsubscribing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/subscribe", response_model=SubscriptionResponse)
async def update_subscription(phone_number: str, request: SubscriptionUpdate):
    """Update subscription preferences"""
    try:
        service = get_notification_service()
        subscription = service.update_subscription(phone_number, request)
        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")
        return subscription
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating subscription: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/subscription/{phone_number}", response_model=SubscriptionResponse)
async def get_subscription(phone_number: str):
    """Get subscription status by phone number"""
    try:
        service = get_notification_service()
        subscription = service.get_subscription(phone_number)
        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")
        return subscription
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting subscription: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
async def get_categories():
    """Get available news categories"""
    try:
        service = get_news_service()
        categories = service.get_categories()
        return {"categories": categories}
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources")
async def get_sources():
    """Get available news sources"""
    try:
        service = get_news_service()
        sources = service.get_sources()
        return {"sources": sources}
    except Exception as e:
        logger.error(f"Error getting sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))
