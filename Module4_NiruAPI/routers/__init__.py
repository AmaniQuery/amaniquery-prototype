"""
AmaniQuery API Routers
Modular endpoints for the AmaniQuery API
"""

from Module4_NiruAPI.routers.query_router import router as query_router
from Module4_NiruAPI.routers.chat_router import router as chat_router
from Module4_NiruAPI.routers.admin_router import router as admin_router
from Module4_NiruAPI.routers.research_router import router as research_router
from Module4_NiruAPI.routers.sms_router import router as sms_router
from Module4_NiruAPI.routers.alignment_router import router as alignment_router
from Module4_NiruAPI.routers.monitoring_router import router as monitoring_router
from Module4_NiruAPI.routers.hybrid_rag_router import router as hybrid_rag_router
from Module4_NiruAPI.routers.news_router import router as news_router
from Module4_NiruAPI.routers.websocket_router import router as websocket_router
from Module4_NiruAPI.routers.notification_router import router as notification_router

__all__ = [
    "query_router",
    "chat_router",
    "admin_router",
    "research_router",
    "sms_router",
    "alignment_router",
    "monitoring_router",
    "hybrid_rag_router",
    "news_router",
    "websocket_router",
    "notification_router",
]
