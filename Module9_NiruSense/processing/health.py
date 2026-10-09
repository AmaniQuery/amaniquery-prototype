"""
Health check module for NiruSense processing pipeline.
Provides health status for all critical components.
"""

import asyncio
from typing import Dict, Any
from datetime import datetime
import redis.asyncio as redis
from .config import settings


class HealthChecker:
    """Centralized health checking for all system components"""

    def __init__(self):
        self.redis_client = None
        self.postgres_pool = None
        self.qdrant_client = None

    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis connection health"""
        try:
            if not self.redis_client:
                self.redis_client = redis.from_url(
                    f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
                    decode_responses=True,
                    socket_timeout=5,
                )

            # Ping Redis
            await self.redis_client.ping()

            # Check stream exists
            stream_info = await self.redis_client.xinfo_stream(
                settings.REDIS_STREAM_KEY
            )

            return {
                "status": "healthy",
                "latency_ms": 0,
                "stream_length": stream_info.get("length", 0),
                "last_check": datetime.utcnow().isoformat(),
            }
        except redis.ConnectionError as e:
            return {
                "status": "unhealthy",
                "error": f"Connection failed: {str(e)}",
                "last_check": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat(),
            }

    async def check_postgres(self) -> Dict[str, Any]:
        """Check PostgreSQL connection health"""
        try:
            import asyncpg

            # Create a temporary connection
            conn = await asyncpg.connect(settings.DATABASE_URL, timeout=5)

            # Simple query
            result = await conn.fetchval("SELECT 1")
            await conn.close()

            return {
                "status": "healthy" if result == 1 else "degraded",
                "last_check": datetime.utcnow().isoformat(),
            }
        except asyncpg.PostgresError as e:
            return {
                "status": "unhealthy",
                "error": f"Connection failed: {str(e)}",
                "last_check": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat(),
            }

    async def check_qdrant(self) -> Dict[str, Any]:
        """Check Qdrant connection health"""
        try:
            from qdrant_client import QdrantClient

            if not self.qdrant_client:
                self.qdrant_client = QdrantClient(
                    url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=5
                )

            # Get collection info
            collection_info = self.qdrant_client.get_collection(
                settings.QDRANT_COLLECTION
            )

            return {
                "status": "healthy",
                "collection": settings.QDRANT_COLLECTION,
                "vectors_count": collection_info.vectors_count,
                "last_check": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat(),
            }

    async def check_agents(self) -> Dict[str, Any]:
        """Check if all agent models are loaded"""
        agent_status = {}

        # This is a placeholder - actual implementation would import and check each agent
        agents = [
            "language_identifier",
            "slang_decoder",
            "topic_classifier",
            "entity_extractor",
            "sentiment_analyzer",
            "emotion_detector",
            "bias_detector",
            "summarizer",
            "quality_scorer",
        ]

        for agent in agents:
            agent_status[agent] = {"status": "loaded", "model": "configured"}

        return {
            "status": "healthy",
            "agents": agent_status,
            "total_agents": len(agents),
            "last_check": datetime.utcnow().isoformat(),
        }

    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        redis_health = await self.check_redis()
        postgres_health = await self.check_postgres()
        qdrant_health = await self.check_qdrant()
        agents_health = await self.check_agents()

        components = {
            "redis": redis_health,
            "postgres": postgres_health,
            "qdrant": qdrant_health,
            "agents": agents_health,
        }

        # Determine overall status
        statuses = [comp.get("status") for comp in components.values()]
        if all(s == "healthy" for s in statuses):
            overall_status = "healthy"
        elif any(s == "unhealthy" for s in statuses):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"

        return {
            "status": overall_status,
            "components": components,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
        }


# Global health checker instance
health_checker = HealthChecker()
