"""
Monitoring and Health Checks for News Crawler
"""

import sys
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta
from loguru import logger
import os

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


class CrawlerMonitor:
    """Monitor crawler performance and source status"""

    def __init__(self):
        """Initialize monitor"""
        self.db_storage = None
        self.dedup_engine = None
        self.rate_limiter = None

        try:
            from Module3_NiruDB.database_storage import DatabaseStorage

            self.db_storage = DatabaseStorage()
        except Exception as e:
            logger.warning(f"Database storage not available for monitoring: {e}")

        try:
            from .deduplication import DeduplicationEngine

            self.dedup_engine = DeduplicationEngine()
        except Exception as e:
            logger.warning(f"Deduplication engine not available for monitoring: {e}")

        try:
            from .rate_limiter import RateLimiter

            self.rate_limiter = RateLimiter()
        except Exception as e:
            logger.warning(f"Rate limiter not available for monitoring: {e}")

    def get_health_status(self) -> Dict:
        """
        Get overall health status

        Returns:
            Dictionary with health status
        """
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
        }

        # Check database
        db_health = self._check_database()
        health["components"]["database"] = db_health
        if db_health["status"] != "healthy":
            health["status"] = "degraded"

        # Check deduplication
        dedup_health = self._check_deduplication()
        health["components"]["deduplication"] = dedup_health
        if dedup_health["status"] != "healthy":
            health["status"] = "degraded"

        # Check rate limiter
        rate_limiter_health = self._check_rate_limiter()
        health["components"]["rate_limiter"] = rate_limiter_health

        return health

    def _check_database(self) -> Dict:
        """Check database health"""
        if not self.db_storage:
            return {
                "status": "unavailable",
                "message": "Database storage not initialized",
            }

        try:
            # Try to get a session
            db = self.db_storage.get_db_session()
            db.close()

            # Get stats
            stats = self.db_storage.get_stats()

            return {"status": "healthy", "stats": stats}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _check_deduplication(self) -> Dict:
        """Check deduplication engine health"""
        if not self.dedup_engine:
            return {
                "status": "unavailable",
                "message": "Deduplication engine not initialized",
            }

        try:
            stats = self.dedup_engine.get_duplicate_stats(days=7)
            return {"status": "healthy", "stats": stats}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _check_rate_limiter(self) -> Dict:
        """Check rate limiter health"""
        if not self.rate_limiter:
            return {"status": "unavailable", "message": "Rate limiter not initialized"}

        try:
            stats = self.rate_limiter.get_stats()
            return {
                "status": "healthy" if stats.get("status") == "active" else "degraded",
                "stats": stats,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def get_source_status(self, days: int = 7) -> Dict:
        """
        Get status of news sources

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with source status
        """
        if not self.db_storage:
            return {"sources": []}

        try:
            from sqlalchemy import func, text
            from Module3_NiruDB.database_storage import RawDocument

            cutoff_date = datetime.utcnow() - timedelta(days=days)

            db = self.db_storage.get_db_session()

            # Try PostgreSQL JSON path extraction, fallback to None
            try:
                source_stats = (
                    db.query(
                        RawDocument.source_name,
                        func.count(RawDocument.id).label("total_articles"),
                        func.max(RawDocument.crawl_date).label("last_crawl"),
                        func.avg(
                            func.cast(
                                func.json_extract_path_text(
                                    RawDocument.metadata_json, "quality_score"
                                ),
                                db.Float,
                            )
                        ).label("avg_quality_score"),
                    )
                    .filter(RawDocument.crawl_date >= cutoff_date)
                    .group_by(RawDocument.source_name)
                    .all()
                )
            except Exception:
                # Fallback for SQLite (no json_extract_path_text)
                source_stats = (
                    db.query(
                        RawDocument.source_name,
                        func.count(RawDocument.id).label("total_articles"),
                        func.max(RawDocument.crawl_date).label("last_crawl"),
                        text("NULL").label("avg_quality_score"),
                    )
                    .filter(RawDocument.crawl_date >= cutoff_date)
                    .group_by(RawDocument.source_name)
                    .all()
                )

            db.close()

            sources = []
            for stat in source_stats:
                sources.append(
                    {
                        "name": stat.source_name,
                        "total_articles": stat.total_articles,
                        "last_crawl": (
                            stat.last_crawl.isoformat() if stat.last_crawl else None
                        ),
                        "avg_quality_score": (
                            float(stat.avg_quality_score)
                            if stat.avg_quality_score
                            else None
                        ),
                        "status": (
                            "active"
                            if stat.last_crawl
                            and (datetime.utcnow() - stat.last_crawl)
                            < timedelta(days=1)
                            else "inactive"
                        ),
                    }
                )

            return {"sources": sources}

        except Exception as e:
            logger.error(f"Error getting source status: {e}")
            return {"sources": [], "error": str(e)}

    def get_crawler_stats(self) -> Dict:
        """Get crawler statistics"""
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "database": {},
            "deduplication": {},
            "rate_limiting": {},
        }

        if self.db_storage:
            try:
                stats["database"] = self.db_storage.get_stats()
            except Exception as e:
                stats["database"] = {"error": str(e)}

        if self.dedup_engine:
            try:
                stats["deduplication"] = self.dedup_engine.get_duplicate_stats()
            except Exception as e:
                stats["deduplication"] = {"error": str(e)}

        if self.rate_limiter:
            try:
                stats["rate_limiting"] = self.rate_limiter.get_stats()
            except Exception as e:
                stats["rate_limiting"] = {"error": str(e)}

        return stats
