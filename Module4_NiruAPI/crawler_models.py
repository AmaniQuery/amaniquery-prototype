"""
Database Models for Crawler Status and Logs
"""

from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Integer, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.exc import OperationalError
from loguru import logger
import os

Base = declarative_base()


class CrawlerStatus(Base):
    """Database model for crawler status"""

    __tablename__ = "crawler_status"

    crawler_name = Column(String(100), primary_key=True)
    status = Column(String(50), nullable=False, default="idle")  # idle, running, failed
    last_run = Column(DateTime, nullable=True)
    pid = Column(Integer, nullable=True)
    start_time = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CrawlerLog(Base):
    """Database model for crawler logs"""

    __tablename__ = "crawler_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    crawler_name = Column(String(100), nullable=False, index=True)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Index for efficient queries
    __table_args__ = (
        (
            {"postgresql_partition_by": "RANGE (timestamp)"}
            if os.getenv("ENABLE_PARTITIONING", "false").lower() == "true"
            else {}
        ),
    )


class CrawlerDatabaseManager:
    """Manages crawler status and logs in PostgreSQL database"""

    def __init__(self, database_url: str = None):
        """
        Initialize crawler database manager

        Args:
            database_url: PostgreSQL connection URL (defaults to DATABASE_URL env var)
        """
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.engine = None
        self.SessionLocal = None
        self._initialized = False

        if not self.database_url:
            logger.warning(
                "DATABASE_URL not configured, crawler status will use in-memory storage"
            )
            return

        try:
            self._init_database()
        except Exception as e:
            logger.error(f"Failed to initialize crawler database: {e}")
            logger.warning("Crawler status will use in-memory storage as fallback")

    def _init_database(self):
        """Initialize database connection and create tables"""
        poolclass = QueuePool if "postgresql" in self.database_url else NullPool

        self.engine = create_engine(
            self.database_url,
            poolclass=poolclass,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={
                "connect_timeout": 10,
                "sslmode": "prefer",
                "application_name": "amaniquery_crawler_manager",
            },
            echo=False,
        )

        # Create tables
        Base.metadata.create_all(self.engine)

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._initialized = True
        logger.info("Crawler database manager initialized")

    def get_session(self) -> Session:
        """Get a database session"""
        if not self._initialized or not self.SessionLocal:
            raise RuntimeError("Database not initialized")
        return self.SessionLocal()

    def get_crawler_status(self, crawler_name: str = None) -> dict:
        """
        Get crawler status(es)

        Args:
            crawler_name: Specific crawler name, or None for all crawlers

        Returns:
            Dictionary of crawler statuses
        """
        if not self._initialized:
            return {}

        try:
            with self.get_session() as db:
                if crawler_name:
                    status = (
                        db.query(CrawlerStatus)
                        .filter(CrawlerStatus.crawler_name == crawler_name)
                        .first()
                    )
                    if status:
                        return {
                            status.crawler_name: {
                                "status": status.status,
                                "last_run": (
                                    status.last_run.isoformat() + "Z"
                                    if status.last_run
                                    else None
                                ),
                                "pid": status.pid,
                                "start_time": (
                                    status.start_time.isoformat() + "Z"
                                    if status.start_time
                                    else None
                                ),
                            }
                        }
                    return {}
                else:
                    statuses = db.query(CrawlerStatus).all()
                    result = {}
                    for status in statuses:
                        result[status.crawler_name] = {
                            "status": status.status,
                            "last_run": (
                                status.last_run.isoformat() + "Z"
                                if status.last_run
                                else None
                            ),
                            "pid": status.pid,
                            "start_time": (
                                status.start_time.isoformat() + "Z"
                                if status.start_time
                                else None
                            ),
                        }
                    return result
        except Exception as e:
            logger.error(f"Error getting crawler status from database: {e}")
            return {}

    def update_crawler_status(
        self,
        crawler_name: str,
        status: str,
        last_run: datetime = None,
        pid: int = None,
        start_time: datetime = None,
    ):
        """Update crawler status"""
        if not self._initialized:
            return

        try:
            with self.get_session() as db:
                crawler_status = (
                    db.query(CrawlerStatus)
                    .filter(CrawlerStatus.crawler_name == crawler_name)
                    .first()
                )

                if crawler_status:
                    crawler_status.status = status
                    if last_run is not None:
                        crawler_status.last_run = last_run
                    if pid is not None:
                        crawler_status.pid = pid
                    if start_time is not None:
                        crawler_status.start_time = start_time
                    crawler_status.updated_at = datetime.utcnow()
                else:
                    crawler_status = CrawlerStatus(
                        crawler_name=crawler_name,
                        status=status,
                        last_run=last_run,
                        pid=pid,
                        start_time=start_time,
                    )
                    db.add(crawler_status)

                db.commit()
        except Exception as e:
            logger.error(f"Error updating crawler status in database: {e}")
            if db:
                db.rollback()

    def add_log(self, crawler_name: str, message: str, timestamp: datetime = None):
        """Add a log entry for a crawler"""
        if not self._initialized:
            return

        try:
            with self.get_session() as db:
                log_entry = CrawlerLog(
                    crawler_name=crawler_name,
                    message=message,
                    timestamp=timestamp or datetime.utcnow(),
                )
                db.add(log_entry)
                db.commit()

                # Keep only last 100 logs per crawler (cleanup old logs)
                self._cleanup_old_logs(db, crawler_name)
        except Exception as e:
            logger.error(f"Error adding crawler log to database: {e}")
            if db:
                db.rollback()

    def get_logs(self, crawler_name: str, limit: int = 100) -> list:
        """Get logs for a crawler"""
        if not self._initialized:
            return []

        try:
            with self.get_session() as db:
                logs = (
                    db.query(CrawlerLog)
                    .filter(CrawlerLog.crawler_name == crawler_name)
                    .order_by(CrawlerLog.timestamp.desc())
                    .limit(limit)
                    .all()
                )

                # Format logs as strings with timestamps
                result = []
                for log in reversed(logs):  # Reverse to get chronological order
                    timestamp_str = log.timestamp.isoformat() + "Z"
                    result.append(f"[{timestamp_str}] {log.message}")

                return result
        except Exception as e:
            logger.error(f"Error getting crawler logs from database: {e}")
            return []

    def _cleanup_old_logs(self, db: Session, crawler_name: str, keep_count: int = 100):
        """Keep only the most recent logs for a crawler"""
        try:
            # Count total logs for this crawler
            total_logs = (
                db.query(CrawlerLog)
                .filter(CrawlerLog.crawler_name == crawler_name)
                .count()
            )

            if total_logs > keep_count:
                # Get IDs of logs to delete (oldest ones)
                logs_to_delete = (
                    db.query(CrawlerLog.id)
                    .filter(CrawlerLog.crawler_name == crawler_name)
                    .order_by(CrawlerLog.timestamp.asc())
                    .limit(total_logs - keep_count)
                    .all()
                )

                if logs_to_delete:
                    delete_ids = [log_id[0] for log_id in logs_to_delete]
                    db.query(CrawlerLog).filter(CrawlerLog.id.in_(delete_ids)).delete(
                        synchronize_session=False
                    )
                    db.commit()
        except Exception as e:
            logger.warning(f"Error cleaning up old logs: {e}")
            db.rollback()

    def delete_crawler_status(self, crawler_name: str):
        """Delete crawler status (for cleanup)"""
        if not self._initialized:
            return

        try:
            with self.get_session() as db:
                db.query(CrawlerStatus).filter(
                    CrawlerStatus.crawler_name == crawler_name
                ).delete()
                db.commit()
        except Exception as e:
            logger.error(f"Error deleting crawler status: {e}")
            if db:
                db.rollback()
