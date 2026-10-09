"""
Deduplication System for News Articles
Handles URL-based and content-based deduplication
"""

import hashlib
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
from sqlalchemy import Column, String, DateTime, Float, Index, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os

# Load environment variables
try:
    from dotenv import load_dotenv

    # Load from project root
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

Base = declarative_base()


class ArticleDeduplication(Base):
    """Table for tracking article deduplication"""

    __tablename__ = "article_deduplication"

    id = Column(String(64), primary_key=True)  # URL hash
    url = Column(String(500), nullable=False, unique=True, index=True)
    url_hash = Column(String(64), nullable=False, index=True)
    content_hash = Column(String(64), index=True)  # Hash of article content
    title_hash = Column(
        String(64), index=True
    )  # Hash of title for near-duplicate detection
    source_name = Column(String(200), index=True)
    publication_date = Column(DateTime, index=True)
    first_seen = Column(DateTime, default=datetime.utcnow, index=True)
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    crawl_count = Column(Float, default=1.0)  # Track how many times we've seen this

    # Index for faster lookups
    __table_args__ = (
        Index("idx_url_hash", "url_hash"),
        Index("idx_content_hash", "content_hash"),
        Index("idx_title_hash", "title_hash"),
        Index("idx_source_date", "source_name", "publication_date"),
    )


class DeduplicationEngine:
    """
    Deduplication engine for news articles
    Uses URL hashing and content similarity for deduplication
    """

    def __init__(self, database_url: Optional[str] = None):
        """Initialize deduplication engine"""
        if database_url is None:
            database_url = os.getenv(
                "DATABASE_URL", "postgresql://localhost/amaniquery"
            )

        # Handle Neon connection pooling
        if "neon.tech" in database_url and "pooler" in database_url:
            unpooled_url = os.getenv("DATABASE_URL_UNPOOLED")
            if unpooled_url:
                database_url = unpooled_url

        self.database_url = database_url

        # Configure SSL for Neon and other cloud databases
        connect_args = {}
        if "neon.tech" in database_url or "sslmode=require" in database_url:
            connect_args["connect_args"] = {"sslmode": "require"}

        self.engine = create_engine(database_url, echo=False, **connect_args)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

        # Create tables
        Base.metadata.create_all(bind=self.engine)
        logger.info("Deduplication engine initialized")

    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()

    def is_duplicate(
        self, url: str, content: Optional[str] = None, title: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if article is a duplicate

        Args:
            url: Article URL
            content: Article content (optional, for content-based dedup)
            title: Article title (optional, for near-duplicate detection)

        Returns:
            Tuple of (is_duplicate, reason)
        """
        url_hash = self._hash_string(url)

        with self.get_session() as db:
            # Check for exact URL match
            existing = (
                db.query(ArticleDeduplication).filter_by(url_hash=url_hash).first()
            )
            if existing:
                # Update last_seen and increment crawl_count
                existing.last_seen = datetime.utcnow()
                existing.crawl_count += 1
                db.commit()
                return True, "exact_url_match"

            # Check for content-based deduplication if content provided
            if content:
                content_hash = self._hash_string(content)
                existing_content = (
                    db.query(ArticleDeduplication)
                    .filter_by(content_hash=content_hash)
                    .first()
                )
                if existing_content:
                    # Check if it's the same article (same content, different URL)
                    # Allow if URLs are from same domain (might be canonical URLs)
                    from urllib.parse import urlparse

                    existing_domain = urlparse(existing_content.url).netloc
                    new_domain = urlparse(url).netloc

                    if existing_domain == new_domain:
                        # Same domain, likely duplicate
                        return True, "content_match_same_domain"
                    else:
                        # Different domain, might be syndicated content
                        # Log but don't block (could be legitimate republishing)
                        logger.debug(
                            f"Content match across domains: {existing_content.url} vs {url}"
                        )

            # Check for title-based near-duplicates if title provided
            if title:
                title_hash = self._hash_string(title.lower().strip())
                existing_title = (
                    db.query(ArticleDeduplication)
                    .filter_by(title_hash=title_hash)
                    .first()
                )
                if existing_title:
                    # Check if published recently (within 24 hours)
                    if existing_title.publication_date:
                        time_diff = datetime.utcnow() - existing_title.publication_date
                        if time_diff < timedelta(hours=24):
                            # Same title published recently, likely duplicate
                            return True, "title_match_recent"

        return False, None

    def register_article(
        self,
        url: str,
        content: Optional[str] = None,
        title: Optional[str] = None,
        source_name: Optional[str] = None,
        publication_date: Optional[datetime] = None,
    ) -> bool:
        """
        Register a new article in the deduplication database

        Args:
            url: Article URL
            content: Article content
            title: Article title
            source_name: Source name
            publication_date: Publication date

        Returns:
            True if registered successfully, False if duplicate
        """
        is_dup, reason = self.is_duplicate(url, content, title)
        if is_dup:
            logger.debug(f"Skipping duplicate article: {url} (reason: {reason})")
            return False

        url_hash = self._hash_string(url)
        content_hash = self._hash_string(content) if content else None
        title_hash = self._hash_string(title.lower().strip()) if title else None

        with self.get_session() as db:
            try:
                article = ArticleDeduplication(
                    id=url_hash,
                    url=url,
                    url_hash=url_hash,
                    content_hash=content_hash,
                    title_hash=title_hash,
                    source_name=source_name,
                    publication_date=publication_date,
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    crawl_count=1.0,
                )
                db.add(article)
                db.commit()
                logger.debug(f"Registered new article: {url}")
                return True
            except Exception as e:
                db.rollback()
                # If it's a unique constraint violation, it's a duplicate
                if (
                    "unique constraint" in str(e).lower()
                    or "duplicate key" in str(e).lower()
                ):
                    logger.debug(f"Article already exists: {url}")
                    return False
                logger.error(f"Error registering article: {e}")
                return False

    def get_duplicate_stats(self, days: int = 7) -> Dict:
        """
        Get statistics about duplicates

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with duplicate statistics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        with self.get_session() as db:
            total_articles = db.query(ArticleDeduplication).count()
            recent_articles = (
                db.query(ArticleDeduplication)
                .filter(ArticleDeduplication.first_seen >= cutoff_date)
                .count()
            )

            # Articles seen multiple times
            duplicates = (
                db.query(ArticleDeduplication)
                .filter(ArticleDeduplication.crawl_count > 1.0)
                .count()
            )

            # Average crawl count
            from sqlalchemy import func

            avg_crawl_count = (
                db.query(func.avg(ArticleDeduplication.crawl_count)).scalar() or 0.0
            )

            return {
                "total_articles": total_articles,
                "recent_articles": recent_articles,
                "duplicates": duplicates,
                "average_crawl_count": float(avg_crawl_count),
                "unique_articles": total_articles - duplicates,
            }

    def cleanup_old_articles(self, days: int = 90):
        """
        Clean up old article records

        Args:
            days: Delete articles older than this many days
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        with self.get_session() as db:
            try:
                deleted = (
                    db.query(ArticleDeduplication)
                    .filter(ArticleDeduplication.last_seen < cutoff_date)
                    .delete()
                )
                db.commit()
                logger.info(f"Cleaned up {deleted} old article records")
                return deleted
            except Exception as e:
                db.rollback()
                logger.error(f"Error cleaning up old articles: {e}")
                return 0

    def _hash_string(self, text: str) -> str:
        """Generate SHA256 hash of string"""
        if not text:
            return ""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
