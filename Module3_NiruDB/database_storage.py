"""
Database Storage - Store raw and processed data in PostgreSQL
"""

import os
import json
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
from loguru import logger
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    JSON,
    Boolean,
    LargeBinary,
)
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from Module3_NiruDB.connection_pool import pool_manager, EngineConfig

Base = declarative_base()


class RawDocument(Base):
    """Raw document storage"""

    __tablename__ = "raw_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_url = Column(String(500), nullable=False, index=True)
    title = Column(String(500))
    category = Column(String(100), index=True)
    source_name = Column(String(200), index=True)
    author = Column(String(200))
    publication_date = Column(DateTime)
    crawl_date = Column(DateTime, default=datetime.utcnow, index=True)
    content_type = Column(String(50))  # html, pdf, text
    raw_content = Column(Text)  # HTML content or text
    raw_html = Column(Text)  # Original HTML if available
    pdf_path = Column(String(500))  # Path to PDF file if applicable
    metadata_json = Column(JSON)  # Additional metadata as JSON
    processed = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ProcessedChunk(Base):
    """Processed document chunks storage"""

    __tablename__ = "processed_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(String(100), nullable=False, unique=True, index=True)
    doc_id = Column(String(100), index=True)  # Reference to original document
    source_url = Column(String(500), index=True)
    title = Column(String(500))
    category = Column(String(100), index=True)
    source_name = Column(String(200), index=True)
    author = Column(String(200))
    publication_date = Column(DateTime)
    crawl_date = Column(DateTime, index=True)
    content_type = Column(String(50))
    text = Column(Text, nullable=False)
    chunk_index = Column(Integer)
    total_chunks = Column(Integer)
    embedding = Column(JSON)  # Store embedding as JSON array
    metadata_json = Column(JSON)  # Additional metadata
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseStorage:
    """Database storage for raw and processed data"""

    def __init__(self, database_url: Optional[str] = None):
        """Initialize database connection"""
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
        # Use unified connection pool manager for all DB connections
        self.engine_group = pool_manager.get_engine_group(
            database_url=database_url,
            config=EngineConfig(
                pool_size=100,
                max_overflow=200,
                pool_recycle=300,
                pool_pre_ping=True,
                pool_timeout=30,
                connect_timeout=10,
            ),
        )
        self.engine = self.engine_group.writer_engine
        self.SessionLocal = self.engine_group.writer_session_factory

        # Create tables
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database storage initialized")

    def get_db_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()

    def save_raw_documents(
        self, documents: List[Dict], notification_callback=None
    ) -> int:
        """Save raw documents to database"""
        saved_count = 0
        new_articles = []

        # Use default callback if available and none provided
        if notification_callback is None:
            notification_callback = getattr(
                self, "_default_notification_callback", None
            )
            if notification_callback is None:
                # Try to get from module-level default
                try:
                    import Module3_NiruDB.database_storage as db_module

                    notification_callback = getattr(
                        db_module, "default_notification_callback", None
                    )
                except:
                    pass

        with self.get_db_session() as db:
            try:
                for doc in documents:
                    # Check if document already exists
                    existing = (
                        db.query(RawDocument)
                        .filter_by(source_url=doc.get("url"))
                        .first()
                    )
                    if existing:
                        logger.debug(f"Document already exists: {doc.get('url')}")
                        continue

                    # Create new raw document
                    raw_doc = RawDocument(
                        source_url=doc.get("url", ""),
                        title=doc.get("title", "Untitled"),
                        category=doc.get("category", "Unknown"),
                        source_name=doc.get("source_name", "Unknown"),
                        author=doc.get("author"),
                        publication_date=self._parse_date(doc.get("publication_date")),
                        crawl_date=self._parse_date(doc.get("crawl_date")),
                        content_type=doc.get("content_type", "html"),
                        raw_content=doc.get("content", ""),
                        raw_html=doc.get("raw_html"),
                        pdf_path=doc.get("pdf_path"),
                        metadata_json=doc.get("metadata", {}),
                        processed=False,
                    )

                    db.add(raw_doc)
                    saved_count += 1

                    # Collect article data for notifications
                    new_articles.append(
                        {
                            "url": doc.get("url", ""),
                            "title": doc.get("title", "Untitled"),
                            "category": doc.get("category", "Unknown"),
                            "source_name": doc.get("source_name", "Unknown"),
                            "author": doc.get("author"),
                            "publication_date": doc.get("publication_date"),
                            "summary": (
                                doc.get("metadata", {}).get("summary", "")
                                if isinstance(doc.get("metadata"), dict)
                                else ""
                            ),
                        }
                    )

                db.commit()
                logger.info(f"Saved {saved_count} raw documents to database")

                # Trigger notifications for new articles
                if notification_callback and new_articles:
                    for article in new_articles:
                        try:
                            notification_callback(article)
                        except Exception as e:
                            logger.error(
                                f"Error triggering notification for article {article.get('url')}: {e}"
                            )

            except Exception as e:
                db.rollback()
                logger.error(f"Error saving raw documents: {e}")
                raise

        return saved_count

    def save_processed_chunks(self, chunks: List[Dict]) -> int:
        """Save processed chunks to database with retry logic for connection issues"""
        saved_count = 0
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            db = None
            try:
                db = self.get_db_session()
                for chunk in chunks:
                    existing = (
                        db.query(ProcessedChunk)
                        .filter_by(chunk_id=chunk.get("chunk_id"))
                        .first()
                    )
                    if existing:
                        continue

                    processed_chunk = ProcessedChunk(
                        chunk_id=chunk.get("chunk_id", ""),
                        doc_id=chunk.get("doc_id", ""),
                        source_url=chunk.get("source_url", ""),
                        title=chunk.get("title", "Untitled"),
                        category=chunk.get("category", "Unknown"),
                        source_name=chunk.get("source_name", "Unknown"),
                        author=chunk.get("author"),
                        publication_date=self._parse_date(
                            chunk.get("publication_date")
                        ),
                        crawl_date=self._parse_date(chunk.get("crawl_date")),
                        content_type=chunk.get("content_type", "html"),
                        text=chunk.get("text", ""),
                        chunk_index=chunk.get("chunk_index", 0),
                        total_chunks=chunk.get("total_chunks", 1),
                        embedding=chunk.get("embedding", []),
                        metadata_json=chunk.get("metadata", {}),
                    )
                    db.add(processed_chunk)
                    saved_count += 1

                db.commit()
                logger.info(f"Saved {saved_count} processed chunks")
                return saved_count

            except Exception as e:
                if db is not None:
                    try:
                        db.rollback()
                    except Exception:
                        pass
                error_str = str(e).lower()
                if any(
                    k in error_str
                    for k in ["ssl", "connection", "closed", "timeout", "broken"]
                ):
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Conn error (attempt {attempt+1}/{max_retries}), retrying..."
                        )
                        import time

                        time.sleep(retry_delay * (attempt + 1))
                        continue
                logger.error(f"Save failed after {max_retries} attempts: {e}")
                raise
            finally:
                if db is not None:
                    try:
                        db.close()
                    except Exception:
                        pass

        return saved_count

    def mark_raw_documents_processed(self, urls: List[str]):
        """Mark raw documents as processed"""
        with self.get_db_session() as db:
            try:
                db.query(RawDocument).filter(RawDocument.source_url.in_(urls)).update(
                    {"processed": True}
                )
                db.commit()
                logger.info(f"Marked {len(urls)} documents as processed")

            except Exception as e:
                db.rollback()
                logger.error(f"Error marking documents as processed: {e}")

    def get_raw_documents(
        self, category: Optional[str] = None, limit: int = 100
    ) -> List[Dict]:
        """Get raw documents from database"""
        with self.get_db_session() as db:
            try:
                query = db.query(RawDocument)
                if category:
                    query = query.filter_by(category=category)
                query = query.filter_by(processed=False).limit(limit)

                documents = []
                for doc in query.all():
                    documents.append(
                        {
                            "id": doc.id,
                            "url": doc.source_url,
                            "title": doc.title,
                            "category": doc.category,
                            "source_name": doc.source_name,
                            "author": doc.author,
                            "publication_date": (
                                doc.publication_date.isoformat()
                                if doc.publication_date
                                else None
                            ),
                            "crawl_date": (
                                doc.crawl_date.isoformat() if doc.crawl_date else None
                            ),
                            "content_type": doc.content_type,
                            "content": doc.raw_content,
                            "raw_html": doc.raw_html,
                            "pdf_path": doc.pdf_path,
                            "metadata": doc.metadata_json or {},
                            "processed": doc.processed,
                        }
                    )

                return documents

            except Exception as e:
                logger.error(f"Error getting raw documents: {e}")
                return []

    def get_processed_chunks(
        self, category: Optional[str] = None, limit: int = 1000
    ) -> List[Dict]:
        """Get processed chunks from database"""
        with self.get_db_session() as db:
            try:
                query = db.query(ProcessedChunk)
                if category:
                    query = query.filter_by(category=category)
                query = query.limit(limit)

                chunks = []
                for chunk in query.all():
                    chunks.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "doc_id": chunk.doc_id,
                            "source_url": chunk.source_url,
                            "title": chunk.title,
                            "category": chunk.category,
                            "source_name": chunk.source_name,
                            "author": chunk.author,
                            "publication_date": (
                                chunk.publication_date.isoformat()
                                if chunk.publication_date
                                else None
                            ),
                            "crawl_date": (
                                chunk.crawl_date.isoformat()
                                if chunk.crawl_date
                                else None
                            ),
                            "content_type": chunk.content_type,
                            "text": chunk.text,
                            "chunk_index": chunk.chunk_index,
                            "total_chunks": chunk.total_chunks,
                            "embedding": chunk.embedding or [],
                            "metadata": chunk.metadata_json or {},
                        }
                    )

                return chunks

            except Exception as e:
                logger.error(f"Error getting processed chunks: {e}")
                return []

    def get_stats(self) -> Dict:
        """Get database statistics"""
        with self.get_db_session() as db:
            try:
                raw_count = db.query(RawDocument).count()
                processed_count = db.query(ProcessedChunk).count()
                processed_docs = db.query(RawDocument).filter_by(processed=True).count()

                # Get category breakdown
                from sqlalchemy import func

                raw_categories = (
                    db.query(
                        RawDocument.category, func.count(RawDocument.id).label("count")
                    )
                    .group_by(RawDocument.category)
                    .all()
                )

                processed_categories = (
                    db.query(
                        ProcessedChunk.category,
                        func.count(ProcessedChunk.id).label("count"),
                    )
                    .group_by(ProcessedChunk.category)
                    .all()
                )

                return {
                    "raw_documents": {
                        "total": raw_count,
                        "processed": processed_docs,
                        "unprocessed": raw_count - processed_docs,
                        "categories": dict(raw_categories),
                    },
                    "processed_chunks": {
                        "total": processed_count,
                        "categories": dict(processed_categories),
                    },
                }

            except Exception as e:
                logger.error(f"Error getting database stats: {e}")
                return {
                    "raw_documents": {
                        "total": 0,
                        "processed": 0,
                        "unprocessed": 0,
                        "categories": {},
                    },
                    "processed_chunks": {"total": 0, "categories": {}},
                }

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime object"""
        if not date_str:
            return None

        try:
            # Try different date formats
            formats = [
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%Y/%m/%d",
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue

            # If all formats fail, try to parse as ISO
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))

        except Exception:
            logger.warning(f"Could not parse date: {date_str}")
            return None
