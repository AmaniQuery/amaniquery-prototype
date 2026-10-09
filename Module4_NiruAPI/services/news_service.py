"""
News Service - Business logic for news aggregation
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
from loguru import logger
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class NewsService:
    """Service for news article operations"""

    def __init__(self):
        """Initialize news service"""
        self.db_storage = None
        self.vector_store = None

        try:
            from Module3_NiruDB.database_storage import DatabaseStorage

            self.db_storage = DatabaseStorage()
        except Exception as e:
            logger.warning(f"Database storage not available: {e}")

        try:
            from Module3_NiruDB.vector_store import VectorStore

            self.vector_store = VectorStore()
        except Exception as e:
            logger.warning(f"Vector store not available: {e}")

    def get_articles(
        self,
        sources: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        min_quality_score: Optional[float] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[List[dict], int]:
        """
        Get articles with filtering

        Returns:
            Tuple of (articles list, total count)
        """
        if not self.db_storage:
            return [], 0

        try:
            from sqlalchemy import and_, or_
            from Module3_NiruDB.database_storage import RawDocument

            db = self.db_storage.get_db_session()

            # Build query
            query = db.query(RawDocument)

            # Apply filters
            if sources:
                query = query.filter(RawDocument.source_name.in_(sources))

            if categories:
                query = query.filter(RawDocument.category.in_(categories))

            if date_from:
                try:
                    date_from_dt = datetime.fromisoformat(
                        date_from.replace("Z", "+00:00")
                    )
                    query = query.filter(RawDocument.publication_date >= date_from_dt)
                except:
                    pass

            if date_to:
                try:
                    date_to_dt = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
                    query = query.filter(RawDocument.publication_date <= date_to_dt)
                except:
                    pass

            # Get total count
            total = query.count()

            # Apply ordering and pagination
            query = query.order_by(RawDocument.publication_date.desc())
            query = query.offset(offset).limit(limit)

            # Convert to dicts
            articles = []
            for doc in query.all():
                article = self._doc_to_article(doc)
                if article:
                    # Filter by quality score if specified
                    if min_quality_score is not None:
                        quality_score = article.get("quality_score")
                        if quality_score is None or quality_score < min_quality_score:
                            continue
                    articles.append(article)

            db.close()
            return articles, total

        except Exception as e:
            logger.error(f"Error getting articles: {e}")
            return [], 0

    def get_article_by_id(self, article_id: str) -> Optional[dict]:
        """Get article by ID (URL hash)"""
        if not self.db_storage:
            return None

        try:
            from Module3_NiruDB.database_storage import RawDocument

            db = self.db_storage.get_db_session()

            # Try to find by URL hash
            doc = db.query(RawDocument).filter_by(source_url=article_id).first()

            if not doc:
                # Try to find by ID
                try:
                    doc_id = int(article_id)
                    doc = db.query(RawDocument).filter_by(id=doc_id).first()
                except:
                    pass

            db.close()

            if doc:
                return self._doc_to_article(doc)
            return None

        except Exception as e:
            logger.error(f"Error getting article by ID: {e}")
            return None

    def search_articles(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        min_quality_score: Optional[float] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[List[dict], int]:
        """
        Search articles semantically using vector store

        Returns:
            Tuple of (articles list, total count)
        """
        if not self.vector_store:
            # Fallback to text search
            return self._text_search(query, sources, categories, limit, offset)

        try:
            # Use vector store for semantic search
            results = self.vector_store.search(query, top_k=limit + offset)

            # Get article IDs from results
            article_urls = [
                r.get("source_url")
                for r in results[offset : offset + limit]
                if r.get("source_url")
            ]

            # Fetch full articles
            articles = []
            for url in article_urls:
                article = self.get_article_by_id(url)
                if article:
                    # Apply filters
                    if sources and article.get("source_name") not in sources:
                        continue
                    if categories and article.get("category") not in categories:
                        continue
                    if min_quality_score:
                        quality_score = article.get("quality_score")
                        if quality_score is None or quality_score < min_quality_score:
                            continue

                    articles.append(article)

            return articles, len(results)

        except Exception as e:
            logger.error(f"Error searching articles: {e}")
            return self._text_search(query, sources, categories, limit, offset)

    def _text_search(
        self,
        query: str,
        sources: Optional[List[str]],
        categories: Optional[List[str]],
        limit: int,
        offset: int,
    ) -> Tuple[List[dict], int]:
        """Fallback text search"""
        if not self.db_storage:
            return [], 0

        try:
            from Module3_NiruDB.database_storage import RawDocument
            from sqlalchemy import or_

            db = self.db_storage.get_db_session()

            # Simple text search in title and content
            query_lower = query.lower()
            results = db.query(RawDocument).filter(
                or_(
                    RawDocument.title.ilike(f"%{query}%"),
                    RawDocument.raw_content.ilike(f"%{query}%"),
                )
            )

            if sources:
                results = results.filter(RawDocument.source_name.in_(sources))

            if categories:
                results = results.filter(RawDocument.category.in_(categories))

            total = results.count()
            results = results.offset(offset).limit(limit)

            articles = [
                self._doc_to_article(doc)
                for doc in results.all()
                if self._doc_to_article(doc)
            ]
            db.close()

            return articles, total

        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return [], 0

    def get_sources(self) -> List[dict]:
        """Get list of all news sources"""
        if not self.db_storage:
            return []

        try:
            from sqlalchemy import func
            from Module3_NiruDB.database_storage import RawDocument

            db = self.db_storage.get_db_session()

            sources = (
                db.query(
                    RawDocument.source_name,
                    func.count(RawDocument.id).label("article_count"),
                )
                .group_by(RawDocument.source_name)
                .all()
            )

            db.close()

            return [{"name": name, "article_count": count} for name, count in sources]

        except Exception as e:
            logger.error(f"Error getting sources: {e}")
            return []

    def get_categories(self) -> List[str]:
        """Get list of all categories"""
        if not self.db_storage:
            return []

        try:
            from Module3_NiruDB.database_storage import RawDocument

            db = self.db_storage.get_db_session()

            categories = db.query(RawDocument.category).distinct().all()
            db.close()

            return [cat[0] for cat in categories if cat[0]]

        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []

    def _doc_to_article(self, doc) -> Optional[dict]:
        """Convert database document to article dict"""
        try:
            # Extract quality score from metadata if available
            quality_score = None
            if doc.metadata_json and isinstance(doc.metadata_json, dict):
                quality_score = doc.metadata_json.get("quality_score")

            article_id = hashlib.md5(doc.source_url.encode()).hexdigest()

            return {
                "id": article_id,
                "url": doc.source_url,
                "title": doc.title or "Untitled",
                "content": doc.raw_content or "",
                "summary": (
                    doc.metadata_json.get("summary", "") if doc.metadata_json else ""
                ),
                "author": doc.author,
                "publication_date": (
                    doc.publication_date.isoformat() if doc.publication_date else None
                ),
                "source_name": doc.source_name or "Unknown",
                "category": doc.category or "Unknown",
                "quality_score": quality_score,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
            }
        except Exception as e:
            logger.error(f"Error converting doc to article: {e}")
            return None
