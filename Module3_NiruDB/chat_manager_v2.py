"""
Chat Database Manager v2 - High Performance Edition
Optimized for 5000+ concurrent operations with connection pooling,
async support, batch operations, and caching.
"""

import os
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager, asynccontextmanager
import uuid
from collections import OrderedDict
import threading
from functools import lru_cache

from loguru import logger
from sqlalchemy import text, Index
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, OperationalError

from Module3_NiruDB.connection_pool import pool_manager, EngineConfig

# Async support
try:
    from sqlalchemy.ext.asyncio import AsyncSession

    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

from Module3_NiruDB.chat_models import (
    ChatSession,
    ChatMessage,
    UserFeedback,
    Base,
    ChatSessionResponse,
    ChatMessageResponse,
    FeedbackResponse,
)

# =============================================================================
# LRU CACHE WITH TTL
# =============================================================================


class TTLCache:
    """Thread-safe LRU cache with TTL expiration"""

    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 300):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, datetime] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None

            # Check TTL
            if datetime.utcnow() - self._timestamps[key] > timedelta(
                seconds=self.ttl_seconds
            ):
                del self._cache[key]
                del self._timestamps[key]
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

    def set(self, key: str, value: Any):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.maxsize:
                    # Remove oldest
                    oldest = next(iter(self._cache))
                    del self._cache[oldest]
                    del self._timestamps[oldest]

            self._cache[key] = value
            self._timestamps[key] = datetime.utcnow()

    def delete(self, key: str):
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()


# =============================================================================
# HIGH-PERFORMANCE CHAT DATABASE MANAGER
# =============================================================================


class ChatDatabaseManagerV2:
    """
    High-performance chat database manager optimized for 5000+ operations.

    Features:
    - Connection pooling with QueuePool
    - Session caching with TTL
    - Batch operations for bulk inserts
    - Async support for non-blocking operations
    - Optimized queries with proper indexing
    - Thread-safe operations
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = 20,
        max_overflow: int = 30,
        cache_size: int = 1000,
        cache_ttl: int = 300,
    ):
        """
        Initialize high-performance chat database manager.

        Args:
            database_url: PostgreSQL connection URL
            pool_size: Connection pool size (default: 20)
            max_overflow: Max overflow connections (default: 30)
            cache_size: Session cache size (default: 1000)
            cache_ttl: Cache TTL in seconds (default: 300)
        """
        if database_url is None:
            database_url = os.getenv(
                "DATABASE_URL", "postgresql://localhost/amaniquery"
            )

            # For Neon databases, prefer unpooled for direct connections
            if "neon.tech" in database_url and "pooler" in database_url:
                unpooled_url = os.getenv("DATABASE_URL_UNPOOLED")
                if unpooled_url:
                    database_url = unpooled_url
                    logger.info("Using unpooled Neon connection for chat database")

        self.database_url = database_url

        # Use unified connection pool manager
        self.engine_group = pool_manager.get_engine_group(
            database_url=database_url,
            config=EngineConfig(
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_recycle=1800,
            ),
        )
        self.engine = self.engine_group.writer_engine
        self.SessionFactory = self.engine_group.writer_session_factory

        # Initialize caches
        self._session_cache = TTLCache(maxsize=cache_size, ttl_seconds=cache_ttl)
        self._message_count_cache = TTLCache(maxsize=cache_size * 2, ttl_seconds=60)

        # Ensure tables and indexes exist
        self._ensure_schema()

        # Async support
        self._async_available = ASYNC_AVAILABLE
        if self._async_available:
            try:
                self.AsyncSessionFactory = pool_manager.get_async_session_factory(
                    database_url
                )
            except Exception as e:
                logger.warning(f"Async engine initialization failed: {e}")
                self._async_available = False

        logger.info(
            f"ChatDatabaseManagerV2 initialized (pool_size={pool_size}, cache_size={cache_size})"
        )

    def _ensure_schema(self):
        """Ensure tables and indexes exist"""
        try:
            Base.metadata.create_all(self.engine)

            # Create additional indexes for performance
            with self.engine.connect() as conn:
                # Index on session_id for messages (most common query)
                try:
                    conn.execute(
                        text(
                            "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id "
                            "ON chat_messages(session_id)"
                        )
                    )
                except Exception:
                    pass

                # Index on created_at for ordering
                try:
                    conn.execute(
                        text(
                            "CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at "
                            "ON chat_messages(session_id, created_at)"
                        )
                    )
                except Exception:
                    pass

                # Index on user_id for sessions
                try:
                    conn.execute(
                        text(
                            "CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id "
                            "ON chat_sessions(user_id)"
                        )
                    )
                except Exception:
                    pass

                # Index on updated_at for listing sessions
                try:
                    conn.execute(
                        text(
                            "CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at "
                            "ON chat_sessions(updated_at DESC)"
                        )
                    )
                except Exception:
                    pass

                conn.commit()

        except Exception as e:
            logger.warning(f"Schema setup warning: {e}")

    @contextmanager
    def _get_db_session(self) -> Session:
        """Get a database session from the pool with proper cleanup"""
        session = self.SessionFactory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    # =========================================================================
    # SESSION OPERATIONS
    # =========================================================================

    def create_session(
        self, title: Optional[str] = None, user_id: Optional[str] = None
    ) -> str:
        """Create a new chat session - optimized for speed"""
        session_id = str(uuid.uuid4())

        with self._get_db_session() as db:
            chat_session = ChatSession(
                id=session_id,
                title=title,
                user_id=user_id,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                is_active=True,
            )
            db.add(chat_session)

        # Cache the new session
        self._session_cache.set(
            session_id,
            {
                "id": session_id,
                "title": title,
                "user_id": user_id,
                "is_active": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            },
        )
        self._message_count_cache.set(session_id, 0)

        logger.debug(f"Created chat session: {session_id}")
        return session_id

    def create_session_with_message(
        self,
        content: str,
        title: Optional[str] = None,
        user_id: Optional[str] = None,
        role: str = "user",
    ) -> Tuple[str, str]:
        """
        Create session and first message in a single transaction.
        This is the optimized path for new conversations.

        Returns:
            Tuple of (session_id, message_id)
        """
        session_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Auto-generate title from content if not provided
        if title is None:
            title = (
                content[:50].rsplit(" ", 1)[0] + "..." if len(content) > 50 else content
            )

        with self._get_db_session() as db:
            # Create session
            chat_session = ChatSession(
                id=session_id,
                title=title,
                user_id=user_id,
                created_at=now,
                updated_at=now,
                is_active=True,
            )
            db.add(chat_session)

            # Create message
            message = ChatMessage(
                id=message_id,
                session_id=session_id,
                role=role,
                content=content,
                created_at=now,
            )
            db.add(message)

        # Update caches
        self._session_cache.set(
            session_id,
            {
                "id": session_id,
                "title": title,
                "user_id": user_id,
                "is_active": True,
                "created_at": now,
                "updated_at": now,
            },
        )
        self._message_count_cache.set(session_id, 1)

        logger.debug(f"Created session {session_id} with message {message_id}")
        return session_id, message_id

    def get_session(self, session_id: str) -> Optional[ChatSessionResponse]:
        """Get chat session by ID with caching"""
        # Check cache first
        cached = self._session_cache.get(session_id)
        if cached:
            message_count = self._get_message_count(session_id)
            return ChatSessionResponse(
                id=cached["id"],
                title=cached.get("title"),
                created_at=cached["created_at"],
                updated_at=cached["updated_at"],
                is_active=cached["is_active"],
                message_count=message_count,
            )

        # Query database
        with self._get_db_session() as db:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if not session:
                return None

            message_count = self._get_message_count(session_id, db)

            # Cache the result
            self._session_cache.set(
                session_id,
                {
                    "id": session.id,
                    "title": session.title,
                    "user_id": session.user_id,
                    "is_active": session.is_active,
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                },
            )

            return ChatSessionResponse(
                id=session.id,
                title=session.title,
                created_at=session.created_at,
                updated_at=session.updated_at,
                is_active=session.is_active,
                message_count=message_count,
            )

    def _get_message_count(self, session_id: str, db: Optional[Session] = None) -> int:
        """Get message count with caching"""
        cached_count = self._message_count_cache.get(session_id)
        if cached_count is not None:
            return cached_count

        if db is None:
            with self._get_db_session() as db:
                count = (
                    db.query(ChatMessage)
                    .filter(ChatMessage.session_id == session_id)
                    .count()
                )
        else:
            count = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .count()
            )

        self._message_count_cache.set(session_id, count)
        return count

    def get_session_with_user(self, session_id: str) -> Optional[ChatSession]:
        """Get chat session with user_id for ownership verification"""
        with self._get_db_session() as db:
            return db.query(ChatSession).filter(ChatSession.id == session_id).first()

    def update_session_title(self, session_id: str, title: str):
        """Update session title"""
        with self._get_db_session() as db:
            db.query(ChatSession).filter(ChatSession.id == session_id).update(
                {"title": title, "updated_at": datetime.utcnow()},
                synchronize_session=False,
            )

        # Update cache
        cached = self._session_cache.get(session_id)
        if cached:
            cached["title"] = title
            cached["updated_at"] = datetime.utcnow()
            self._session_cache.set(session_id, cached)

    def list_sessions(
        self, user_id: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> List[ChatSessionResponse]:
        """List chat sessions with pagination"""
        try:
            with self._get_db_session() as db:
                query = db.query(ChatSession)
                if user_id:
                    query = query.filter(ChatSession.user_id == user_id)

                sessions = (
                    query.order_by(ChatSession.updated_at.desc())
                    .offset(offset)
                    .limit(limit)
                    .all()
                )

                # Batch get message counts
                session_ids = [s.id for s in sessions]

                # Use single query for all message counts
                from sqlalchemy import func

                counts = (
                    db.query(ChatMessage.session_id, func.count(ChatMessage.id))
                    .filter(ChatMessage.session_id.in_(session_ids))
                    .group_by(ChatMessage.session_id)
                    .all()
                )

                count_map = {sid: cnt for sid, cnt in counts}

                result = []
                for session in sessions:
                    msg_count = count_map.get(session.id, 0)
                    self._message_count_cache.set(session.id, msg_count)

                    result.append(
                        ChatSessionResponse(
                            id=session.id,
                            title=session.title,
                            created_at=session.created_at,
                            updated_at=session.updated_at,
                            is_active=session.is_active,
                            message_count=msg_count,
                        )
                    )

                return result

        except Exception as e:
            logger.error(f"Database error in list_sessions: {e}")
            return []

    def delete_session(self, session_id: str):
        """Delete a chat session and all its messages"""
        with self._get_db_session() as db:
            # Delete messages first (cascade should handle this, but being explicit)
            db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
            db.query(ChatSession).filter(ChatSession.id == session_id).delete()

        # Clear caches
        self._session_cache.delete(session_id)
        self._message_count_cache.delete(session_id)

        logger.debug(f"Deleted chat session: {session_id}")

    # =========================================================================
    # MESSAGE OPERATIONS
    # =========================================================================

    def add_message(
        self,
        session_id: str,
        content: str,
        role: str,
        token_count: Optional[int] = None,
        model_used: Optional[str] = None,
        sources: Optional[List[Dict]] = None,
        attachments: Optional[List[Dict]] = None,
    ) -> str:
        """Add a message to a chat session - optimized"""
        message_id = str(uuid.uuid4())
        now = datetime.utcnow()

        with self._get_db_session() as db:
            message = ChatMessage(
                id=message_id,
                session_id=session_id,
                role=role,
                content=content,
                created_at=now,
                token_count=token_count,
                model_used=model_used,
                sources=sources,
                attachments=attachments,
            )
            db.add(message)

            # Update session timestamp
            db.query(ChatSession).filter(ChatSession.id == session_id).update(
                {"updated_at": now}, synchronize_session=False
            )

        # Update caches
        cached_count = self._message_count_cache.get(session_id)
        if cached_count is not None:
            self._message_count_cache.set(session_id, cached_count + 1)

        cached_session = self._session_cache.get(session_id)
        if cached_session:
            cached_session["updated_at"] = now
            self._session_cache.set(session_id, cached_session)

        logger.debug(f"Added {role} message {message_id} to session {session_id}")
        return message_id

    def add_messages_batch(self, messages: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple messages in a single transaction.

        Args:
            messages: List of dicts with keys:
                - session_id, content, role, token_count, model_used, sources, attachments

        Returns:
            List of message IDs
        """
        message_ids = []
        now = datetime.utcnow()
        session_ids = set()

        with self._get_db_session() as db:
            for msg_data in messages:
                message_id = str(uuid.uuid4())
                message_ids.append(message_id)
                session_ids.add(msg_data["session_id"])

                message = ChatMessage(
                    id=message_id,
                    session_id=msg_data["session_id"],
                    role=msg_data["role"],
                    content=msg_data["content"],
                    created_at=now,
                    token_count=msg_data.get("token_count"),
                    model_used=msg_data.get("model_used"),
                    sources=msg_data.get("sources"),
                    attachments=msg_data.get("attachments"),
                )
                db.add(message)

            # Batch update session timestamps
            db.query(ChatSession).filter(ChatSession.id.in_(session_ids)).update(
                {"updated_at": now}, synchronize_session=False
            )

        # Invalidate caches for affected sessions
        for sid in session_ids:
            self._message_count_cache.delete(sid)

        logger.debug(f"Batch added {len(messages)} messages")
        return message_ids

    def get_messages(
        self, session_id: str, limit: int = 100, offset: int = 0
    ) -> List[ChatMessageResponse]:
        """Get messages for a session with pagination"""
        with self._get_db_session() as db:
            messages = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at)
                .offset(offset)
                .limit(limit)
                .all()
            )

            return [
                ChatMessageResponse(
                    id=msg.id,
                    session_id=msg.session_id,
                    role=msg.role,
                    content=msg.content,
                    created_at=msg.created_at,
                    token_count=msg.token_count,
                    model_used=msg.model_used,
                    sources=msg.sources,
                    attachments=msg.attachments,
                    feedback_type=msg.feedback_type,
                )
                for msg in messages
            ]

    def get_messages_by_message_id(self, message_id: str) -> List[ChatMessageResponse]:
        """Get a message by its ID"""
        with self._get_db_session() as db:
            message = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
            if not message:
                return []

            return [
                ChatMessageResponse(
                    id=message.id,
                    session_id=message.session_id,
                    role=message.role,
                    content=message.content,
                    created_at=message.created_at,
                    token_count=message.token_count,
                    model_used=message.model_used,
                    sources=message.sources,
                    attachments=message.attachments,
                    feedback_type=message.feedback_type,
                )
            ]

    def generate_session_title(self, session_id: str) -> str:
        """Generate a title from the first user message"""
        with self._get_db_session() as db:
            first_message = (
                db.query(ChatMessage)
                .filter(
                    ChatMessage.session_id == session_id, ChatMessage.role == "user"
                )
                .order_by(ChatMessage.created_at)
                .first()
            )

            if first_message:
                content = first_message.content.strip()
                title = (
                    content[:50].rsplit(" ", 1)[0] + "..."
                    if len(content) > 50
                    else content
                )

                db.query(ChatSession).filter(ChatSession.id == session_id).update(
                    {"title": title}, synchronize_session=False
                )

                # Update cache
                cached = self._session_cache.get(session_id)
                if cached:
                    cached["title"] = title
                    self._session_cache.set(session_id, cached)

                return title

        return "New Chat"

    # =========================================================================
    # FEEDBACK OPERATIONS
    # =========================================================================

    def add_feedback(
        self,
        message_id: str,
        feedback_type: str,
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> int:
        """Add feedback for a message"""
        with self._get_db_session() as db:
            # Check if message exists
            message = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
            if not message:
                raise ValueError(f"Message {message_id} not found")

            # Check for existing feedback
            existing = (
                db.query(UserFeedback)
                .filter(
                    UserFeedback.message_id == message_id,
                    UserFeedback.feedback_type == feedback_type,
                )
                .first()
            )

            if existing:
                if comment is not None:
                    existing.content = comment
                db.commit()
                return existing.id

            # Create new feedback
            feedback = UserFeedback(
                user_id=user_id,
                message_id=message_id,
                feedback_type=feedback_type,
                content=comment,
                created_at=datetime.utcnow(),
            )
            db.add(feedback)
            db.flush()

            return feedback.id

    def get_feedback_stats(self) -> Dict[str, int]:
        """Get feedback statistics"""
        with self._get_db_session() as db:
            from sqlalchemy import func

            counts = (
                db.query(UserFeedback.feedback_type, func.count(UserFeedback.id))
                .group_by(UserFeedback.feedback_type)
                .all()
            )

            return {ft: cnt for ft, cnt in counts}

    # =========================================================================
    # ASYNC OPERATIONS (for high-concurrency scenarios)
    # =========================================================================

    async def acreate_session(
        self, title: Optional[str] = None, user_id: Optional[str] = None
    ) -> str:
        """Async create session"""
        if not self._async_available:
            return self.create_session(title, user_id)

        session_id = str(uuid.uuid4())

        async with self.AsyncSessionFactory() as session:
            chat_session = ChatSession(
                id=session_id,
                title=title,
                user_id=user_id,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                is_active=True,
            )
            session.add(chat_session)
            await session.commit()

        # Cache update (sync cache is fine)
        self._session_cache.set(
            session_id,
            {
                "id": session_id,
                "title": title,
                "user_id": user_id,
                "is_active": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            },
        )
        self._message_count_cache.set(session_id, 0)

        return session_id

    def get_user_interaction_history(
        self, user_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent interaction history for a user across all sessions.
        Returns a list of message dictionaries.
        """
        if not user_id:
            return []

        with self._get_db_session() as db:
            # Join ChatMessage with ChatSession to filter by user_id
            # We want user messages and assistant responses
            messages = (
                db.query(ChatMessage)
                .join(ChatSession)
                .filter(ChatSession.user_id == user_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(limit)
                .all()
            )

            # Reverse to get chronological order
            messages.reverse()

            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat(),
                    "session_id": msg.session_id,
                }
                for msg in messages
            ]

    def get_recent_queries(
        self, limit: int = 50, user_id: Optional[str] = None, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get recent user queries for clustering analysis.

        Args:
            limit: Maximum number of queries to return
            user_id: Optional filter by user_id
            hours: Only fetch queries from last N hours

        Returns:
            List of query dictionaries with metadata
        """
        with self._get_db_session() as db:
            from datetime import datetime, timedelta

            cutoff_time = datetime.utcnow() - timedelta(hours=hours)

            query = (
                db.query(ChatMessage)
                .join(ChatSession)
                .filter(
                    ChatMessage.role == "user", ChatMessage.created_at >= cutoff_time
                )
            )

            if user_id:
                query = query.filter(ChatSession.user_id == user_id)

            messages = query.order_by(ChatMessage.created_at.desc()).limit(limit).all()

            return [
                {
                    "id": msg.id,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat(),
                    "session_id": msg.session_id,
                    "user_id": msg.session.user_id if msg.session else None,
                }
                for msg in messages
            ]

    def get_query_intents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent queries with their detected intents from sources metadata.

        Args:
            limit: Maximum number of queries

        Returns:
            List of queries with intent information
        """
        with self._get_db_session() as db:
            # Get user messages and their corresponding assistant responses
            user_messages = (
                db.query(ChatMessage)
                .filter(ChatMessage.role == "user")
                .order_by(ChatMessage.created_at.desc())
                .limit(limit)
                .all()
            )

            results = []
            for user_msg in user_messages:
                # Find the next assistant message in the same session
                assistant_msg = (
                    db.query(ChatMessage)
                    .filter(
                        ChatMessage.session_id == user_msg.session_id,
                        ChatMessage.role == "assistant",
                        ChatMessage.created_at > user_msg.created_at,
                    )
                    .order_by(ChatMessage.created_at)
                    .first()
                )

                intent = None
                if assistant_msg and assistant_msg.sources:
                    # Try to extract intent from sources metadata
                    for source in assistant_msg.sources:
                        if isinstance(source, dict) and "intent" in source:
                            intent = source["intent"]
                            break

                results.append(
                    {
                        "query": user_msg.content,
                        "intent": intent,
                        "created_at": user_msg.created_at.isoformat(),
                        "session_id": user_msg.session_id,
                    }
                )

            return results

    async def acreate_session_with_message(
        self,
        content: str,
        title: Optional[str] = None,
        user_id: Optional[str] = None,
        role: str = "user",
    ) -> Tuple[str, str]:
        """Async create session with first message"""
        if not self._async_available:
            return self.create_session_with_message(content, title, user_id, role)

        session_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        now = datetime.utcnow()

        if title is None:
            title = (
                content[:50].rsplit(" ", 1)[0] + "..." if len(content) > 50 else content
            )

        async with self.AsyncSessionFactory() as db:
            chat_session = ChatSession(
                id=session_id,
                title=title,
                user_id=user_id,
                created_at=now,
                updated_at=now,
                is_active=True,
            )
            db.add(chat_session)

            message = ChatMessage(
                id=message_id,
                session_id=session_id,
                role=role,
                content=content,
                created_at=now,
            )
            db.add(message)
            await db.commit()

        self._session_cache.set(
            session_id,
            {
                "id": session_id,
                "title": title,
                "user_id": user_id,
                "is_active": True,
                "created_at": now,
                "updated_at": now,
            },
        )
        self._message_count_cache.set(session_id, 1)

        return session_id, message_id

    async def aadd_message(
        self,
        session_id: str,
        content: str,
        role: str,
        token_count: Optional[int] = None,
        model_used: Optional[str] = None,
        sources: Optional[List[Dict]] = None,
        attachments: Optional[List[Dict]] = None,
    ) -> str:
        """Async add message"""
        if not self._async_available:
            return self.add_message(
                session_id, content, role, token_count, model_used, sources, attachments
            )

        message_id = str(uuid.uuid4())
        now = datetime.utcnow()

        async with self.AsyncSessionFactory() as db:
            message = ChatMessage(
                id=message_id,
                session_id=session_id,
                role=role,
                content=content,
                created_at=now,
                token_count=token_count,
                model_used=model_used,
                sources=sources,
                attachments=attachments,
            )
            db.add(message)

            # Update session timestamp using raw SQL for efficiency
            await db.execute(
                text("UPDATE chat_sessions SET updated_at = :now WHERE id = :sid"),
                {"now": now, "sid": session_id},
            )
            await db.commit()

        # Update caches
        cached_count = self._message_count_cache.get(session_id)
        if cached_count is not None:
            self._message_count_cache.set(session_id, cached_count + 1)

        return message_id

    async def aget_messages(
        self, session_id: str, limit: int = 100, offset: int = 0
    ) -> List[ChatMessageResponse]:
        """Async get messages"""
        if not self._async_available:
            return self.get_messages(session_id, limit, offset)

        async with self.AsyncSessionFactory() as db:
            from sqlalchemy import select

            stmt = (
                select(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at)
                .offset(offset)
                .limit(limit)
            )

            result = await db.execute(stmt)
            messages = result.scalars().all()

            return [
                ChatMessageResponse(
                    id=msg.id,
                    session_id=msg.session_id,
                    role=msg.role,
                    content=msg.content,
                    created_at=msg.created_at,
                    token_count=msg.token_count,
                    model_used=msg.model_used,
                    sources=msg.sources,
                    attachments=msg.attachments,
                    feedback_type=msg.feedback_type,
                )
                for msg in messages
            ]

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_db_session(self):
        """Get database session for external use (compatibility)"""
        return self._get_db_session()

    def clear_cache(self):
        """Clear all caches"""
        self._session_cache.clear()
        self._message_count_cache.clear()

    def health_check(self) -> Dict[str, Any]:
        """Check database connectivity and pool status"""
        try:
            with self._get_db_session() as db:
                db.execute(text("SELECT 1"))

            pool_health = pool_manager.get_health()
            return {
                "status": "healthy",
                **pool_health,
                "cache_session_count": len(self._session_cache._cache),
                "cache_message_count_count": len(self._message_count_cache._cache),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# =============================================================================
# BACKWARDS COMPATIBILITY ALIAS
# =============================================================================

# Create alias for drop-in replacement
ChatDatabaseManager = ChatDatabaseManagerV2


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_manager_instance: Optional[ChatDatabaseManagerV2] = None
_manager_lock = threading.Lock()


def get_chat_manager(
    database_url: Optional[str] = None, **kwargs
) -> ChatDatabaseManagerV2:
    """
    Get or create singleton chat manager instance.

    This is the recommended way to get a chat manager for most use cases.
    """
    global _manager_instance

    if _manager_instance is None:
        with _manager_lock:
            if _manager_instance is None:
                _manager_instance = ChatDatabaseManagerV2(database_url, **kwargs)

    return _manager_instance
