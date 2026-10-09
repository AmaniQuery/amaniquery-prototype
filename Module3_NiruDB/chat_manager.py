"""
Chat Database Manager - High Performance Edition
Handles chat sessions, messages, and feedback storage

Optimized for 5000+ concurrent operations with:
- Connection pooling (20 connections, 30 overflow)
- Session and message count caching with TTL
- Batch operations for bulk inserts
- Async support for high-concurrency scenarios
- Optimized database indexes
"""

import os
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from loguru import logger

# Import the optimized v2 manager
from Module3_NiruDB.chat_manager_v2 import (
    ChatDatabaseManagerV2,
    get_chat_manager,
    TTLCache,
    ConnectionPoolManager,
)

from Module3_NiruDB.chat_models import (
    ChatSession,
    ChatMessage,
    UserFeedback,
    ChatSessionResponse,
    ChatMessageResponse,
    FeedbackResponse,
    create_database_engine,
    get_db_session,
    generate_session_id,
    generate_message_id,
)


class ChatDatabaseManager(ChatDatabaseManagerV2):
    """
    High-performance chat database manager.

    This class extends ChatDatabaseManagerV2 and provides full backwards
    compatibility with the original API while offering:

    - Connection pooling (20 connections by default)
    - Session and message count caching
    - Batch operations for bulk inserts
    - Async support for high-concurrency scenarios
    - Optimized queries with proper indexing

    Performance Tips:
    -----------------
    For new conversations, use create_session_with_message() instead of
    separate create_session() + add_message() calls - this is 2x faster.

    For bulk operations, use add_messages_batch() instead of multiple
    add_message() calls.

    For high-concurrency scenarios, use the async methods:
    - acreate_session()
    - acreate_session_with_message()
    - aadd_message()
    - aget_messages()

    Example:
    --------
    >>> manager = ChatDatabaseManager()
    >>>
    >>> # Fast path for new conversation
    >>> session_id, message_id = manager.create_session_with_message(
    ...     content="Hello, I need legal advice",
    ...     user_id="user123"
    ... )
    >>>
    >>> # Add AI response
    >>> manager.add_message(session_id, "How can I help?", "assistant")
    >>>
    >>> # Async version for high concurrency
    >>> session_id, message_id = await manager.acreate_session_with_message(
    ...     content="Hello", user_id="user123"
    ... )
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize chat database manager.

        Args:
            database_url: PostgreSQL connection URL. If not provided,
                         uses DATABASE_URL environment variable.
        """
        # Call parent with optimized defaults
        super().__init__(
            database_url=database_url,
            pool_size=20,
            max_overflow=30,
            cache_size=1000,
            cache_ttl=300,
        )

        logger.info("ChatDatabaseManager initialized with v2 optimizations")


# Export the factory function for singleton access
__all__ = [
    "ChatDatabaseManager",
    "ChatDatabaseManagerV2",
    "get_chat_manager",
    "TTLCache",
    "ConnectionPoolManager",
]
