"""
Module 3: NiruDB - Database & Vector Storage
Unified connection pooling, vector stores, and data persistence for 1M+ concurrent users.
"""

__version__ = "2.0.0"

from .vector_store import VectorStore
from .metadata_manager import MetadataManager
from .database_storage import DatabaseStorage, RawDocument, ProcessedChunk
from .chat_manager import ChatDatabaseManager
from .chat_manager_v2 import ChatDatabaseManagerV2, get_chat_manager, TTLCache
from .chat_models import (
    ChatSession,
    ChatMessage,
    UserFeedback,
    TaskCluster,
    TrainingDataset,
    create_database_engine,
    get_db_session,
)
from .agent_models import AgentQueryLog
from .agent_monitoring import log_agent_query, get_agent_metrics, get_review_queue
from .connection_pool import (
    pool_manager,
    ConnectionPoolManager,
    EngineConfig,
    EngineGroup,
    PoolMetrics,
)

__all__ = [
    "VectorStore",
    "MetadataManager",
    "DatabaseStorage",
    "RawDocument",
    "ProcessedChunk",
    "ChatDatabaseManager",
    "ChatDatabaseManagerV2",
    "get_chat_manager",
    "TTLCache",
    "ChatSession",
    "ChatMessage",
    "UserFeedback",
    "TaskCluster",
    "TrainingDataset",
    "create_database_engine",
    "get_db_session",
    "AgentQueryLog",
    "log_agent_query",
    "get_agent_metrics",
    "get_review_queue",
    "pool_manager",
    "ConnectionPoolManager",
    "EngineConfig",
    "EngineGroup",
    "PoolMetrics",
]
