"""
Database Models for Chat Sessions, Messages, and Feedback
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    Float,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from pydantic import BaseModel

Base = declarative_base()


class ChatSession(Base):
    """Chat session model"""

    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True)
    user_id = Column(
        String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )  # Tied to user authentication
    title = Column(String, nullable=True)  # Auto-generated from first message
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    messages = relationship(
        "ChatMessage", back_populates="session", cascade="all, delete-orphan"
    )


class ChatMessage(Base):
    """Chat message model"""

    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Message metadata
    token_count = Column(Integer, nullable=True)
    model_used = Column(String, nullable=True)
    sources = Column(JSON, nullable=True)  # Store sources as JSON
    attachments = Column(JSON, nullable=True)  # Store attachments as JSON

    # Feedback
    feedback_type = Column(String, nullable=True)  # "like", "dislike", or None
    feedback_comment = Column(Text, nullable=True)
    feedback_at = Column(DateTime, nullable=True)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")


class UserFeedback(Base):
    """General user feedback model"""

    __tablename__ = "user_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=True)
    message_id = Column(String, ForeignKey("chat_messages.id"), nullable=True)
    feedback_type = Column(String, nullable=False)  # "like", "dislike", "share", "copy"
    content = Column(Text, nullable=True)  # Additional feedback content
    feedback_metadata = Column(JSON, nullable=True)  # Additional metadata
    created_at = Column(DateTime, default=datetime.utcnow)


class TaskCluster(Base):
    """Task cluster model for grouping similar user queries"""

    __tablename__ = "task_clusters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_name = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=False)
    representative_queries = Column(JSON, nullable=False)  # List of 3-5 example queries
    metadata_tags = Column(JSON, nullable=True)  # Additional metadata
    query_count = Column(Integer, default=0)  # Number of queries matching this cluster
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TrainingDataset(Base):
    """Training dataset for fine-tuning from production interactions"""

    __tablename__ = "training_dataset"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String, ForeignKey("chat_messages.id"), nullable=True)

    # Content
    user_query = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)  # Citations and sources used

    # Quality metrics
    quality_score = Column(Float, nullable=False)  # 1-5 scale
    score_criteria = Column(JSON, nullable=True)  # Detailed scoring breakdown
    keep_for_finetune = Column(Boolean, default=False)
    scoring_reason = Column(Text, nullable=True)

    # Metadata
    intent = Column(String, nullable=True)  # From supervisor
    expertise_level = Column(String, nullable=True)  # From user profile
    cluster_tags = Column(JSON, nullable=True)  # Related task clusters

    # Export tracking
    exported_at = Column(DateTime, nullable=True)
    export_format = Column(String, nullable=True)  # 'alpaca', 'sharegpt', etc.

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Pydantic models for API
class ChatSessionCreate(BaseModel):
    title: Optional[str] = None
    user_id: Optional[str] = None


class ChatSessionResponse(BaseModel):
    id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    is_active: bool
    message_count: int = 0


class ChatAttachment(BaseModel):
    """Attachment metadata model"""

    id: str
    filename: str
    file_type: str  # "pdf", "image", "text"
    file_size: int  # in bytes
    uploaded_at: datetime
    processed: bool = False
    cloudinary_url: Optional[str] = None  # Cloudinary URL for persistent storage


class ChatMessageCreate(BaseModel):
    content: str
    role: str = "user"
    stream: bool = False
    attachment_ids: Optional[List[str]] = None  # List of attachment IDs


class ChatMessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    created_at: datetime
    token_count: Optional[int]
    model_used: Optional[str]
    sources: Optional[List[dict]]
    attachments: Optional[List[dict]] = None
    feedback_type: Optional[str]

    model_config = {"protected_namespaces": ()}


class FeedbackCreate(BaseModel):
    message_id: str
    feedback_type: str  # "like", "dislike", "share", "copy"
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: int
    message_id: str
    feedback_type: str
    comment: Optional[str]
    created_at: datetime


class TaskClusterCreate(BaseModel):
    cluster_name: str
    description: str
    representative_queries: List[str]  # 3-5 example queries
    metadata_tags: Optional[List[str]] = None


class TaskClusterResponse(BaseModel):
    id: int
    cluster_name: str
    description: str
    representative_queries: List[str]
    metadata_tags: Optional[List[str]]
    query_count: int
    is_active: bool
    created_at: datetime
    last_updated: datetime


class ClusterSuggestion(BaseModel):
    """Model for suggested new clusters from analysis"""

    group_name: str
    description: str
    representative_queries: List[str]
    suggested_metadata_tags: List[str]
    confidence: float  # 0.0 to 1.0


class TrainingDataCreate(BaseModel):
    user_query: str
    assistant_response: str
    sources: Optional[List[dict]] = None
    quality_score: float
    score_criteria: Optional[dict] = None
    keep_for_finetune: bool
    scoring_reason: Optional[str] = None
    intent: Optional[str] = None
    expertise_level: Optional[str] = None
    cluster_tags: Optional[List[str]] = None


class TrainingDataResponse(BaseModel):
    id: int
    user_query: str
    assistant_response: str
    sources: Optional[List[Dict[str, Any]]]
    quality_score: float
    score_criteria: Optional[Dict[str, Any]]
    keep_for_finetune: bool
    scoring_reason: Optional[str]
    intent: Optional[str]
    expertise_level: Optional[str]
    cluster_tags: Optional[List[str]]
    exported_at: Optional[datetime]
    export_format: Optional[str]
    created_at: datetime


class QualityScoreResult(BaseModel):
    """Result of quality scoring"""

    score: float
    keep_for_finetune: bool
    criteria: dict
    reason: str


# Database connection and session management
def create_database_engine(database_url: str):
    """Create SQLAlchemy engine with connection pooling"""
    from sqlalchemy.pool import QueuePool

    return create_engine(
        database_url,
        echo=False,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # Check connection before using
        pool_recycle=300,  # Recycle connections every 5 minutes
        pool_timeout=30,
        connect_args={"connect_timeout": 10},
    )


def create_tables(engine):
    """Create all tables"""
    Base.metadata.create_all(engine)


from contextlib import contextmanager


@contextmanager
def get_db_session(engine):
    """
    Get database session as context manager.

    Usage:
        with get_db_session(engine) as db:
            db.query(...)
    """
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Utility functions
def generate_session_id() -> str:
    """Generate a unique session ID"""
    import uuid

    return str(uuid.uuid4())


def generate_message_id() -> str:
    """Generate a unique message ID"""
    import uuid

    return str(uuid.uuid4())
