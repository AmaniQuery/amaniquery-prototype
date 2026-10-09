"""
Database Models for Agent Query Monitoring
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    Boolean,
    Float,
    Integer,
    ForeignKey,
    JSON,
    Index,
)
from sqlalchemy.orm import declarative_base
from pydantic import BaseModel

from .chat_models import Base


class AgentQueryLog(Base):
    """Agent query execution log for monitoring and analysis"""

    __tablename__ = "agent_query_logs"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Query information
    query = Column(Text, nullable=False)
    persona = Column(String, nullable=False)  # wanjiku, wakili, mwanahabari
    intent = Column(String, nullable=False)  # news, law, hybrid, general

    # Performance metrics
    confidence = Column(Float, nullable=False)
    response_time_ms = Column(Integer, nullable=False)
    evidence_count = Column(Integer, default=0)
    reasoning_steps = Column(Integer, default=0)

    # Quality control
    human_review_required = Column(Boolean, default=False)
    agent_path = Column(JSON, nullable=True)  # Array of node names
    quality_issues = Column(JSON, nullable=True)  # Array of issue descriptions
    reasoning_path = Column(JSON, nullable=True)  # Full reasoning trace

    # User feedback
    user_feedback = Column(String, nullable=True)  # positive, negative, null

    # Review information
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_by = Column(
        String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    review_status = Column(String, nullable=True)  # pending, approved, rejected
    review_feedback = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_query_logs_timestamp", "timestamp"),
        Index("idx_query_logs_persona", "persona"),
        Index("idx_query_logs_intent", "intent"),
        Index("idx_query_logs_confidence", "confidence"),
        Index("idx_query_logs_review", "human_review_required", "review_status"),
    )


# Pydantic models for API responses
class AgentQueryLogResponse(BaseModel):
    id: str
    timestamp: datetime
    query: str
    persona: str
    intent: str
    confidence: float
    response_time_ms: int
    evidence_count: int
    reasoning_steps: int
    human_review_required: bool
    agent_path: List[str]
    quality_issues: List[str]
    reasoning_path: Optional[Dict[str, Any]] = None
    user_feedback: Optional[str] = None

    model_config = {"from_attributes": True}


class ReviewQueueItem(AgentQueryLogResponse):
    """Extended model for review queue items"""

    review_reason: str
    priority: str  # low, medium, high

    model_config = {"from_attributes": True}


def generate_query_log_id() -> str:
    """Generate a unique query log ID"""
    import uuid

    return str(uuid.uuid4())
