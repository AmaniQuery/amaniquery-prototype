"""
Database Models for Notification Subscriptions
"""

from datetime import datetime, time
from typing import Optional, List
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    JSON,
    Time,
    Index,
)
from sqlalchemy.orm import sessionmaker, declarative_base
from pydantic import BaseModel

Base = declarative_base()


class NotificationSubscription(Base):
    """Notification subscription model"""

    __tablename__ = "notification_subscriptions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    phone_number = Column(String(20), nullable=False, index=True)
    notification_type = Column(
        String(20), nullable=False, default="whatsapp"
    )  # whatsapp, sms, both
    schedule_type = Column(
        String(20), nullable=False, default="immediate"
    )  # immediate, daily_digest
    digest_time = Column(Time, nullable=True)  # Time for daily digest
    categories = Column(JSON, nullable=True)  # List of category filters
    sources = Column(JSON, nullable=True)  # List of source filters
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Index for faster lookups
    __table_args__ = (Index("idx_phone_active", "phone_number", "is_active"),)


# Pydantic models for API
class SubscriptionCreate(BaseModel):
    phone_number: str
    notification_type: str = "whatsapp"  # whatsapp, sms, both
    schedule_type: str = "immediate"  # immediate, daily_digest
    digest_time: Optional[str] = None  # HH:MM format
    categories: Optional[List[str]] = None
    sources: Optional[List[str]] = None


class SubscriptionUpdate(BaseModel):
    notification_type: Optional[str] = None
    schedule_type: Optional[str] = None
    digest_time: Optional[str] = None
    categories: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    is_active: Optional[bool] = None


class SubscriptionResponse(BaseModel):
    id: int
    phone_number: str
    notification_type: str
    schedule_type: str
    digest_time: Optional[str] = None
    categories: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"protected_namespaces": ()}


# Database connection and session management
def create_database_engine(database_url: str):
    """Create SQLAlchemy engine with connection pooling"""
    return create_engine(
        database_url,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={"connect_timeout": 10},
    )


def create_tables(engine):
    """Create all tables"""
    Base.metadata.create_all(engine)


def get_db_session(engine):
    """Get database session"""
    Session = sessionmaker(bind=engine)
    return Session()
