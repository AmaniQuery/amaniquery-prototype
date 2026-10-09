"""
User Profile Store - Persistent User Profile Management

Features:
- PostgreSQL-backed user profile storage
- User preferences and interests tracking
- Communication style adaptation
- Session-based profile caching
- Cross-session profile persistence
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
from loguru import logger

# Database imports
try:
    from sqlalchemy import Column, String, DateTime, Text, JSON, Float
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import Session
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logger.warning("SQLAlchemy not available for user profile persistence")


# Base for SQLAlchemy models
if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()

    class UserProfileModel(Base):
        """PostgreSQL model for persistent user profiles."""

        __tablename__ = "user_profiles"

        user_id = Column(String(255), primary_key=True)
        display_name = Column(String(255), nullable=True)
        email = Column(String(255), nullable=True)

        # Preferences
        expertise_level = Column(
            String(50), default="general"
        )  # beginner, intermediate, expert
        communication_style = Column(
            String(50), default="balanced"
        )  # formal, casual, balanced
        preferred_language = Column(String(10), default="en")  # en, sw, mixed

        # Interests and context
        interests = Column(JSON, default=list)  # List of interest topics
        task_groups = Column(JSON, default=list)  # Legal, news, general
        location = Column(String(100), default="Kenya")

        # Behavioral tracking
        total_queries = Column(Float, default=0)
        legal_queries = Column(Float, default=0)
        news_queries = Column(Float, default=0)
        general_queries = Column(Float, default=0)

        # Timestamps
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        last_active_at = Column(DateTime, default=datetime.utcnow)

        # Profile summary (for quick LLM context)
        profile_summary = Column(Text, nullable=True)

        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary."""
            return {
                "user_id": self.user_id,
                "display_name": self.display_name,
                "email": self.email,
                "expertise_level": self.expertise_level,
                "communication_style": self.communication_style,
                "preferred_language": self.preferred_language,
                "interests": self.interests or [],
                "task_groups": self.task_groups or [],
                "location": self.location,
                "total_queries": self.total_queries,
                "legal_queries": self.legal_queries,
                "news_queries": self.news_queries,
                "general_queries": self.general_queries,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "last_active_at": (
                    self.last_active_at.isoformat() if self.last_active_at else None
                ),
                "profile_summary": self.profile_summary,
            }


@dataclass
class UserProfile:
    """User profile data class."""

    user_id: str
    display_name: Optional[str] = None
    email: Optional[str] = None

    # Preferences
    expertise_level: str = "general"
    communication_style: str = "balanced"
    preferred_language: str = "en"

    # Interests
    interests: List[str] = field(default_factory=list)
    task_groups: List[str] = field(default_factory=list)
    location: str = "Kenya"

    # Stats
    total_queries: int = 0
    legal_queries: int = 0
    news_queries: int = 0
    general_queries: int = 0

    # Metadata
    profile_summary: Optional[str] = None
    created_at: Optional[str] = None
    last_active_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def get_context_summary(self) -> str:
        """Generate context summary for LLM."""
        parts = []

        if self.display_name:
            parts.append(f"User: {self.display_name}")

        if self.interests:
            parts.append(f"Interests: {', '.join(self.interests[:5])}")

        parts.append(f"Expertise: {self.expertise_level}")
        parts.append(f"Style: {self.communication_style}")

        if self.total_queries > 0:
            # Infer primary use case
            if (
                self.legal_queries > self.news_queries
                and self.legal_queries > self.general_queries
            ):
                parts.append("Primary use: Legal queries")
            elif self.news_queries > self.legal_queries:
                parts.append("Primary use: News queries")

        return " | ".join(parts)


class UserProfileStore:
    """
    Persistent user profile storage and management.

    Features:
    - PostgreSQL persistence
    - In-memory caching
    - Profile inference from behavior
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize user profile store.

        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self._session_cache: Dict[str, UserProfile] = {}

        # Initialize database if available
        self._engine = None
        self._session_maker = None

        if SQLALCHEMY_AVAILABLE and self.database_url:
            try:
                self._engine = create_engine(self.database_url)
                Base.metadata.create_all(self._engine)
                self._session_maker = sessionmaker(bind=self._engine)
                logger.info("UserProfileStore initialized with PostgreSQL")
            except Exception as e:
                logger.error(f"Failed to initialize database for profiles: {e}")
        else:
            logger.info("UserProfileStore running in memory-only mode")

    async def get_profile(self, user_id: str) -> UserProfile:
        """
        Get user profile (from cache or database).

        Args:
            user_id: User identifier

        Returns:
            UserProfile object
        """
        # Check cache first
        if user_id in self._session_cache:
            return self._session_cache[user_id]

        # Try database
        if self._session_maker:
            try:
                with self._session_maker() as session:
                    db_profile = (
                        session.query(UserProfileModel)
                        .filter(UserProfileModel.user_id == user_id)
                        .first()
                    )

                    if db_profile:
                        profile = UserProfile.from_dict(db_profile.to_dict())
                        self._session_cache[user_id] = profile
                        return profile
            except Exception as e:
                logger.error(f"Error fetching profile from database: {e}")

        # Create new profile
        profile = UserProfile(user_id=user_id, created_at=datetime.utcnow().isoformat())
        self._session_cache[user_id] = profile
        return profile

    async def update_profile(
        self, user_id: str, updates: Dict[str, Any]
    ) -> UserProfile:
        """
        Update user profile.

        Args:
            user_id: User identifier
            updates: Fields to update

        Returns:
            Updated UserProfile
        """
        profile = await self.get_profile(user_id)

        # Apply updates
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        profile.last_active_at = datetime.utcnow().isoformat()
        self._session_cache[user_id] = profile

        # Persist to database
        await self._persist_profile(profile)

        return profile

    async def track_query(self, user_id: str, query_type: str) -> UserProfile:
        """
        Track a query and update user statistics.

        Args:
            user_id: User identifier
            query_type: Type of query (LEGAL_RESEARCH, NEWS_SUMMARY, GENERAL_CHAT)

        Returns:
            Updated UserProfile
        """
        profile = await self.get_profile(user_id)

        profile.total_queries += 1

        if "LEGAL" in query_type.upper():
            profile.legal_queries += 1
            if "legal" not in profile.task_groups:
                profile.task_groups.append("legal")
        elif "NEWS" in query_type.upper():
            profile.news_queries += 1
            if "news" not in profile.task_groups:
                profile.task_groups.append("news")
        else:
            profile.general_queries += 1

        profile.last_active_at = datetime.utcnow().isoformat()
        self._session_cache[user_id] = profile

        # Persist periodically (every 5 queries)
        if profile.total_queries % 5 == 0:
            await self._persist_profile(profile)

        return profile

    async def add_interest(self, user_id: str, interest: str) -> UserProfile:
        """Add an interest to user profile."""
        profile = await self.get_profile(user_id)

        if interest not in profile.interests:
            profile.interests.append(interest)
            # Keep only top 20 interests
            profile.interests = profile.interests[-20:]

        self._session_cache[user_id] = profile
        return profile

    async def update_preferences(
        self,
        user_id: str,
        expertise_level: Optional[str] = None,
        communication_style: Optional[str] = None,
        preferred_language: Optional[str] = None,
    ) -> UserProfile:
        """Update user preferences."""
        updates = {}

        if expertise_level:
            updates["expertise_level"] = expertise_level
        if communication_style:
            updates["communication_style"] = communication_style
        if preferred_language:
            updates["preferred_language"] = preferred_language

        return await self.update_profile(user_id, updates)

    async def _persist_profile(self, profile: UserProfile):
        """Persist profile to database."""
        if not self._session_maker:
            return

        try:
            with self._session_maker() as session:
                db_profile = (
                    session.query(UserProfileModel)
                    .filter(UserProfileModel.user_id == profile.user_id)
                    .first()
                )

                if db_profile:
                    # Update existing
                    for key, value in profile.to_dict().items():
                        if hasattr(db_profile, key) and key != "user_id":
                            setattr(db_profile, key, value)
                else:
                    # Create new
                    db_profile = UserProfileModel(
                        user_id=profile.user_id,
                        display_name=profile.display_name,
                        email=profile.email,
                        expertise_level=profile.expertise_level,
                        communication_style=profile.communication_style,
                        preferred_language=profile.preferred_language,
                        interests=profile.interests,
                        task_groups=profile.task_groups,
                        location=profile.location,
                        total_queries=profile.total_queries,
                        legal_queries=profile.legal_queries,
                        news_queries=profile.news_queries,
                        general_queries=profile.general_queries,
                        profile_summary=profile.profile_summary,
                    )
                    session.add(db_profile)

                session.commit()
                logger.debug(f"Persisted profile for user: {profile.user_id}")
        except Exception as e:
            logger.error(f"Error persisting profile: {e}")

    def get_cached_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get profile from cache only (sync, fast)."""
        return self._session_cache.get(user_id)

    def clear_cache(self, user_id: Optional[str] = None):
        """Clear profile cache."""
        if user_id:
            self._session_cache.pop(user_id, None)
        else:
            self._session_cache.clear()


# Global instance
_profile_store: Optional[UserProfileStore] = None


def get_profile_store() -> UserProfileStore:
    """Get global profile store instance."""
    global _profile_store
    if _profile_store is None:
        _profile_store = UserProfileStore()
    return _profile_store
