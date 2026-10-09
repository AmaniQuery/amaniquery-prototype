"""
SQLAlchemy Database Models for Authentication and Authorization
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    Float,
    ForeignKey,
    JSON,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

# Use the same Base as chat_models for consistency
from Module3_NiruDB.chat_models import Base

from .enums import (
    AuthMethod,
    RoleType,
    UserStatus,
    IntegrationStatus,
    PermissionResource,
    PermissionAction,
)


class User(Base):
    """User account model"""

    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)  # bcrypt/argon2 hash
    name = Column(String, nullable=True)
    phone_number = Column(
        String, nullable=True, index=True
    )  # Phone number for OTP verification
    phone_verified = Column(
        Boolean, default=False, nullable=False
    )  # Phone verification status
    status = Column(String, default=UserStatus.ACTIVE.value, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    email_verification_token = Column(String, nullable=True)
    password_reset_token = Column(String, nullable=True)
    password_reset_expires = Column(DateTime, nullable=True)
    last_login = Column(DateTime, nullable=True)
    last_login_ip = Column(String, nullable=True)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime, nullable=True)
    extra_data = Column(
        JSON, nullable=True
    )  # Additional user data (renamed from metadata to avoid SQLAlchemy conflict)
    profile_image_url = Column(
        String, nullable=True
    )  # Cloudinary URL for profile image
    mfa_secret = Column(String, nullable=True)
    mfa_enabled = Column(Boolean, default=False, nullable=False)
    mfa_backup_codes = Column(Text, nullable=True)
    mfa_method = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    roles = relationship(
        "UserRole",
        back_populates="user",
        foreign_keys="[UserRole.user_id]",
        cascade="all, delete-orphan",
    )
    sessions = relationship(
        "UserSession", back_populates="user", cascade="all, delete-orphan"
    )
    api_keys = relationship(
        "APIKey",
        back_populates="user",
        foreign_keys="[APIKey.user_id]",
        cascade="all, delete-orphan",
    )
    integrations = relationship(
        "Integration",
        back_populates="owner",
        foreign_keys="[Integration.owner_user_id]",
        cascade="all, delete-orphan",
    )
    oauth_clients = relationship(
        "OAuthClient",
        back_populates="owner",
        foreign_keys="[OAuthClient.owner_user_id]",
        cascade="all, delete-orphan",
    )
    usage_logs = relationship(
        "UsageLog",
        back_populates="user",
        foreign_keys="[UsageLog.user_id]",
        cascade="all, delete-orphan",
    )


class Role(Base):
    """Role definition model"""

    __tablename__ = "roles"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    role_type = Column(String, nullable=False)  # 'user' or 'integration'
    permissions = Column(
        JSON, nullable=True
    )  # List of permission strings like ["query:read", "research:write"]
    is_system = Column(
        Boolean, default=False, nullable=False
    )  # System roles cannot be deleted
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    user_roles = relationship(
        "UserRole", back_populates="role", cascade="all, delete-orphan"
    )
    integration_roles = relationship(
        "IntegrationRole", back_populates="role", cascade="all, delete-orphan"
    )


class UserRole(Base):
    """User role assignment"""

    __tablename__ = "user_roles"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role_id = Column(
        String, ForeignKey("roles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    assigned_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    assigned_by = Column(
        String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # Relationships
    user = relationship("User", back_populates="roles", foreign_keys=[user_id])
    role = relationship("Role", back_populates="user_roles")
    assigner = relationship("User", foreign_keys=[assigned_by])

    __table_args__ = (UniqueConstraint("user_id", "role_id", name="uq_user_role"),)


class Integration(Base):
    """Third-party integration model"""

    __tablename__ = "integrations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    type = Column(String, nullable=True)  # e.g., "webhook", "api_client", "mobile_app"
    owner_user_id = Column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status = Column(String, default=IntegrationStatus.ACTIVE.value, nullable=False)
    extra_data = Column(
        JSON, nullable=True
    )  # Additional integration data (renamed from metadata to avoid SQLAlchemy conflict)
    webhook_url = Column(String, nullable=True)
    ip_whitelist = Column(JSON, nullable=True)  # List of allowed IPs
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    owner = relationship(
        "User", back_populates="integrations", foreign_keys=[owner_user_id]
    )
    roles = relationship(
        "IntegrationRole", back_populates="integration", cascade="all, delete-orphan"
    )
    api_keys = relationship(
        "APIKey", back_populates="integration", cascade="all, delete-orphan"
    )
    oauth_clients = relationship(
        "OAuthClient", back_populates="integration", cascade="all, delete-orphan"
    )
    usage_logs = relationship(
        "UsageLog", back_populates="integration", cascade="all, delete-orphan"
    )


class IntegrationRole(Base):
    """Integration role assignment"""

    __tablename__ = "integration_roles"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    integration_id = Column(
        String,
        ForeignKey("integrations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role_id = Column(
        String, ForeignKey("roles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    assigned_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    assigned_by = Column(
        String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # Relationships
    integration = relationship("Integration", back_populates="roles")
    role = relationship("Role", back_populates="integration_roles")
    assigner = relationship("User", foreign_keys=[assigned_by])

    __table_args__ = (
        UniqueConstraint("integration_id", "role_id", name="uq_integration_role"),
    )


class APIKey(Base):
    """API key model"""

    __tablename__ = "api_keys"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    key_hash = Column(String, nullable=False, index=True)  # Hashed API key
    key_prefix = Column(
        String, nullable=False, index=True
    )  # First 8-12 chars for identification
    name = Column(String, nullable=True)  # User-friendly name
    user_id = Column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )
    integration_id = Column(
        String,
        ForeignKey("integrations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    scopes = Column(JSON, nullable=True)  # List of permission strings
    rate_limit_per_minute = Column(Integer, default=60, nullable=False)
    rate_limit_per_hour = Column(Integer, default=1000, nullable=False)
    rate_limit_per_day = Column(Integer, default=10000, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_used = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    extra_data = Column(
        JSON, nullable=True
    )  # Additional key data (renamed from metadata to avoid SQLAlchemy conflict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    user = relationship("User", back_populates="api_keys", foreign_keys=[user_id])
    integration = relationship(
        "Integration", back_populates="api_keys", foreign_keys=[integration_id]
    )

    __table_args__ = (
        Index("idx_api_key_prefix", "key_prefix"),
        Index("idx_api_key_user_integration", "user_id", "integration_id"),
    )


class OAuthClient(Base):
    """OAuth 2.0 client model"""

    __tablename__ = "oauth_clients"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = Column(String, unique=True, nullable=False, index=True)
    client_secret_hash = Column(String, nullable=False)  # Hashed client secret
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    owner_user_id = Column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    integration_id = Column(
        String,
        ForeignKey("integrations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    redirect_uris = Column(JSON, nullable=True)  # List of allowed redirect URIs
    grant_types = Column(
        JSON, nullable=True
    )  # ["authorization_code", "client_credentials", "refresh_token"]
    scopes = Column(JSON, nullable=True)  # Available scopes
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    owner = relationship(
        "User", back_populates="oauth_clients", foreign_keys=[owner_user_id]
    )
    integration = relationship(
        "Integration", back_populates="oauth_clients", foreign_keys=[integration_id]
    )
    tokens = relationship(
        "OAuthToken", back_populates="client", cascade="all, delete-orphan"
    )


class OAuthToken(Base):
    """OAuth 2.0 token model"""

    __tablename__ = "oauth_tokens"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    access_token = Column(
        String, unique=True, nullable=False, index=True
    )  # JWT or opaque token
    refresh_token = Column(String, unique=True, nullable=True, index=True)
    token_type = Column(String, default="Bearer", nullable=False)
    client_id = Column(
        String,
        ForeignKey("oauth_clients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )
    scopes = Column(JSON, nullable=True)  # Granted scopes
    expires_at = Column(DateTime, nullable=False, index=True)
    revoked = Column(Boolean, default=False, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    client = relationship("OAuthClient", back_populates="tokens")
    user = relationship("User", foreign_keys=[user_id])

    __table_args__ = (
        Index("idx_oauth_token_expires", "expires_at"),
        Index("idx_oauth_token_client_user", "client_id", "user_id"),
    )


class UserSession(Base):
    """User session model"""

    __tablename__ = "user_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    session_token = Column(
        String, unique=True, nullable=False, index=True
    )  # Hashed session token
    refresh_token = Column(String, unique=True, nullable=True, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    last_activity = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="sessions", foreign_keys=[user_id])

    __table_args__ = (
        Index("idx_session_expires", "expires_at"),
        Index("idx_session_user_active", "user_id", "is_active"),
    )


class Permission(Base):
    """Permission definition model (for reference/audit)"""

    __tablename__ = "permissions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    resource = Column(String, nullable=False, index=True)  # e.g., "query", "research"
    action = Column(String, nullable=False, index=True)  # e.g., "read", "write"
    description = Column(Text, nullable=True)
    conditions = Column(
        JSON, nullable=True
    )  # Additional conditions (time-based, IP-based, etc.)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("resource", "action", name="uq_permission_resource_action"),
        Index("idx_permission_resource_action", "resource", "action"),
    )


class UsageLog(Base):
    """API usage logging for cost tracking"""

    __tablename__ = "usage_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(
        String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    integration_id = Column(
        String,
        ForeignKey("integrations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    api_key_id = Column(
        String,
        ForeignKey("api_keys.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    endpoint = Column(String, nullable=False, index=True)
    method = Column(String, nullable=False)  # GET, POST, etc.
    status_code = Column(Integer, nullable=False)
    tokens_used = Column(Integer, default=0, nullable=False)  # LLM tokens consumed
    cost = Column(Float, default=0.0, nullable=False)  # Estimated cost in USD
    response_time_ms = Column(Float, nullable=True)
    request_size_bytes = Column(Integer, nullable=True)
    response_size_bytes = Column(Integer, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    extra_data = Column(
        JSON, nullable=True
    )  # Additional request/response data (renamed from metadata to avoid SQLAlchemy conflict)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="usage_logs", foreign_keys=[user_id])
    integration = relationship(
        "Integration", back_populates="usage_logs", foreign_keys=[integration_id]
    )
    api_key = relationship("APIKey", foreign_keys=[api_key_id])

    __table_args__ = (
        Index("idx_usage_timestamp", "timestamp"),
        Index("idx_usage_user_timestamp", "user_id", "timestamp"),
        Index("idx_usage_integration_timestamp", "integration_id", "timestamp"),
        Index("idx_usage_endpoint_timestamp", "endpoint", "timestamp"),
    )


class RateLimit(Base):
    """Rate limiting configuration and state"""

    __tablename__ = "rate_limits"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )
    integration_id = Column(
        String,
        ForeignKey("integrations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    api_key_id = Column(
        String, ForeignKey("api_keys.id", ondelete="CASCADE"), nullable=True, index=True
    )
    endpoint = Column(String, nullable=True, index=True)  # None for global limits
    limit_per_minute = Column(Integer, nullable=False)
    limit_per_hour = Column(Integer, nullable=False)
    limit_per_day = Column(Integer, nullable=False)
    current_minute_count = Column(Integer, default=0, nullable=False)
    current_hour_count = Column(Integer, default=0, nullable=False)
    current_day_count = Column(Integer, default=0, nullable=False)
    last_reset_minute = Column(DateTime, nullable=True)
    last_reset_hour = Column(DateTime, nullable=True)
    last_reset_day = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("idx_rate_limit_user_endpoint", "user_id", "endpoint"),
        Index("idx_rate_limit_integration_endpoint", "integration_id", "endpoint"),
        Index("idx_rate_limit_api_key_endpoint", "api_key_id", "endpoint"),
    )
