"""
SQLAlchemy Database Models for Blog Platform
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
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
import uuid

# Use the same Base as chat_models for consistency
from Module3_NiruDB.chat_models import Base


class BlogPost(Base):
    """Blog post model"""

    __tablename__ = "blog_posts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False, index=True)
    slug = Column(String, unique=True, nullable=False, index=True)
    markdown_content = Column(Text, nullable=True)
    html_content = Column(Text, nullable=True)
    excerpt = Column(Text, nullable=True)
    post_type = Column(
        String, nullable=False, index=True
    )  # 'news', 'announcement', 'update'
    author_id = Column(
        String, ForeignKey("users.id", ondelete="SET NULL"), nullable=False, index=True
    )
    status = Column(
        String, nullable=False, default="draft", index=True
    )  # 'draft', 'published'
    featured_image_url = Column(String, nullable=True)
    published_at = Column(DateTime, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    author = relationship("User", foreign_keys=[author_id])
    categories = relationship(
        "BlogPostCategory", back_populates="post", cascade="all, delete-orphan"
    )
    tags = relationship(
        "BlogPostTag", back_populates="post", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_blog_post_status_type", "status", "post_type"),
        Index("idx_blog_post_published", "status", "published_at"),
    )


class BlogCategory(Base):
    """Blog category model"""

    __tablename__ = "blog_categories"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, nullable=False, index=True)
    slug = Column(String, unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    posts = relationship(
        "BlogPostCategory", back_populates="category", cascade="all, delete-orphan"
    )


class BlogTag(Base):
    """Blog tag model"""

    __tablename__ = "blog_tags"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, nullable=False, index=True)
    slug = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    posts = relationship(
        "BlogPostTag", back_populates="tag", cascade="all, delete-orphan"
    )


class BlogPostCategory(Base):
    """Many-to-many relationship between posts and categories"""

    __tablename__ = "blog_post_categories"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    post_id = Column(
        String,
        ForeignKey("blog_posts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    category_id = Column(
        String,
        ForeignKey("blog_categories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    post = relationship("BlogPost", back_populates="categories")
    category = relationship("BlogCategory", back_populates="posts")

    __table_args__ = (
        UniqueConstraint("post_id", "category_id", name="uq_blog_post_category"),
        Index("idx_blog_post_category", "post_id", "category_id"),
    )


class BlogPostTag(Base):
    """Many-to-many relationship between posts and tags"""

    __tablename__ = "blog_post_tags"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    post_id = Column(
        String,
        ForeignKey("blog_posts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tag_id = Column(
        String,
        ForeignKey("blog_tags.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    post = relationship("BlogPost", back_populates="tags")
    tag = relationship("BlogTag", back_populates="posts")

    __table_args__ = (
        UniqueConstraint("post_id", "tag_id", name="uq_blog_post_tag"),
        Index("idx_blog_post_tag", "post_id", "tag_id"),
    )
