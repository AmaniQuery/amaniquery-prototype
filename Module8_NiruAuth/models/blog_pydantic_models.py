"""
Pydantic Models for Blog API Requests and Responses
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, validator
import re

# ==================== Blog Post Models ====================


class BlogPostCreate(BaseModel):
    """Blog post creation request"""

    title: str = Field(..., min_length=1, max_length=200)
    slug: Optional[str] = None
    markdown_content: Optional[str] = None
    html_content: Optional[str] = None
    excerpt: Optional[str] = Field(None, max_length=500)
    post_type: str = Field(
        ..., description="Post type: 'news', 'announcement', or 'update'"
    )
    status: str = Field(
        default="draft", description="Post status: 'draft' or 'published'"
    )
    featured_image_url: Optional[str] = None
    category_ids: Optional[List[str]] = []
    tag_ids: Optional[List[str]] = []
    published_at: Optional[datetime] = None

    @validator("featured_image_url")
    def normalize_featured_image_url(cls, v):
        # Treat empty strings as None
        if v == "":
            return None
        return v

    @validator("post_type")
    def validate_post_type(cls, v):
        allowed_types = ["news", "announcement", "update"]
        if v not in allowed_types:
            raise ValueError(f'post_type must be one of: {", ".join(allowed_types)}')
        return v

    @validator("status")
    def validate_status(cls, v):
        allowed_statuses = ["draft", "published"]
        if v not in allowed_statuses:
            raise ValueError(f'status must be one of: {", ".join(allowed_statuses)}')
        return v

    @validator("slug")
    def generate_slug(cls, v, values):
        # Treat empty strings as None
        if v == "":
            v = None
        if not v and "title" in values:
            # Generate slug from title
            slug = re.sub(r"[^\w\s-]", "", values["title"]).strip().lower()
            slug = re.sub(r"[-\s]+", "-", slug)
            return slug
        elif v:
            # Validate slug format
            if not re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", v):
                raise ValueError(
                    "slug must contain only lowercase letters, numbers, and hyphens"
                )
            return v
        return v


class BlogPostUpdate(BaseModel):
    """Blog post update request"""

    title: Optional[str] = Field(None, min_length=1, max_length=200)
    slug: Optional[str] = None
    markdown_content: Optional[str] = None
    html_content: Optional[str] = None
    excerpt: Optional[str] = Field(None, max_length=500)
    post_type: Optional[str] = None
    status: Optional[str] = None
    featured_image_url: Optional[str] = None
    category_ids: Optional[List[str]] = None
    tag_ids: Optional[List[str]] = None
    published_at: Optional[datetime] = None

    @validator("post_type")
    def validate_post_type(cls, v):
        if v is not None:
            allowed_types = ["news", "announcement", "update"]
            if v not in allowed_types:
                raise ValueError(
                    f'post_type must be one of: {", ".join(allowed_types)}'
                )
        return v

    @validator("status")
    def validate_status(cls, v):
        if v is not None:
            allowed_statuses = ["draft", "published"]
            if v not in allowed_statuses:
                raise ValueError(
                    f'status must be one of: {", ".join(allowed_statuses)}'
                )
        return v

    @validator("slug")
    def validate_slug(cls, v):
        # Treat empty strings as None
        if v == "":
            v = None
        if v is not None and not re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", v):
            raise ValueError(
                "slug must contain only lowercase letters, numbers, and hyphens"
            )
        return v

    @validator("featured_image_url")
    def normalize_featured_image_url(cls, v):
        # Treat empty strings as None
        if v == "":
            return None
        return v


class CategoryResponse(BaseModel):
    """Category response model"""

    id: str
    name: str
    slug: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TagResponse(BaseModel):
    """Tag response model"""

    id: str
    name: str
    slug: str
    created_at: datetime

    class Config:
        from_attributes = True


class AuthorResponse(BaseModel):
    """Author response model (simplified user info)"""

    id: str
    name: Optional[str]
    email: str
    profile_image_url: Optional[str] = None

    class Config:
        from_attributes = True


class BlogPostResponse(BaseModel):
    """Blog post response model"""

    id: str
    title: str
    slug: str
    markdown_content: Optional[str] = None
    html_content: Optional[str] = None
    excerpt: Optional[str] = None
    post_type: str
    author: AuthorResponse
    status: str
    featured_image_url: Optional[str] = None
    categories: List[CategoryResponse] = []
    tags: List[TagResponse] = []
    published_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class BlogPostListResponse(BaseModel):
    """Blog post list response with pagination"""

    posts: List[BlogPostResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


# ==================== Category Models ====================


class CategoryCreate(BaseModel):
    """Category creation request"""

    name: str = Field(..., min_length=1, max_length=100)
    slug: Optional[str] = None
    description: Optional[str] = None

    @validator("slug")
    def generate_slug(cls, v, values):
        if not v and "name" in values:
            slug = re.sub(r"[^\w\s-]", "", values["name"]).strip().lower()
            slug = re.sub(r"[-\s]+", "-", slug)
            return slug
        elif v:
            if not re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", v):
                raise ValueError(
                    "slug must contain only lowercase letters, numbers, and hyphens"
                )
            return v
        return v


class CategoryUpdate(BaseModel):
    """Category update request"""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    slug: Optional[str] = None
    description: Optional[str] = None

    @validator("slug")
    def validate_slug(cls, v):
        if v is not None and not re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", v):
            raise ValueError(
                "slug must contain only lowercase letters, numbers, and hyphens"
            )
        return v


# ==================== Tag Models ====================


class TagCreate(BaseModel):
    """Tag creation request"""

    name: str = Field(..., min_length=1, max_length=50)
    slug: Optional[str] = None

    @validator("slug")
    def generate_slug(cls, v, values):
        if not v and "name" in values:
            slug = re.sub(r"[^\w\s-]", "", values["name"]).strip().lower()
            slug = re.sub(r"[-\s]+", "-", slug)
            return slug
        elif v:
            if not re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", v):
                raise ValueError(
                    "slug must contain only lowercase letters, numbers, and hyphens"
                )
            return v
        return v


class TagUpdate(BaseModel):
    """Tag update request"""

    name: Optional[str] = Field(None, min_length=1, max_length=50)
    slug: Optional[str] = None

    @validator("slug")
    def validate_slug(cls, v):
        if v is not None and not re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", v):
            raise ValueError(
                "slug must contain only lowercase letters, numbers, and hyphens"
            )
        return v
