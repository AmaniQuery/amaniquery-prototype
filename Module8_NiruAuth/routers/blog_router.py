"""
Blog Router
API endpoints for blog posts, categories, and tags
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_
from typing import List, Optional
from datetime import datetime
import re

from ..models.blog_models import (
    BlogPost,
    BlogCategory,
    BlogTag,
    BlogPostCategory,
    BlogPostTag,
)
from ..models.blog_pydantic_models import (
    BlogPostCreate,
    BlogPostUpdate,
    BlogPostResponse,
    BlogPostListResponse,
    CategoryCreate,
    CategoryUpdate,
    CategoryResponse,
    TagCreate,
    TagUpdate,
    TagResponse,
    AuthorResponse,
)
from ..models.auth_models import User
from ..dependencies import get_db, require_admin, get_current_user
from ..models.pydantic_models import UserResponse

router = APIRouter(prefix="/api/v1/blog", tags=["Blog"])


# ==================== Helper Functions ====================


def generate_slug(text: str) -> str:
    """Generate URL-friendly slug from text"""
    slug = re.sub(r"[^\w\s-]", "", text).strip().lower()
    slug = re.sub(r"[-\s]+", "-", slug)
    return slug


# ==================== Public Blog Post Endpoints ====================


@router.get("/posts", response_model=BlogPostListResponse)
async def list_posts(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    post_type: Optional[str] = Query(
        None, description="Filter by post type: news, announcement, update"
    ),
    category_slug: Optional[str] = Query(None, description="Filter by category slug"),
    tag_slug: Optional[str] = Query(None, description="Filter by tag slug"),
    search: Optional[str] = Query(None, description="Search in title and content"),
    db: Session = Depends(get_db),
):
    """List published blog posts (public endpoint)"""
    offset = (page - 1) * page_size

    # Base query - only published posts
    query = db.query(BlogPost).filter(BlogPost.status == "published")

    # Filter by post type
    if post_type:
        if post_type not in ["news", "announcement", "update"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid post_type. Must be: news, announcement, or update",
            )
        query = query.filter(BlogPost.post_type == post_type)

    # Filter by category
    if category_slug:
        category = (
            db.query(BlogCategory).filter(BlogCategory.slug == category_slug).first()
        )
        if category:
            query = query.join(BlogPostCategory).filter(
                BlogPostCategory.category_id == category.id
            )
        else:
            # Return empty result if category doesn't exist
            query = query.filter(BlogPost.id == None)

    # Filter by tag
    if tag_slug:
        tag = db.query(BlogTag).filter(BlogTag.slug == tag_slug).first()
        if tag:
            query = query.join(BlogPostTag).filter(BlogPostTag.tag_id == tag.id)
        else:
            # Return empty result if tag doesn't exist
            query = query.filter(BlogPost.id == None)

    # Search
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                BlogPost.title.ilike(search_term),
                BlogPost.excerpt.ilike(search_term),
                BlogPost.markdown_content.ilike(search_term),
            )
        )

    # Get total count before pagination
    total = query.count()

    # Apply pagination and ordering
    posts = (
        query.order_by(BlogPost.published_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    # Build response
    post_responses = []
    for post in posts:
        # Get author
        author = db.query(User).filter(User.id == post.author_id).first()
        author_response = AuthorResponse(
            id=author.id if author else post.author_id,
            name=author.name if author else None,
            email=author.email if author else "",
            profile_image_url=author.profile_image_url if author else None,
        )

        # Get categories
        categories = []
        for post_cat in post.categories:
            category = (
                db.query(BlogCategory)
                .filter(BlogCategory.id == post_cat.category_id)
                .first()
            )
            if category:
                categories.append(
                    CategoryResponse(
                        id=category.id,
                        name=category.name,
                        slug=category.slug,
                        description=category.description,
                        created_at=category.created_at,
                        updated_at=category.updated_at,
                    )
                )

        # Get tags
        tags = []
        for post_tag in post.tags:
            tag = db.query(BlogTag).filter(BlogTag.id == post_tag.tag_id).first()
            if tag:
                tags.append(
                    TagResponse(
                        id=tag.id,
                        name=tag.name,
                        slug=tag.slug,
                        created_at=tag.created_at,
                    )
                )

        post_responses.append(
            BlogPostResponse(
                id=post.id,
                title=post.title,
                slug=post.slug,
                markdown_content=post.markdown_content,
                html_content=post.html_content,
                excerpt=post.excerpt,
                post_type=post.post_type,
                author=author_response,
                status=post.status,
                featured_image_url=post.featured_image_url,
                categories=categories,
                tags=tags,
                published_at=post.published_at,
                created_at=post.created_at,
                updated_at=post.updated_at,
            )
        )

    total_pages = (total + page_size - 1) // page_size

    return BlogPostListResponse(
        posts=post_responses,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/posts/{slug}", response_model=BlogPostResponse)
async def get_post_by_slug(slug: str, db: Session = Depends(get_db)):
    """Get a published blog post by slug (public endpoint)"""
    post = (
        db.query(BlogPost)
        .filter(BlogPost.slug == slug, BlogPost.status == "published")
        .first()
    )

    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Post not found"
        )

    # Get author
    author = db.query(User).filter(User.id == post.author_id).first()
    author_response = AuthorResponse(
        id=author.id if author else post.author_id,
        name=author.name if author else None,
        email=author.email if author else "",
        profile_image_url=author.profile_image_url if author else None,
    )

    # Get categories
    categories = []
    for post_cat in post.categories:
        category = (
            db.query(BlogCategory)
            .filter(BlogCategory.id == post_cat.category_id)
            .first()
        )
        if category:
            categories.append(
                CategoryResponse(
                    id=category.id,
                    name=category.name,
                    slug=category.slug,
                    description=category.description,
                    created_at=category.created_at,
                    updated_at=category.updated_at,
                )
            )

    # Get tags
    tags = []
    for post_tag in post.tags:
        tag = db.query(BlogTag).filter(BlogTag.id == post_tag.tag_id).first()
        if tag:
            tags.append(
                TagResponse(
                    id=tag.id, name=tag.name, slug=tag.slug, created_at=tag.created_at
                )
            )

    return BlogPostResponse(
        id=post.id,
        title=post.title,
        slug=post.slug,
        markdown_content=post.markdown_content,
        html_content=post.html_content,
        excerpt=post.excerpt,
        post_type=post.post_type,
        author=author_response,
        status=post.status,
        featured_image_url=post.featured_image_url,
        categories=categories,
        tags=tags,
        published_at=post.published_at,
        created_at=post.created_at,
        updated_at=post.updated_at,
    )


# ==================== Admin Blog Post Endpoints ====================


@router.post(
    "/posts", response_model=BlogPostResponse, status_code=status.HTTP_201_CREATED
)
async def create_post(
    post_data: BlogPostCreate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Create a new blog post (admin only)"""
    # Generate slug if not provided
    slug = post_data.slug or generate_slug(post_data.title)

    # Check if slug already exists
    existing = db.query(BlogPost).filter(BlogPost.slug == slug).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Post with slug '{slug}' already exists",
        )

    # Set published_at if status is published
    published_at = post_data.published_at
    if post_data.status == "published" and not published_at:
        published_at = datetime.utcnow()

    # Create post
    post = BlogPost(
        title=post_data.title,
        slug=slug,
        markdown_content=post_data.markdown_content,
        html_content=post_data.html_content,
        excerpt=post_data.excerpt,
        post_type=post_data.post_type,
        author_id=admin.id,
        status=post_data.status,
        featured_image_url=post_data.featured_image_url,
        published_at=published_at,
    )

    db.add(post)
    db.flush()  # Get the post ID

    # Add categories
    if post_data.category_ids:
        for category_id in post_data.category_ids:
            category = (
                db.query(BlogCategory).filter(BlogCategory.id == category_id).first()
            )
            if category:
                post_category = BlogPostCategory(
                    post_id=post.id, category_id=category_id
                )
                db.add(post_category)

    # Add tags
    if post_data.tag_ids:
        for tag_id in post_data.tag_ids:
            tag = db.query(BlogTag).filter(BlogTag.id == tag_id).first()
            if tag:
                post_tag = BlogPostTag(post_id=post.id, tag_id=tag_id)
                db.add(post_tag)

    db.commit()
    db.refresh(post)

    # Build response
    author_response = AuthorResponse(
        id=admin.id,
        name=admin.name,
        email=admin.email,
        profile_image_url=admin.profile_image_url,
    )

    categories = []
    tags = []
    for post_cat in post.categories:
        category = (
            db.query(BlogCategory)
            .filter(BlogCategory.id == post_cat.category_id)
            .first()
        )
        if category:
            categories.append(
                CategoryResponse(
                    id=category.id,
                    name=category.name,
                    slug=category.slug,
                    description=category.description,
                    created_at=category.created_at,
                    updated_at=category.updated_at,
                )
            )

    for post_tag in post.tags:
        tag = db.query(BlogTag).filter(BlogTag.id == post_tag.tag_id).first()
        if tag:
            tags.append(
                TagResponse(
                    id=tag.id, name=tag.name, slug=tag.slug, created_at=tag.created_at
                )
            )

    return BlogPostResponse(
        id=post.id,
        title=post.title,
        slug=post.slug,
        markdown_content=post.markdown_content,
        html_content=post.html_content,
        excerpt=post.excerpt,
        post_type=post.post_type,
        author=author_response,
        status=post.status,
        featured_image_url=post.featured_image_url,
        categories=categories,
        tags=tags,
        published_at=post.published_at,
        created_at=post.created_at,
        updated_at=post.updated_at,
    )


@router.put("/posts/{post_id}", response_model=BlogPostResponse)
async def update_post(
    post_id: str,
    post_data: BlogPostUpdate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Update a blog post (admin only)"""
    post = db.query(BlogPost).filter(BlogPost.id == post_id).first()
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Post not found"
        )

    # Update fields
    if post_data.title is not None:
        post.title = post_data.title

    if post_data.slug is not None:
        # Check if new slug already exists (excluding current post)
        existing = (
            db.query(BlogPost)
            .filter(BlogPost.slug == post_data.slug, BlogPost.id != post_id)
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Post with slug '{post_data.slug}' already exists",
            )
        post.slug = post_data.slug

    if post_data.markdown_content is not None:
        post.markdown_content = post_data.markdown_content

    if post_data.html_content is not None:
        post.html_content = post_data.html_content

    if post_data.excerpt is not None:
        post.excerpt = post_data.excerpt

    if post_data.post_type is not None:
        post.post_type = post_data.post_type

    if post_data.status is not None:
        post.status = post_data.status
        # Set published_at if publishing for the first time
        if post_data.status == "published" and not post.published_at:
            post.published_at = datetime.utcnow()

    if post_data.featured_image_url is not None:
        post.featured_image_url = post_data.featured_image_url

    if post_data.published_at is not None:
        post.published_at = post_data.published_at

    # Update categories
    if post_data.category_ids is not None:
        # Remove existing categories
        db.query(BlogPostCategory).filter(BlogPostCategory.post_id == post_id).delete()
        # Add new categories
        for category_id in post_data.category_ids:
            category = (
                db.query(BlogCategory).filter(BlogCategory.id == category_id).first()
            )
            if category:
                post_category = BlogPostCategory(
                    post_id=post.id, category_id=category_id
                )
                db.add(post_category)

    # Update tags
    if post_data.tag_ids is not None:
        # Remove existing tags
        db.query(BlogPostTag).filter(BlogPostTag.post_id == post_id).delete()
        # Add new tags
        for tag_id in post_data.tag_ids:
            tag = db.query(BlogTag).filter(BlogTag.id == tag_id).first()
            if tag:
                post_tag = BlogPostTag(post_id=post.id, tag_id=tag_id)
                db.add(post_tag)

    db.commit()
    db.refresh(post)

    # Build response
    author = db.query(User).filter(User.id == post.author_id).first()
    author_response = AuthorResponse(
        id=author.id if author else post.author_id,
        name=author.name if author else None,
        email=author.email if author else "",
        profile_image_url=author.profile_image_url if author else None,
    )

    categories = []
    tags = []
    for post_cat in post.categories:
        category = (
            db.query(BlogCategory)
            .filter(BlogCategory.id == post_cat.category_id)
            .first()
        )
        if category:
            categories.append(
                CategoryResponse(
                    id=category.id,
                    name=category.name,
                    slug=category.slug,
                    description=category.description,
                    created_at=category.created_at,
                    updated_at=category.updated_at,
                )
            )

    for post_tag in post.tags:
        tag = db.query(BlogTag).filter(BlogTag.id == post_tag.tag_id).first()
        if tag:
            tags.append(
                TagResponse(
                    id=tag.id, name=tag.name, slug=tag.slug, created_at=tag.created_at
                )
            )

    return BlogPostResponse(
        id=post.id,
        title=post.title,
        slug=post.slug,
        markdown_content=post.markdown_content,
        html_content=post.html_content,
        excerpt=post.excerpt,
        post_type=post.post_type,
        author=author_response,
        status=post.status,
        featured_image_url=post.featured_image_url,
        categories=categories,
        tags=tags,
        published_at=post.published_at,
        created_at=post.created_at,
        updated_at=post.updated_at,
    )


@router.delete("/posts/{post_id}")
async def delete_post(
    post_id: str, admin: User = Depends(require_admin), db: Session = Depends(get_db)
):
    """Delete a blog post (admin only)"""
    post = db.query(BlogPost).filter(BlogPost.id == post_id).first()
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Post not found"
        )

    db.delete(post)
    db.commit()

    return {"message": "Post deleted successfully"}


@router.post("/posts/{post_id}/featured-image")
async def upload_featured_image(
    post_id: str,
    file: UploadFile = File(...),
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Upload featured image for a blog post (admin only)"""
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}",
        )

    # Validate file size (5MB limit)
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 5MB limit",
        )

    # Check if post exists
    post = db.query(BlogPost).filter(BlogPost.id == post_id).first()
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Post not found"
        )

    try:
        # Upload to Cloudinary
        from Module4_NiruAPI.services.cloudinary_service import CloudinaryService

        cloudinary_service = CloudinaryService()

        # Get file extension
        file_ext = (
            "." + file.filename.split(".")[-1] if "." in file.filename else ".jpg"
        )
        filename = f"blog_post_{post_id}{file_ext}"

        # Upload to Cloudinary
        result = cloudinary_service.upload_bytes(
            file_content=file_content,
            filename=filename,
            session_id=post_id,
            resource_type="image",
            folder="blog_posts",
        )

        cloudinary_url = result.get("secure_url") or result.get("url")

        # Update post
        post.featured_image_url = cloudinary_url
        db.commit()
        db.refresh(post)

        return {"featured_image_url": cloudinary_url}

    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloudinary service not available",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload image: {str(e)}",
        )


# ==================== Category Endpoints ====================


@router.get("/categories", response_model=List[CategoryResponse])
async def list_categories(db: Session = Depends(get_db)):
    """List all categories (public endpoint)"""
    categories = db.query(BlogCategory).order_by(BlogCategory.name).all()
    return [
        CategoryResponse(
            id=cat.id,
            name=cat.name,
            slug=cat.slug,
            description=cat.description,
            created_at=cat.created_at,
            updated_at=cat.updated_at,
        )
        for cat in categories
    ]


@router.post(
    "/categories", response_model=CategoryResponse, status_code=status.HTTP_201_CREATED
)
async def create_category(
    category_data: CategoryCreate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Create a new category (admin only)"""
    slug = category_data.slug or generate_slug(category_data.name)

    # Check if slug already exists
    existing = db.query(BlogCategory).filter(BlogCategory.slug == slug).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Category with slug '{slug}' already exists",
        )

    category = BlogCategory(
        name=category_data.name, slug=slug, description=category_data.description
    )

    db.add(category)
    db.commit()
    db.refresh(category)

    return CategoryResponse(
        id=category.id,
        name=category.name,
        slug=category.slug,
        description=category.description,
        created_at=category.created_at,
        updated_at=category.updated_at,
    )


@router.put("/categories/{category_id}", response_model=CategoryResponse)
async def update_category(
    category_id: str,
    category_data: CategoryUpdate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Update a category (admin only)"""
    category = db.query(BlogCategory).filter(BlogCategory.id == category_id).first()
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Category not found"
        )

    if category_data.name is not None:
        category.name = category_data.name

    if category_data.slug is not None:
        # Check if new slug already exists
        existing = (
            db.query(BlogCategory)
            .filter(
                BlogCategory.slug == category_data.slug, BlogCategory.id != category_id
            )
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Category with slug '{category_data.slug}' already exists",
            )
        category.slug = category_data.slug

    if category_data.description is not None:
        category.description = category_data.description

    db.commit()
    db.refresh(category)

    return CategoryResponse(
        id=category.id,
        name=category.name,
        slug=category.slug,
        description=category.description,
        created_at=category.created_at,
        updated_at=category.updated_at,
    )


@router.delete("/categories/{category_id}")
async def delete_category(
    category_id: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Delete a category (admin only)"""
    category = db.query(BlogCategory).filter(BlogCategory.id == category_id).first()
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Category not found"
        )

    db.delete(category)
    db.commit()

    return {"message": "Category deleted successfully"}


# ==================== Tag Endpoints ====================


@router.get("/tags", response_model=List[TagResponse])
async def list_tags(db: Session = Depends(get_db)):
    """List all tags (public endpoint)"""
    tags = db.query(BlogTag).order_by(BlogTag.name).all()
    return [
        TagResponse(id=tag.id, name=tag.name, slug=tag.slug, created_at=tag.created_at)
        for tag in tags
    ]


@router.post("/tags", response_model=TagResponse, status_code=status.HTTP_201_CREATED)
async def create_tag(
    tag_data: TagCreate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Create a new tag (admin only)"""
    slug = tag_data.slug or generate_slug(tag_data.name)

    # Check if slug already exists
    existing = db.query(BlogTag).filter(BlogTag.slug == slug).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tag with slug '{slug}' already exists",
        )

    tag = BlogTag(name=tag_data.name, slug=slug)

    db.add(tag)
    db.commit()
    db.refresh(tag)

    return TagResponse(
        id=tag.id, name=tag.name, slug=tag.slug, created_at=tag.created_at
    )


@router.put("/tags/{tag_id}", response_model=TagResponse)
async def update_tag(
    tag_id: str,
    tag_data: TagUpdate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Update a tag (admin only)"""
    tag = db.query(BlogTag).filter(BlogTag.id == tag_id).first()
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Tag not found"
        )

    if tag_data.name is not None:
        tag.name = tag_data.name

    if tag_data.slug is not None:
        # Check if new slug already exists
        existing = (
            db.query(BlogTag)
            .filter(BlogTag.slug == tag_data.slug, BlogTag.id != tag_id)
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tag with slug '{tag_data.slug}' already exists",
            )
        tag.slug = tag_data.slug

    db.commit()
    db.refresh(tag)

    return TagResponse(
        id=tag.id, name=tag.name, slug=tag.slug, created_at=tag.created_at
    )


@router.delete("/tags/{tag_id}")
async def delete_tag(
    tag_id: str, admin: User = Depends(require_admin), db: Session = Depends(get_db)
):
    """Delete a tag (admin only)"""
    tag = db.query(BlogTag).filter(BlogTag.id == tag_id).first()
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Tag not found"
        )

    db.delete(tag)
    db.commit()

    return {"message": "Tag deleted successfully"}
