"""
News Aggregation API Router
"""

from fastapi import APIRouter, Query, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel
from loguru import logger
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

router = APIRouter(prefix="/api/v1/news", tags=["news"])


class NewsArticle(BaseModel):
    """News article response model"""

    id: str
    url: str
    title: str
    content: str
    summary: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[str] = None
    source_name: str
    category: str
    quality_score: Optional[float] = None
    created_at: str


class NewsListResponse(BaseModel):
    """News list response"""

    articles: List[NewsArticle]
    total: int
    page: int
    page_size: int


class NewsSearchRequest(BaseModel):
    """News search request"""

    query: str
    sources: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    min_quality_score: Optional[float] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    limit: int = 20
    offset: int = 0


@router.get("/", response_model=NewsListResponse)
async def list_news(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sources: Optional[str] = Query(None, description="Comma-separated source names"),
    categories: Optional[str] = Query(None, description="Comma-separated categories"),
    min_quality_score: Optional[float] = Query(None, ge=0.0, le=1.0),
    days: int = Query(7, ge=1, le=30, description="Number of days to look back"),
):
    """List news articles with filtering"""
    try:
        from Module4_NiruAPI.services.news_service import NewsService

        service = NewsService()

        source_list = sources.split(",") if sources else None
        category_list = categories.split(",") if categories else None

        date_from = (datetime.utcnow() - timedelta(days=days)).isoformat()

        articles, total = service.get_articles(
            sources=source_list,
            categories=category_list,
            min_quality_score=min_quality_score,
            date_from=date_from,
            limit=page_size,
            offset=(page - 1) * page_size,
        )

        return NewsListResponse(
            articles=articles, total=total, page=page, page_size=page_size
        )
    except Exception as e:
        logger.error(f"Error listing news: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{article_id}", response_model=NewsArticle)
async def get_article(article_id: str):
    """Get a single article by ID"""
    try:
        from Module4_NiruAPI.services.news_service import NewsService

        service = NewsService()

        article = service.get_article_by_id(article_id)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")

        return article
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting article: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=NewsListResponse)
async def search_news(request: NewsSearchRequest):
    """Search news articles semantically"""
    try:
        from Module4_NiruAPI.services.news_service import NewsService

        service = NewsService()

        articles, total = service.search_articles(
            query=request.query,
            sources=request.sources,
            categories=request.categories,
            min_quality_score=request.min_quality_score,
            date_from=request.date_from,
            date_to=request.date_to,
            limit=request.limit,
            offset=request.offset,
        )

        return NewsListResponse(
            articles=articles,
            total=total,
            page=(request.offset // request.limit) + 1,
            page_size=request.limit,
        )
    except Exception as e:
        logger.error(f"Error searching news: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources/list")
async def list_sources():
    """List all available news sources"""
    try:
        from Module4_NiruAPI.services.news_service import NewsService

        service = NewsService()

        sources = service.get_sources()
        return {"sources": sources}
    except Exception as e:
        logger.error(f"Error listing sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories/list")
async def list_categories():
    """List all available categories"""
    try:
        from Module4_NiruAPI.services.news_service import NewsService

        service = NewsService()

        categories = service.get_categories()
        return {"categories": categories}
    except Exception as e:
        logger.error(f"Error listing categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))
