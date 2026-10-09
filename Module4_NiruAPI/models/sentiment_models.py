"""
Sentiment API Endpoints
Provides public sentiment analysis for topics
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    """Request model for sentiment analysis"""

    topic: str = Field(..., description="Topic to analyze sentiment for", min_length=1)
    category: Optional[str] = Field(
        None, description="Filter by category (Kenyan News, Global Trend)"
    )
    days: int = Field(30, description="Number of days to look back", ge=1, le=365)


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis"""

    topic: str
    sentiment_percentages: dict = Field(
        ..., description="Percentage breakdown (positive, negative, neutral)"
    )
    sentiment_distribution: dict = Field(
        ..., description="Count of articles by sentiment"
    )
    average_polarity: float = Field(
        ..., description="Average polarity score (-1.0 to 1.0)"
    )
    average_subjectivity: float = Field(
        ..., description="Average subjectivity score (0.0 to 1.0)"
    )
    total_articles: int = Field(..., description="Total articles analyzed")
    category_filter: Optional[str] = None
    time_period_days: int


class TopicSentimentTrend(BaseModel):
    """Sentiment trend over time for a topic"""

    date: str
    polarity: float
    article_count: int
