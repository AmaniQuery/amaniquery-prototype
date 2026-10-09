"""
Pydantic models for API requests and responses
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for query endpoint"""

    query: str = Field(..., description="The question or query text", min_length=1)
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    category: Optional[str] = Field(None, description="Filter by category")
    source: Optional[str] = Field(None, description="Filter by source name")
    include_sources: bool = Field(True, description="Include source citations")
    temperature: float = Field(0.7, description="LLM temperature", ge=0.0, le=2.0)
    max_tokens: int = Field(
        1500, description="Maximum tokens in response", ge=100, le=4000
    )
    session_id: Optional[str] = Field(
        None, description="Optional chat session ID to save messages"
    )


class Source(BaseModel):
    """Source citation model"""

    title: str
    url: str
    source_name: str
    category: str
    author: Optional[str] = None
    publication_date: Optional[str] = None
    relevance_score: Optional[float] = None
    excerpt: Optional[str] = None
    sentiment_label: Optional[str] = None  # New: sentiment for news sources
    sentiment_polarity: Optional[float] = None  # New
    # YouTube video metadata
    video_id: Optional[str] = None
    timestamp_url: Optional[str] = None
    timestamp_formatted: Optional[str] = None
    start_time_seconds: Optional[float] = None


class WidgetInput(BaseModel):
    """Input field for interactive widget"""

    name: str
    label: str
    type: str = "number"
    placeholder: Optional[str] = None
    default_value: Optional[str] = None


class WidgetOutput(BaseModel):
    """Output field for interactive widget"""

    label: str
    format: str = "{value}"  # e.g. "KES {value}"


class InteractiveWidget(BaseModel):
    """Interactive widget definition"""

    type: str  # salary_calculator, fine_calculator, etc.
    title: str
    description: str
    formula: str  # JavaScript evaluable string
    inputs: List[WidgetInput]
    outputs: List[WidgetOutput]
    source_citation: Optional[str] = None


class GithubDiff(BaseModel):
    """GitHub-style diff for legal amendments"""

    old_text: str
    new_text: str
    title: str
    highlight_type: str = "side_by_side"  # side_by_side or unified


class QueryResponse(BaseModel):
    """Response model for query endpoint"""

    answer: str = Field(..., description="The generated answer")
    sources: List[Source] = Field(default_factory=list, description="Source citations")
    query_time: float = Field(..., description="Query processing time in seconds")
    retrieved_chunks: int = Field(..., description="Number of chunks retrieved")
    model_used: str = Field(..., description="LLM model used")
    structured_data: Optional[Dict] = Field(
        None, description="Structured response data from AK-RAG"
    )
    interactive_widgets: Optional[List[InteractiveWidget]] = Field(
        None, description="Interactive widgets for policy queries"
    )
    github_diff: Optional[GithubDiff] = Field(
        None, description="GitHub-style diff for amendments"
    )


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    database_chunks: int
    embedding_model: str
    llm_provider: str


class StatsResponse(BaseModel):
    """Statistics response"""

    total_chunks: int
    categories: Dict[str, int]
    sources: List[str]
    database_size_mb: Optional[float] = None


class AlignmentRequest(BaseModel):
    """Request model for constitutional alignment analysis"""

    query: str = Field(
        ..., description="Query about bill-constitution alignment", min_length=10
    )
    bill_top_k: int = Field(
        3, description="Number of Bill chunks to retrieve", ge=1, le=10
    )
    constitution_top_k: int = Field(
        3, description="Number of Constitution chunks to retrieve", ge=1, le=10
    )
    temperature: float = Field(
        0.3, description="LLM temperature (lower for more factual)", ge=0.0, le=1.0
    )
    max_tokens: int = Field(
        2000, description="Maximum tokens in analysis", ge=500, le=4000
    )


class BillContext(BaseModel):
    """Bill context chunk"""

    text: str
    clause_number: Optional[str] = None
    subject: Optional[str] = None
    title: str


class ConstitutionContext(BaseModel):
    """Constitution context chunk"""

    text: str
    article_number: Optional[str] = None
    article_title: Optional[str] = None
    clause: Optional[str] = None


class AlignmentMetadata(BaseModel):
    """Metadata for alignment analysis"""

    bill_name: Optional[str] = None
    legal_concepts: List[str] = Field(default_factory=list)
    analysis_type: str = "alignment"
    bill_chunks_count: int
    constitution_chunks_count: int


class AlignmentResponse(BaseModel):
    """Response model for constitutional alignment analysis"""

    analysis: str = Field(..., description="Structured comparative analysis")
    bill_context: List[BillContext] = Field(
        default_factory=list, description="Bill chunks used"
    )
    constitution_context: List[ConstitutionContext] = Field(
        default_factory=list, description="Constitution chunks used"
    )
    metadata: AlignmentMetadata = Field(..., description="Analysis metadata")
    query_time: Optional[float] = None


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
