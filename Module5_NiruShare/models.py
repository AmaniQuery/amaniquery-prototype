"""
Pydantic models for sharing API
"""

from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field


class FormatRequest(BaseModel):
    """Request to format a response for sharing"""

    answer: str = Field(..., description="The RAG answer to format")
    sources: List[Dict] = Field(default_factory=list, description="Source citations")
    platform: str = Field(..., description="Target platform")
    query: Optional[str] = Field(None, description="Original query")
    include_hashtags: bool = Field(True, description="Include hashtags")
    style: Optional[str] = Field(
        None, description="Formatting style (professional, casual, engaging)"
    )


class FormatResponse(BaseModel):
    """Formatted post response"""

    platform: str
    content: Union[str, List[str]]  # String for single post, List for thread
    character_count: Optional[int] = None
    hashtags: List[str] = Field(default_factory=list)
    metadata: Dict = Field(default_factory=dict)


class ShareLinkRequest(BaseModel):
    """Request to generate share link"""

    platform: str = Field(..., description="Target platform")
    content: Union[str, List[str]] = Field(..., description="Formatted content")
    url: Optional[str] = Field(None, description="Optional URL to include")


class ShareLinkResponse(BaseModel):
    """Share link response"""

    platform: str
    share_url: str
    instructions: Optional[str] = None


class PreviewRequest(BaseModel):
    """Request to preview posts for all platforms"""

    answer: str
    sources: List[Dict] = Field(default_factory=list)
    query: Optional[str] = None
    style: Optional[str] = Field(None, description="Formatting style")


class PreviewResponse(BaseModel):
    """Preview response with all platforms"""

    platforms: Dict[str, Dict] = Field(
        ..., description="Dictionary of platform names to formatted posts"
    )


class PostRequest(BaseModel):
    """Request to post to a platform"""

    platform: str = Field(
        ..., description="Target platform (twitter, linkedin, facebook)"
    )
    content: Union[str, List[str]] = Field(..., description="Content to post")
    message_id: Optional[str] = Field(None, description="Chat message ID for tracking")


class PostResponse(BaseModel):
    """Response from posting to a platform"""

    platform: str
    post_id: Optional[str] = None
    status: str  # "success", "error"
    message: str
    url: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)


class AuthRequest(BaseModel):
    """Request for platform authentication"""

    platform: str = Field(..., description="Platform to authenticate with")


class AuthResponse(BaseModel):
    """Authentication response"""

    platform: str
    auth_url: Optional[str] = None
    status: str  # "authenticated", "needs_auth", "error"
    message: str
    user_info: Optional[Dict] = None


class AuthCallbackRequest(BaseModel):
    """OAuth callback request"""

    platform: str
    code: str
    state: Optional[str] = None


class ImageGenerationRequest(BaseModel):
    """Request to generate an image from text"""

    text: str = Field(..., description="Text content for image")
    title: Optional[str] = Field(None, description="Optional title text")
    color_scheme: str = Field(
        "default", description="Color scheme (default, dark, professional, vibrant)"
    )
    width: Optional[int] = Field(None, description="Image width in pixels")
    height: Optional[int] = Field(None, description="Image height in pixels")
    format: str = Field("PNG", description="Image format (PNG, JPEG)")


class ImageGenerationFromPostRequest(BaseModel):
    """Request to generate an image from formatted post"""

    post_content: str = Field(..., description="Formatted post content")
    query: Optional[str] = Field(None, description="Original query (used as title)")
    color_scheme: str = Field("default", description="Color scheme")
    width: Optional[int] = Field(None, description="Image width in pixels")
    height: Optional[int] = Field(None, description="Image height in pixels")
    format: str = Field("PNG", description="Image format (PNG, JPEG)")


class ImageGenerationResponse(BaseModel):
    """Response from image generation"""

    status: str
    format: str
    image_base64: str = Field(..., description="Base64 encoded image")
    width: int
    height: int
