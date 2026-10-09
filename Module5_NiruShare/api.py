"""
Sharing API endpoints
"""

import sys
from pathlib import Path
import os
from typing import Dict
import requests
import jwt
from fastapi import APIRouter, HTTPException, Request
from Module5_NiruShare.service import ShareService
from Module5_NiruShare.models import (
    FormatRequest,
    FormatResponse,
    ShareLinkRequest,
    ShareLinkResponse,
    PreviewRequest,
    PreviewResponse,
    PostRequest,
    PostResponse,
    AuthRequest,
    AuthResponse,
    AuthCallbackRequest,
    ImageGenerationRequest,
    ImageGenerationFromPostRequest,
    ImageGenerationResponse,
)
from Module5_NiruShare.utils.token_manager import token_manager

# Create router
router = APIRouter(prefix="/share", tags=["Social Sharing"])

# Initialize service
share_service = ShareService()


def get_current_user_id(request: Request) -> str:
    """
    Get current user ID from JWT session token
    Validates JWT tokens for production security
    """
    session_token = request.headers.get("X-Session-Token")
    if not session_token:
        raise HTTPException(status_code=401, detail="Session token required")

    try:
        payload = jwt.decode(
            session_token, os.getenv("JWT_SECRET_KEY"), algorithms=["HS256"]
        )
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: no user_id")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/format", response_model=FormatResponse)
async def format_for_platform(request: FormatRequest):
    """
    Format a response for specific social media platform

    **Supported platforms:**
    - `twitter`: 280 chars (or thread)
    - `linkedin`: Professional format (3000 char limit)
    - `facebook`: Engaging format

    **Example:**
    ```json
    {
      "answer": "The Kenyan Constitution protects freedom of expression...",
      "sources": [...],
      "platform": "twitter",
      "query": "What does the Constitution say about free speech?"
    }
    ```
    """
    try:
        formatted = share_service.format_for_platform(
            answer=request.answer,
            sources=request.sources or [],
            platform=request.platform,
            query=request.query,
            include_hashtags=request.include_hashtags,
            style=request.style,
        )

        return FormatResponse(
            platform=formatted.get("platform", request.platform),
            content=formatted.get("content", ""),
            character_count=formatted.get("character_count"),
            hashtags=formatted.get("hashtags", []),
            metadata={
                k: v
                for k, v in formatted.items()
                if k not in ["platform", "content", "character_count", "hashtags"]
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error formatting post: {str(e)}")


@router.post("/generate-link", response_model=ShareLinkResponse)
async def generate_share_link(request: ShareLinkRequest):
    """
    Generate platform-specific share link

    **Example:**
    ```json
    {
      "platform": "twitter",
      "content": "Check out this insight from AmaniQuery! #Kenya",
      "url": "https://amaniquery.ke/query/123"
    }
    ```
    """
    try:
        # Handle thread case (take first tweet)
        content = request.content
        if isinstance(content, list):
            if not content:
                raise ValueError("Content list cannot be empty")
            content = str(content[0]) if content[0] else ""
        else:
            content = str(content) if content else ""

        if not content:
            raise ValueError("Content cannot be empty")

        share_url = share_service.generate_share_link(
            platform=request.platform,
            formatted_content=content,
            url=request.url,
        )

        # Add platform-specific instructions
        instructions = {
            "twitter": "Click to open Twitter with pre-filled text. You can edit before posting.",
            "linkedin": "Click to open LinkedIn sharing dialog. Paste your formatted content.",
            "facebook": "Click to open Facebook sharing dialog. Paste your formatted content.",
        }

        return ShareLinkResponse(
            platform=request.platform,
            share_url=share_url,
            instructions=instructions.get(request.platform.lower()),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating link: {str(e)}")


@router.post("/preview", response_model=PreviewResponse)
async def preview_all_platforms(request: PreviewRequest):
    """
    Preview formatted posts for all platforms

    Returns formatted versions for Twitter, LinkedIn, and Facebook

    **Example:**
    ```json
    {
      "answer": "Recent parliamentary debates focused on...",
      "sources": [...],
      "query": "What are recent Parliament debates about?"
    }
    ```
    """
    try:
        previews = share_service.preview_all_platforms(
            answer=request.answer,
            sources=request.sources or [],
            query=request.query,
            style=request.style,
        )

        return PreviewResponse(platforms=previews)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating previews: {str(e)}"
        )


@router.get("/platforms")
async def get_supported_platforms():
    """Get list of supported platforms"""
    platforms = share_service.get_supported_platforms()
    return {"platforms": platforms}


@router.post("/stats")
async def get_post_stats(formatted_post: dict):
    """Get statistics for a formatted post"""
    try:
        stats = share_service.get_platform_stats(formatted_post)
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error calculating stats: {str(e)}"
        )


@router.post("/post", response_model=PostResponse)
async def post_to_platform(request: PostRequest, req: Request):
    """
    Post content directly to a social media platform

    **Supported platforms:** twitter, linkedin, facebook

    **Requirements:**
    - User must be authenticated with the platform
    - Valid access token must be provided (handled via auth flow)

    **Example:**
    ```json
    {
      "platform": "twitter",
      "content": "Check out this insight from AmaniQuery! #KenyaLaw",
      "message_id": "msg_123"
    }
    ```

    **Note:** Authentication flow must be completed first via /auth endpoints
    """
    try:
        user_id = get_current_user_id(req)

        # Get token from secure storage
        token_data = token_manager.get_token(user_id, request.platform)

        if not token_data or not token_data.get("access_token"):
            raise HTTPException(
                status_code=401,
                detail=f"Not authenticated with {request.platform}. Please complete authentication first.",
            )

        # Validate token before use
        if not token_manager.validate_token(user_id, request.platform):
            # Try to refresh token
            if not token_manager.refresh_token(user_id, request.platform):
                token_manager.delete_token(user_id, request.platform)
                raise HTTPException(
                    status_code=401,
                    detail=f"Authentication expired for {request.platform}. Please re-authenticate.",
                )

        access_token = token_data["access_token"]

        # Handle thread case (take first item for single posts)
        content = request.content
        if isinstance(content, list):
            content = content[0]

        result = await share_service.post_to_platform(
            platform=request.platform,
            content=content,
            access_token=access_token,
            message_id=request.message_id,
        )

        return PostResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error posting to platform: {str(e)}"
        )


@router.post("/auth", response_model=AuthResponse)
async def authenticate_platform(request: AuthRequest, req: Request):
    """
    Initiate OAuth authentication flow for a platform

    **Supported platforms:** twitter, linkedin, facebook

    **Example:**
    ```json
    {
      "platform": "linkedin"
    }
    ```

    Returns authentication URL to redirect user to for OAuth consent.
    """
    try:
        user_id = get_current_user_id(req)

        result = share_service.get_auth_url(request.platform)

        # If platform is Twitter and code_verifier is returned, store it securely
        if request.platform == "twitter" and result.get("code_verifier"):
            code_verifier = result.pop("code_verifier")  # Remove from response
            token_manager.store_oauth_state(
                user_id, request.platform, {"code_verifier": code_verifier}
            )

        return AuthResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initiating auth: {str(e)}")


@router.post("/auth/status", response_model=AuthResponse)
async def check_auth_status(request: AuthRequest, req: Request):
    """
    Check authentication status for a platform

    **Example:**
    ```json
    {
      "platform": "linkedin"
    }
    ```

    Returns whether the user is authenticated with the platform.
    """
    try:
        user_id = get_current_user_id(req)

        # Check if we have a valid token
        token_data = token_manager.get_token(user_id, request.platform)

        if token_data and token_data.get("access_token"):
            # Validate token with platform
            is_valid = token_manager.validate_token(user_id, request.platform)

            if is_valid:
                return AuthResponse(
                    platform=request.platform,
                    status="authenticated",
                    message="Authenticated and token is valid",
                    user_info=token_data.get("user_info", {}),
                )
            else:
                # Try to refresh token
                if token_manager.refresh_token(user_id, request.platform):
                    return AuthResponse(
                        platform=request.platform,
                        status="authenticated",
                        message="Token refreshed successfully",
                        user_info=token_data.get("user_info", {}),
                    )
                else:
                    # Token invalid and couldn't refresh
                    token_manager.delete_token(user_id, request.platform)
                    return AuthResponse(
                        platform=request.platform,
                        status="not_authenticated",
                        message="Authentication expired and could not be refreshed",
                    )

        return AuthResponse(
            platform=request.platform,
            status="not_authenticated",
            message="Not authenticated with this platform",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error checking auth status: {str(e)}"
        )


@router.post("/generate-image", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    """
    Generate an image from text content

    **Example:**
    ```json
    {
      "text": "The Kenyan Constitution protects freedom of expression...",
      "title": "Constitutional Rights",
      "color_scheme": "professional",
      "format": "PNG"
    }
    ```
    """
    try:
        result = share_service.generate_image(
            text=request.text,
            title=request.title,
            color_scheme=request.color_scheme,
            width=request.width,
            height=request.height,
            format=request.format,
        )
        return ImageGenerationResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")


@router.post("/generate-image-from-post", response_model=ImageGenerationResponse)
async def generate_image_from_post(request: ImageGenerationFromPostRequest):
    """
    Generate an image from formatted post content

    **Example:**
    ```json
    {
      "post_content": "Check out this insight from AmaniQuery! #Kenya",
      "query": "What does the Constitution say?",
      "color_scheme": "default",
      "format": "PNG"
    }
    ```
    """
    try:
        result = share_service.generate_image_from_post(
            post_content=request.post_content,
            query=request.query,
            color_scheme=request.color_scheme,
            width=request.width,
            height=request.height,
            format=request.format,
        )
        return ImageGenerationResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")


@router.post("/auth/callback", response_model=AuthResponse)
async def handle_auth_callback(request: AuthCallbackRequest, req: Request):
    """
    Handle OAuth callback from platform

    **Note:** This endpoint receives the authorization code from the OAuth redirect
    and exchanges it for access tokens. In production, this would be called by
    the platform's redirect URI.

    **Example:**
    ```json
    {
      "platform": "linkedin",
      "code": "auth_code_from_platform",
      "state": "linkedin"
    }
    ```
    """
    try:
        user_id = get_current_user_id(req)

        # Validate state parameter for security
        if request.state != request.platform:
            raise HTTPException(status_code=400, detail="Invalid state parameter")

        # Exchange authorization code for access token
        if request.platform == "linkedin":
            token_result = await exchange_linkedin_code(request.code)
        elif request.platform == "facebook":
            token_result = await exchange_facebook_code(request.code)
        elif request.platform == "twitter":
            token_result = await exchange_twitter_code(request.code, user_id)
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported platform: {request.platform}"
            )

        if token_result.get("access_token"):
            # Store access token securely
            token_data = {
                "access_token": token_result["access_token"],
                "user_info": token_result.get("user_info"),
                "platform": request.platform,
                "token_type": token_result.get("token_type", "bearer"),
                "scope": token_result.get("scope"),
            }

            # Add refresh token if available
            if token_result.get("refresh_token"):
                token_data["refresh_token"] = token_result["refresh_token"]

            success = token_manager.store_token(user_id, request.platform, token_data)

            if success:
                return AuthResponse(
                    platform=request.platform,
                    status="authenticated",
                    message="Successfully authenticated",
                    user_info=token_result.get("user_info"),
                )
            else:
                raise HTTPException(
                    status_code=500, detail="Failed to store access token"
                )
        else:
            raise HTTPException(status_code=400, detail="Failed to obtain access token")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error handling auth callback: {str(e)}"
        )


async def exchange_linkedin_code(code: str) -> Dict:
    """Exchange LinkedIn authorization code for access token"""
    client_id = os.getenv("LINKEDIN_CLIENT_ID")
    client_secret = os.getenv("LINKEDIN_CLIENT_SECRET")
    redirect_uri = os.getenv(
        "LINKEDIN_REDIRECT_URI", "http://localhost:8000/share/auth/callback"
    )

    if not client_id or not client_secret:
        raise HTTPException(
            status_code=500, detail="LinkedIn API credentials not configured"
        )

    url = "https://www.linkedin.com/oauth/v2/accessToken"
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "client_secret": client_secret,
    }

    response = requests.post(url, data=data)
    response.raise_for_status()

    token_data = response.json()
    access_token = token_data.get("access_token")

    # Get user info
    headers = {"Authorization": f"Bearer {access_token}"}
    user_response = requests.get(
        "https://api.linkedin.com/v2/people/~", headers=headers
    )

    user_info = None
    if user_response.status_code == 200:
        user_data = user_response.json()
        user_info = {
            "id": user_data.get("id"),
            "name": f"{user_data.get('localizedFirstName', '')} {user_data.get('localizedLastName', '')}".strip(),
        }

    return {
        "access_token": access_token,
        "user_info": user_info,
    }


async def exchange_facebook_code(code: str) -> Dict:
    """Exchange Facebook authorization code for access token"""
    app_id = os.getenv("FACEBOOK_APP_ID")
    app_secret = os.getenv("FACEBOOK_APP_SECRET")
    redirect_uri = os.getenv(
        "FACEBOOK_REDIRECT_URI", "http://localhost:8000/share/auth/callback"
    )

    if not app_id or not app_secret:
        raise HTTPException(
            status_code=500, detail="Facebook API credentials not configured"
        )

    url = "https://graph.facebook.com/v18.0/oauth/access_token"
    params = {
        "client_id": app_id,
        "redirect_uri": redirect_uri,
        "client_secret": app_secret,
        "code": code,
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    token_data = response.json()
    access_token = token_data.get("access_token")

    # Get user info
    user_response = requests.get(
        f"https://graph.facebook.com/me?access_token={access_token}"
    )

    user_info = None
    if user_response.status_code == 200:
        user_data = user_response.json()
        user_info = {
            "id": user_data.get("id"),
            "name": user_data.get("name"),
        }

    return {
        "access_token": access_token,
        "user_info": user_info,
    }


async def exchange_twitter_code(code: str, user_id: str) -> Dict:
    """Exchange Twitter/X authorization code for access token"""
    client_id = os.getenv("TWITTER_CLIENT_ID")
    client_secret = os.getenv("TWITTER_CLIENT_SECRET")
    redirect_uri = os.getenv(
        "TWITTER_REDIRECT_URI", "http://localhost:8000/share/auth/callback"
    )

    if not client_id or not client_secret:
        raise HTTPException(
            status_code=500, detail="Twitter API credentials not configured"
        )

    # Retrieve code_verifier from secure storage
    oauth_state = token_manager.get_oauth_state(user_id, "twitter")

    if not oauth_state or not oauth_state.get("code_verifier"):
        raise HTTPException(
            status_code=400,
            detail="Code verifier not found. Please restart authentication.",
        )

    code_verifier = oauth_state["code_verifier"]

    # Create Basic Auth header
    import base64

    auth_string = f"{client_id}:{client_secret}"
    auth_header = base64.b64encode(auth_string.encode()).decode()

    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }

    response = requests.post(
        "https://api.twitter.com/2/oauth2/token", headers=headers, data=data
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=400, detail=f"Twitter token exchange failed: {response.text}"
        )

    token_data = response.json()
    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")

    if not access_token:
        raise HTTPException(
            status_code=400, detail="No access token received from Twitter"
        )

    # Delete the code_verifier after successful exchange
    token_manager.delete_oauth_state(user_id, "twitter")

    # Get user info using the access token
    headers = {"Authorization": f"Bearer {access_token}"}
    user_response = requests.get(
        "https://api.twitter.com/2/users/me",
        headers=headers,
        params={"user.fields": "id,name,username,profile_image_url"},
    )

    user_info = None
    if user_response.status_code == 200:
        user_data = user_response.json().get("data", {})
        user_info = {
            "id": user_data.get("id"),
            "name": user_data.get("name"),
            "username": user_data.get("username"),
            "profile_image_url": user_data.get("profile_image_url"),
        }

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": token_data.get("token_type", "bearer"),
        "scope": token_data.get("scope"),
        "user_info": user_info,
    }
