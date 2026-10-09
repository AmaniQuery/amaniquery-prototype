"""
OAuth Router
Handles OAuth 2.0 flows
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional

from ..models.pydantic_models import (
    OAuthClientCreate,
    OAuthClientResponse,
    OAuthTokenRequest,
    OAuthTokenResponse,
    OAuthAuthorizeRequest,
)
from ..dependencies import get_db, get_current_user
from ..models.auth_models import User
from ..providers.oauth2_provider import OAuth2Provider

router = APIRouter(prefix="/api/v1/auth/oauth", tags=["OAuth"])


@router.post(
    "/clients", response_model=OAuthClientResponse, status_code=status.HTTP_201_CREATED
)
async def create_oauth_client(
    client_data: OAuthClientCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Register OAuth client"""
    client_id, client_secret, client = OAuth2Provider.create_client(
        db=db,
        owner_user_id=user.id,
        name=client_data.name,
        description=client_data.description,
        redirect_uris=client_data.redirect_uris,
        grant_types=client_data.grant_types,
        scopes=client_data.scopes,
    )

    return OAuthClientResponse(
        id=client.id,
        client_id=client_id,
        client_secret=client_secret,  # Only shown on creation
        name=client.name,
        description=client.description,
        redirect_uris=client.redirect_uris,
        grant_types=client.grant_types,
        scopes=client.scopes,
        is_active=client.is_active,
        created_at=client.created_at,
    )


@router.get("/authorize")
async def authorize(
    client_id: str = Query(...),
    redirect_uri: str = Query(...),
    response_type: str = Query("code"),
    scope: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """OAuth authorization endpoint"""
    # Validate client
    client = OAuth2Provider.get_client_by_id(db, client_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid client_id"
        )

    if redirect_uri not in (client.redirect_uris or []):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid redirect_uri"
        )

    # Generate authorization code
    scopes = scope.split() if scope else []
    code = OAuth2Provider.generate_authorization_code(
        db=db,
        client_id=client_id,
        user_id=user.id,
        redirect_uri=redirect_uri,
        scopes=scopes,
    )

    # Redirect with code
    redirect_url = f"{redirect_uri}?code={code}"
    if state:
        redirect_url += f"&state={state}"

    return {"redirect_uri": redirect_url}


@router.post("/token", response_model=OAuthTokenResponse)
async def token(token_request: OAuthTokenRequest, db: Session = Depends(get_db)):
    """OAuth token endpoint"""
    if token_request.grant_type == "authorization_code":
        if not token_request.code or not token_request.client_secret:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing code or client_secret",
            )

        result = OAuth2Provider.exchange_authorization_code(
            db=db,
            code=token_request.code,
            client_id=token_request.client_id,
            client_secret=token_request.client_secret,
            redirect_uri=token_request.redirect_uri or "",
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid authorization code",
            )

        return OAuthTokenResponse(**result)

    elif token_request.grant_type == "client_credentials":
        if not token_request.client_secret:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Missing client_secret"
            )

        client = OAuth2Provider.validate_client(
            db, token_request.client_id, token_request.client_secret
        )
        if not client:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid client credentials",
            )

        scopes = token_request.scope.split() if token_request.scope else []
        result = OAuth2Provider.generate_access_token(db, client, None, scopes)

        return OAuthTokenResponse(**result)

    elif token_request.grant_type == "refresh_token":
        if not token_request.refresh_token or not token_request.client_secret:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing refresh_token or client_secret",
            )

        result = OAuth2Provider.refresh_access_token(
            db=db,
            refresh_token=token_request.refresh_token,
            client_id=token_request.client_id,
            client_secret=token_request.client_secret,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid refresh token"
            )

        return OAuthTokenResponse(**result)

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported grant_type: {token_request.grant_type}",
        )


@router.post("/revoke")
async def revoke_token(token: str = Query(...), db: Session = Depends(get_db)):
    """Revoke OAuth token"""
    success = OAuth2Provider.revoke_token(db, token)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token not found or already revoked",
        )

    return {"message": "Token revoked successfully"}
