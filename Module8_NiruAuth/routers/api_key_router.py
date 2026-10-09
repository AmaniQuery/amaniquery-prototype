"""
API Key Router
Manages API keys for users and integrations
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from ..models.pydantic_models import APIKeyCreate, APIKeyResponse, APIKeyListResponse
from ..dependencies import get_db, get_current_user
from ..models.auth_models import User, Integration
from ..providers.api_key_provider import APIKeyProvider

router = APIRouter(prefix="/api/v1/auth", tags=["API Keys"])


@router.post(
    "/integrations/{integration_id}/keys",
    response_model=APIKeyResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_integration_api_key(
    integration_id: str,
    key_data: APIKeyCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create API key for integration"""
    # Verify ownership
    integration = db.query(Integration).filter(Integration.id == integration_id).first()
    if not integration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Integration not found"
        )

    if integration.owner_user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized"
        )

    # Create API key
    full_key, api_key = APIKeyProvider.create_api_key(
        db=db,
        integration_id=integration_id,
        name=key_data.name,
        scopes=key_data.scopes,
        rate_limit_per_minute=key_data.rate_limit_per_minute,
        rate_limit_per_hour=key_data.rate_limit_per_hour,
        rate_limit_per_day=key_data.rate_limit_per_day,
        expires_at=key_data.expires_at,
    )

    return APIKeyResponse(
        id=api_key.id,
        key=full_key,  # Only shown on creation
        key_prefix=api_key.key_prefix,
        name=api_key.name,
        scopes=api_key.scopes,
        rate_limit_per_minute=api_key.rate_limit_per_minute,
        rate_limit_per_hour=api_key.rate_limit_per_hour,
        rate_limit_per_day=api_key.rate_limit_per_day,
        is_active=api_key.is_active,
        last_used=api_key.last_used,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
    )


@router.get(
    "/integrations/{integration_id}/keys", response_model=List[APIKeyListResponse]
)
async def list_integration_api_keys(
    integration_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List API keys for integration"""
    # Verify ownership
    integration = db.query(Integration).filter(Integration.id == integration_id).first()
    if not integration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Integration not found"
        )

    if integration.owner_user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized"
        )

    api_keys = APIKeyProvider.list_integration_api_keys(db, integration_id)

    return [
        APIKeyListResponse(
            id=key.id,
            key_prefix=key.key_prefix,
            name=key.name,
            scopes=key.scopes,
            is_active=key.is_active,
            last_used=key.last_used,
            expires_at=key.expires_at,
            created_at=key.created_at,
        )
        for key in api_keys
    ]


@router.delete("/integrations/{integration_id}/keys/{key_id}")
async def revoke_integration_api_key(
    integration_id: str,
    key_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Revoke API key"""
    # Verify ownership
    integration = db.query(Integration).filter(Integration.id == integration_id).first()
    if not integration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Integration not found"
        )

    if integration.owner_user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized"
        )

    success = APIKeyProvider.revoke_api_key(db, key_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )

    return {"message": "API key revoked successfully"}


@router.post(
    "/integrations/{integration_id}/keys/{key_id}/rotate", response_model=APIKeyResponse
)
async def rotate_integration_api_key(
    integration_id: str,
    key_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Rotate API key"""
    # Verify ownership
    integration = db.query(Integration).filter(Integration.id == integration_id).first()
    if not integration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Integration not found"
        )

    if integration.owner_user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized"
        )

    try:
        full_key, new_key = APIKeyProvider.rotate_api_key(db, key_id)

        return APIKeyResponse(
            id=new_key.id,
            key=full_key,  # Only shown on creation
            key_prefix=new_key.key_prefix,
            name=new_key.name,
            scopes=new_key.scopes,
            rate_limit_per_minute=new_key.rate_limit_per_minute,
            rate_limit_per_hour=new_key.rate_limit_per_hour,
            rate_limit_per_day=new_key.rate_limit_per_day,
            is_active=new_key.is_active,
            last_used=new_key.last_used,
            expires_at=new_key.expires_at,
            created_at=new_key.created_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# User API Keys
@router.post(
    "/users/keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED
)
async def create_user_api_key(
    key_data: APIKeyCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create API key for user"""
    full_key, api_key = APIKeyProvider.create_api_key(
        db=db,
        user_id=user.id,
        name=key_data.name,
        scopes=key_data.scopes,
        rate_limit_per_minute=key_data.rate_limit_per_minute,
        rate_limit_per_hour=key_data.rate_limit_per_hour,
        rate_limit_per_day=key_data.rate_limit_per_day,
        expires_at=key_data.expires_at,
    )

    return APIKeyResponse(
        id=api_key.id,
        key=full_key,
        key_prefix=api_key.key_prefix,
        name=api_key.name,
        scopes=api_key.scopes,
        rate_limit_per_minute=api_key.rate_limit_per_minute,
        rate_limit_per_hour=api_key.rate_limit_per_hour,
        rate_limit_per_day=api_key.rate_limit_per_day,
        is_active=api_key.is_active,
        last_used=api_key.last_used,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
    )


@router.get("/users/keys", response_model=List[APIKeyListResponse])
async def list_user_api_keys(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """List user's API keys"""
    api_keys = APIKeyProvider.list_user_api_keys(db, user.id)

    return [
        APIKeyListResponse(
            id=key.id,
            key_prefix=key.key_prefix,
            name=key.name,
            scopes=key.scopes,
            is_active=key.is_active,
            last_used=key.last_used,
            expires_at=key.expires_at,
            created_at=key.created_at,
        )
        for key in api_keys
    ]
