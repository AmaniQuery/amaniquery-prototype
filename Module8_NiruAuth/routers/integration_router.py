"""
Integration Router
Manages third-party integrations
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from ..models.pydantic_models import (
    IntegrationCreate,
    IntegrationUpdate,
    IntegrationResponse,
)
from ..dependencies import get_db, get_current_user
from ..models.auth_models import Integration, User
from ..authorization.role_manager import RoleManager

router = APIRouter(prefix="/api/v1/auth/integrations", tags=["Integrations"])


@router.post(
    "", response_model=IntegrationResponse, status_code=status.HTTP_201_CREATED
)
async def create_integration(
    integration_data: IntegrationCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new integration"""
    integration = Integration(
        name=integration_data.name,
        description=integration_data.description,
        type=integration_data.type,
        owner_user_id=user.id,
        webhook_url=integration_data.webhook_url,
        ip_whitelist=integration_data.ip_whitelist,
        extra_data=(
            integration_data.metadata
            if hasattr(integration_data, "metadata")
            and integration_data.metadata is not None
            else None
        ),
    )

    db.add(integration)
    db.commit()
    db.refresh(integration)

    # Assign default read-only role
    read_only_role = RoleManager.get_role_by_name(db, "integration_read_only")
    if read_only_role:
        RoleManager.assign_role_to_integration(
            db, integration.id, read_only_role.id, user.id
        )

    return IntegrationResponse(
        id=integration.id,
        name=integration.name,
        description=integration.description,
        type=integration.type,
        owner_user_id=integration.owner_user_id,
        status=integration.status,
        webhook_url=integration.webhook_url,
        ip_whitelist=integration.ip_whitelist,
        metadata=integration.extra_data,
        created_at=integration.created_at,
        updated_at=integration.updated_at,
        roles=[],
    )


@router.get("", response_model=List[IntegrationResponse])
async def list_integrations(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """List user's integrations"""
    integrations = (
        db.query(Integration).filter(Integration.owner_user_id == user.id).all()
    )

    return [
        IntegrationResponse(
            id=integration.id,
            name=integration.name,
            description=integration.description,
            type=integration.type,
            owner_user_id=integration.owner_user_id,
            status=integration.status,
            webhook_url=integration.webhook_url,
            ip_whitelist=integration.ip_whitelist,
            metadata=integration.extra_data,
            created_at=integration.created_at,
            updated_at=integration.updated_at,
            roles=RoleManager.get_integration_roles(db, integration.id),
        )
        for integration in integrations
    ]


@router.get("/{integration_id}", response_model=IntegrationResponse)
async def get_integration(
    integration_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get integration by ID"""
    integration = db.query(Integration).filter(Integration.id == integration_id).first()
    if not integration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Integration not found"
        )

    # Check ownership
    if integration.owner_user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized"
        )

    return IntegrationResponse(
        id=integration.id,
        name=integration.name,
        description=integration.description,
        type=integration.type,
        owner_user_id=integration.owner_user_id,
        status=integration.status,
        webhook_url=integration.webhook_url,
        ip_whitelist=integration.ip_whitelist,
        metadata=integration.extra_data,
        created_at=integration.created_at,
        updated_at=integration.updated_at,
        roles=RoleManager.get_integration_roles(db, integration.id),
    )


@router.put("/{integration_id}", response_model=IntegrationResponse)
async def update_integration(
    integration_id: str,
    integration_data: IntegrationUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update integration"""
    integration = db.query(Integration).filter(Integration.id == integration_id).first()
    if not integration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Integration not found"
        )

    # Check ownership
    if integration.owner_user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized"
        )

    if integration_data.name is not None:
        integration.name = integration_data.name
    if integration_data.description is not None:
        integration.description = integration_data.description
    if integration_data.status is not None:
        integration.status = integration_data.status
    if integration_data.webhook_url is not None:
        integration.webhook_url = integration_data.webhook_url
    if integration_data.ip_whitelist is not None:
        integration.ip_whitelist = integration_data.ip_whitelist
    if integration_data.metadata is not None:
        integration.extra_data = integration_data.metadata

    db.commit()
    db.refresh(integration)

    return IntegrationResponse(
        id=integration.id,
        name=integration.name,
        description=integration.description,
        type=integration.type,
        owner_user_id=integration.owner_user_id,
        status=integration.status,
        webhook_url=integration.webhook_url,
        ip_whitelist=integration.ip_whitelist,
        metadata=integration.extra_data,
        created_at=integration.created_at,
        updated_at=integration.updated_at,
        roles=RoleManager.get_integration_roles(db, integration.id),
    )


@router.delete("/{integration_id}")
async def delete_integration(
    integration_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete integration"""
    integration = db.query(Integration).filter(Integration.id == integration_id).first()
    if not integration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Integration not found"
        )

    # Check ownership
    if integration.owner_user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized"
        )

    db.delete(integration)
    db.commit()

    return {"message": "Integration deleted successfully"}
