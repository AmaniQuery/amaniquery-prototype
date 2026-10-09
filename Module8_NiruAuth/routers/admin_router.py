"""
Admin Router
Admin endpoints for user management, role management, etc.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_, func
from typing import List, Optional
from datetime import datetime
import logging

from ..models.pydantic_models import (
    UserResponse,
    UserListResponse,
    UserUpdate,
    RoleResponse,
    RoleCreate,
    RoleUpdate,
    UserRoleAssign,
)
from ..dependencies import get_db, require_admin
from ..models.auth_models import User, Role, UserSession
from ..authorization.role_manager import RoleManager
from ..authorization.user_role_manager import UserRoleManager

router = APIRouter(prefix="/api/v1/auth/admin", tags=["Admin"])
logger = logging.getLogger(__name__)


# User Management
@router.get("/users", response_model=UserListResponse)
async def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, description="Search by email or name"),
    status: Optional[str] = Query(None, description="Filter by status"),
    email_verified: Optional[bool] = Query(
        None, description="Filter by email verification"
    ),
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """List all users with search and filters (admin only)"""
    offset = (page - 1) * page_size
    query = db.query(User)

    # Apply search filter
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(User.email.ilike(search_term), User.name.ilike(search_term))
        )

    # Apply status filter
    if status:
        query = query.filter(User.status == status)

    # Apply email verification filter
    if email_verified is not None:
        query = query.filter(User.email_verified == email_verified)

    total = query.count()
    users = query.order_by(User.created_at.desc()).offset(offset).limit(page_size).all()

    # Get roles for each user
    user_responses = []
    for user in users:
        user_roles = RoleManager.get_user_roles(db, user.id)
        role_names = [role.name for role in user_roles]

        user_responses.append(
            UserResponse(
                id=user.id,
                email=user.email,
                name=user.name,
                status=user.status,
                email_verified=user.email_verified,
                last_login=user.last_login,
                profile_image_url=user.profile_image_url,
                created_at=user.created_at,
                updated_at=user.updated_at,
                roles=role_names,
            )
        )

    return UserListResponse(
        users=user_responses, total=total, page=page, page_size=page_size
    )


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str, admin: User = Depends(require_admin), db: Session = Depends(get_db)
):
    """Get user by ID with full details (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Get user roles
    user_roles = RoleManager.get_user_roles(db, user.id)
    role_names = [role.name for role in user_roles]

    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        status=user.status,
        email_verified=user.email_verified,
        last_login=user.last_login,
        profile_image_url=user.profile_image_url,
        created_at=user.created_at,
        updated_at=user.updated_at,
        roles=role_names,
    )


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    user_data: UserUpdate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Update user details (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Prevent admin from modifying themselves accidentally
    if user.id == admin.id and user_data.status and user_data.status != "active":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own account status",
        )

    if user_data.name is not None:
        user.name = user_data.name  # type: ignore

    if user_data.email is not None:
        existing = db.query(User).filter(User.email == user_data.email.lower()).first()
        if existing and existing.id != user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Email already in use"
            )
        user.email = user_data.email.lower()  # type: ignore

    if user_data.status is not None:
        user.status = user_data.status  # type: ignore
        logger.info(
            f"Admin {admin.email} changed user {user.email} status to {user_data.status}"
        )

    if user_data.email_verified is not None:
        user.email_verified = user_data.email_verified  # type: ignore
        logger.info(
            f"Admin {admin.email} set email_verified={user_data.email_verified} for user {user.email}"
        )

    user.updated_at = datetime.utcnow()  # type: ignore
    db.commit()
    db.refresh(user)

    # Get user roles
    user_roles = RoleManager.get_user_roles(db, user.id)
    role_names = [role.name for role in user_roles]

    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        status=user.status,
        email_verified=user.email_verified,
        last_login=user.last_login,
        profile_image_url=user.profile_image_url,
        created_at=user.created_at,
        updated_at=user.updated_at,
        roles=role_names,
    )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str, admin: User = Depends(require_admin), db: Session = Depends(get_db)
):
    """Permanently delete user and all associated data (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Prevent admin from deleting themselves
    if user.id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account",
        )

    # Log the deletion for audit
    logger.warning(f"Admin {admin.email} deleting user {user.email} (ID: {user.id})")

    # Delete associated sessions
    db.query(UserSession).filter(UserSession.user_id == user_id).delete()

    # Delete the user (cascade will handle other relationships)
    db.delete(user)
    db.commit()

    return {
        "message": "User deleted successfully",
        "deleted_user_id": user_id,
        "deleted_email": user.email,
    }


@router.post("/users/{user_id}/verify-email")
async def verify_user_email(
    user_id: str, admin: User = Depends(require_admin), db: Session = Depends(get_db)
):
    """Manually verify a user's email (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    if user.email_verified:
        return {
            "message": "User email already verified",
            "user_id": user_id,
            "email": user.email,
        }

    user.email_verified = True  # type: ignore
    user.updated_at = datetime.utcnow()  # type: ignore
    db.commit()

    logger.info(f"Admin {admin.email} manually verified email for user {user.email}")

    return {
        "message": "User email verified successfully",
        "user_id": user_id,
        "email": user.email,
    }


@router.post("/users/{user_id}/unverify-email")
async def unverify_user_email(
    user_id: str, admin: User = Depends(require_admin), db: Session = Depends(get_db)
):
    """Manually unverify a user's email (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    if not user.email_verified:
        return {
            "message": "User email already unverified",
            "user_id": user_id,
            "email": user.email,
        }

    user.email_verified = False  # type: ignore
    user.updated_at = datetime.utcnow()  # type: ignore
    db.commit()

    logger.info(f"Admin {admin.email} unverified email for user {user.email}")

    return {
        "message": "User email unverified successfully",
        "user_id": user_id,
        "email": user.email,
    }


@router.post("/users/{user_id}/suspend")
async def suspend_user(
    user_id: str, admin: User = Depends(require_admin), db: Session = Depends(get_db)
):
    """Suspend a user account (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Prevent admin from suspending themselves
    if user.id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot suspend your own account",
        )

    if user.status == "suspended":
        return {
            "message": "User already suspended",
            "user_id": user_id,
            "email": user.email,
        }

    user.status = "suspended"  # type: ignore
    user.updated_at = datetime.utcnow()  # type: ignore

    # Invalidate all active sessions
    db.query(UserSession).filter(
        UserSession.user_id == user_id, UserSession.is_active == True
    ).update({"is_active": False})

    db.commit()

    logger.warning(f"Admin {admin.email} suspended user {user.email}")

    return {
        "message": "User suspended successfully",
        "user_id": user_id,
        "email": user.email,
        "sessions_invalidated": True,
    }


@router.post("/users/{user_id}/activate")
async def activate_user(
    user_id: str, admin: User = Depends(require_admin), db: Session = Depends(get_db)
):
    """Activate a suspended or inactive user account (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    if user.status == "active":
        return {
            "message": "User already active",
            "user_id": user_id,
            "email": user.email,
        }

    previous_status = user.status
    user.status = "active"  # type: ignore
    user.updated_at = datetime.utcnow()  # type: ignore
    db.commit()

    logger.info(
        f"Admin {admin.email} activated user {user.email} (was {previous_status})"
    )

    return {
        "message": "User activated successfully",
        "user_id": user_id,
        "email": user.email,
        "previous_status": previous_status,
    }


@router.get("/users/{user_id}/sessions")
async def get_user_sessions(
    user_id: str, admin: User = Depends(require_admin), db: Session = Depends(get_db)
):
    """Get all sessions for a user (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    sessions = (
        db.query(UserSession)
        .filter(UserSession.user_id == user_id)
        .order_by(UserSession.created_at.desc())
        .all()
    )

    return {
        "user_id": user_id,
        "email": user.email,
        "total_sessions": len(sessions),
        "active_sessions": sum(
            1 for s in sessions if s.is_active and s.expires_at > datetime.utcnow()
        ),
        "sessions": [
            {
                "id": session.id,
                "is_active": session.is_active,
                "created_at": session.created_at,
                "expires_at": session.expires_at,
                "last_activity": session.last_activity,
                "user_agent": session.user_agent,
                "ip_address": session.ip_address,
            }
            for session in sessions
        ],
    }


@router.post("/users/{user_id}/revoke-sessions")
async def revoke_user_sessions(
    user_id: str, admin: User = Depends(require_admin), db: Session = Depends(get_db)
):
    """Revoke all active sessions for a user (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Count active sessions before revocation
    active_count = (
        db.query(UserSession)
        .filter(
            UserSession.user_id == user_id,
            UserSession.is_active == True,
            UserSession.expires_at > datetime.utcnow(),
        )
        .count()
    )

    # Revoke all active sessions
    db.query(UserSession).filter(
        UserSession.user_id == user_id, UserSession.is_active == True
    ).update({"is_active": False})

    db.commit()

    logger.info(
        f"Admin {admin.email} revoked {active_count} sessions for user {user.email}"
    )

    return {
        "message": "All user sessions revoked successfully",
        "user_id": user_id,
        "email": user.email,
        "sessions_revoked": active_count,
    }


@router.get("/stats")
async def get_admin_stats(
    admin: User = Depends(require_admin), db: Session = Depends(get_db)
):
    """Get system statistics (admin only)"""
    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.status == "active").count()
    suspended_users = db.query(User).filter(User.status == "suspended").count()
    verified_users = db.query(User).filter(User.email_verified == True).count()

    active_sessions = (
        db.query(UserSession)
        .filter(
            UserSession.is_active == True, UserSession.expires_at > datetime.utcnow()
        )
        .count()
    )

    # Recent signups (last 30 days)
    from datetime import timedelta

    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    recent_signups = db.query(User).filter(User.created_at >= thirty_days_ago).count()

    return {
        "total_users": total_users,
        "active_users": active_users,
        "suspended_users": suspended_users,
        "inactive_users": total_users - active_users - suspended_users,
        "verified_users": verified_users,
        "unverified_users": total_users - verified_users,
        "verification_rate": round(
            (verified_users / total_users * 100) if total_users > 0 else 0, 2
        ),
        "active_sessions": active_sessions,
        "recent_signups_30d": recent_signups,
    }


@router.post("/users/{user_id}/roles", response_model=RoleResponse)
async def assign_role_to_user(
    user_id: str,
    role_data: UserRoleAssign,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Assign role to user (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    role = db.query(Role).filter(Role.id == role_data.role_id).first()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Role not found"
        )

    # Assign the role
    user_role = RoleManager.assign_role_to_user(
        db, user_id, role_data.role_id, str(admin.id)
    )

    logger.info(f"Admin {admin.email} assigned role {role.name} to user {user.email}")

    return RoleResponse(
        id=role.id,
        name=role.name,
        description=role.description,
        role_type=role.role_type,
        permissions=role.permissions,
        is_system=role.is_system,
        created_at=role.created_at,
    )


@router.delete("/users/{user_id}/roles/{role_id}")
async def remove_role_from_user(
    user_id: str,
    role_id: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Remove role from user (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Role not found"
        )

    success = RoleManager.remove_role_from_user(db, user_id, role_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Role assignment not found"
        )

    logger.info(f"Admin {admin.email} removed role {role.name} from user {user.email}")

    return {
        "message": "Role removed successfully",
        "user_id": user_id,
        "role_id": role_id,
    }


# Role Management
@router.get("/roles", response_model=List[RoleResponse])
async def list_roles(
    role_type: Optional[str] = Query(None),
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """List all roles (admin only)"""
    roles = RoleManager.list_roles(db, role_type)
    return [
        RoleResponse(
            id=role.id,
            name=role.name,
            description=role.description,
            role_type=role.role_type,
            permissions=role.permissions,
            is_system=role.is_system,
            created_at=role.created_at,
        )
        for role in roles
    ]


@router.post("/roles", response_model=RoleResponse, status_code=status.HTTP_201_CREATED)
async def create_role(
    role_data: RoleCreate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Create a new role (admin only)"""
    try:
        role = RoleManager.create_role(
            db=db,
            name=role_data.name,
            description=role_data.description,
            role_type=role_data.role_type,
            permissions=role_data.permissions or [],
        )

        return RoleResponse(
            id=role.id,
            name=role.name,
            description=role.description,
            role_type=role.role_type,
            permissions=role.permissions,
            is_system=role.is_system,
            created_at=role.created_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
