"""
Permission Checker
Checks if user or integration has permission for resource/action
"""

from typing import List, Optional
from sqlalchemy.orm import Session

from ..models.auth_models import User, Integration, Role, UserRole, IntegrationRole
from .role_manager import RoleManager


class PermissionChecker:
    """Checks permissions for users and integrations"""

    @staticmethod
    def get_user_permissions(db: Session, user: User) -> List[str]:
        """Get all permissions for a user"""
        permissions = set()

        # Get permissions from user roles
        user_roles = db.query(UserRole).filter(UserRole.user_id == user.id).all()
        for user_role in user_roles:
            role = db.query(Role).filter(Role.id == user_role.role_id).first()
            if role and role.permissions:
                permissions.update(role.permissions)

        return list(permissions)

    @staticmethod
    def get_integration_permissions(db: Session, integration: Integration) -> List[str]:
        """Get all permissions for an integration"""
        permissions = set()

        # Get permissions from integration roles
        integration_roles = (
            db.query(IntegrationRole)
            .filter(IntegrationRole.integration_id == integration.id)
            .all()
        )

        for integration_role in integration_roles:
            role = db.query(Role).filter(Role.id == integration_role.role_id).first()
            if role and role.permissions:
                permissions.update(role.permissions)

        return list(permissions)

    @staticmethod
    def has_permission(
        db: Session,
        permission: str,
        *,
        user: Optional[User] = None,
        integration: Optional[Integration] = None,
    ) -> bool:
        """Check if user or integration has a specific permission"""
        if user:
            permissions = PermissionChecker.get_user_permissions(db, user)
        elif integration:
            permissions = PermissionChecker.get_integration_permissions(db, integration)
        else:
            return False

        # Check for exact permission or wildcard
        if "*" in permissions or permission in permissions:
            return True

        # Check for resource wildcard (e.g., "query:*" matches "query:read")
        permission_parts = permission.split(":")
        if len(permission_parts) == 2:
            resource, action = permission_parts
            resource_wildcard = f"{resource}:*"
            if resource_wildcard in permissions:
                return True

        return False

    @staticmethod
    def check_resource_action(
        db: Session,
        resource: str,
        action: str,
        *,
        user: Optional[User] = None,
        integration: Optional[Integration] = None,
    ) -> bool:
        """Check if user/integration can perform action on resource"""
        permission = f"{resource}:{action}"
        return PermissionChecker.has_permission(
            db, permission, user=user, integration=integration
        )

    @staticmethod
    def require_permission(
        db: Session,
        permission: str,
        *,
        user: Optional[User] = None,
        integration: Optional[Integration] = None,
    ) -> bool:
        """Require permission or raise exception"""
        if not PermissionChecker.has_permission(
            db, permission, user=user, integration=integration
        ):
            raise PermissionError(f"Permission required: {permission}")
        return True
