"""
User Role Manager
Specific user role management
"""

from typing import List
from sqlalchemy.orm import Session

from ..models.auth_models import User, Role, UserRole
from .role_manager import RoleManager


class UserRoleManager:
    """Specific user role management"""

    @staticmethod
    def assign_admin_role(db: Session, user: User, assigned_by: str) -> bool:
        """Assign admin role to user"""
        admin_role = RoleManager.get_role_by_name(db, "admin")
        if not admin_role:
            # Create default roles if they don't exist
            RoleManager.get_or_create_default_roles(db)
            admin_role = RoleManager.get_role_by_name(db, "admin")

        if admin_role:
            RoleManager.assign_role_to_user(db, user.id, admin_role.id, assigned_by)
            return True
        return False

    @staticmethod
    def revoke_admin_role(db: Session, user: User) -> bool:
        """Revoke admin role from user"""
        admin_role = RoleManager.get_role_by_name(db, "admin")
        if admin_role:
            return RoleManager.remove_role_from_user(db, user.id, admin_role.id)
        return False

    @staticmethod
    def is_admin(db: Session, user: User) -> bool:
        """Check if user is admin"""
        admin_role = RoleManager.get_role_by_name(db, "admin")
        if not admin_role:
            return False

        user_role = (
            db.query(UserRole)
            .filter(UserRole.user_id == user.id, UserRole.role_id == admin_role.id)
            .first()
        )

        return user_role is not None

    @staticmethod
    def get_user_role_names(db: Session, user: User) -> List[str]:
        """Get list of role names for user"""
        roles = RoleManager.get_user_roles(db, user.id)
        return [role.name for role in roles]
