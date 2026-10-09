"""
Role Manager
Manages roles and role assignments
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..models.auth_models import Role, UserRole, IntegrationRole, User, Integration
from ..config import config


class RoleManager:
    """Manages roles and role assignments"""

    @staticmethod
    def get_or_create_default_roles(db: Session) -> List[Role]:
        """Get or create default system roles"""
        roles = []

        for role_name, role_data in config.DEFAULT_ROLES.items():
            role = db.query(Role).filter(Role.name == role_name).first()

            if not role:
                role = Role(
                    name=role_data["name"],
                    description=role_data["description"],
                    role_type=role_data["role_type"],
                    permissions=role_data["permissions"],
                    is_system=role_data["is_system"],
                )
                db.add(role)
                db.commit()
                db.refresh(role)

            roles.append(role)

        return roles

    @staticmethod
    def create_role(
        db: Session,
        name: str,
        description: Optional[str] = None,
        role_type: str = "user",
        permissions: Optional[List[str]] = None,
    ) -> Role:
        """Create a new role"""
        # Check if role already exists
        existing = db.query(Role).filter(Role.name == name).first()
        if existing:
            raise ValueError(f"Role '{name}' already exists")

        role = Role(
            name=name,
            description=description,
            role_type=role_type,
            permissions=permissions or [],
        )

        db.add(role)
        db.commit()
        db.refresh(role)

        return role

    @staticmethod
    def get_role(db: Session, role_id: str) -> Optional[Role]:
        """Get role by ID"""
        return db.query(Role).filter(Role.id == role_id).first()

    @staticmethod
    def get_role_by_name(db: Session, name: str) -> Optional[Role]:
        """Get role by name"""
        return db.query(Role).filter(Role.name == name).first()

    @staticmethod
    def list_roles(db: Session, role_type: Optional[str] = None) -> List[Role]:
        """List all roles, optionally filtered by type"""
        query = db.query(Role)
        if role_type:
            query = query.filter(Role.role_type == role_type)
        return query.all()

    @staticmethod
    def assign_role_to_user(
        db: Session, user_id: str, role_id: str, assigned_by: Optional[str] = None
    ) -> UserRole:
        """Assign role to user"""
        # Check if already assigned
        existing = (
            db.query(UserRole)
            .filter(and_(UserRole.user_id == user_id, UserRole.role_id == role_id))
            .first()
        )

        if existing:
            return existing

        user_role = UserRole(user_id=user_id, role_id=role_id, assigned_by=assigned_by)

        db.add(user_role)
        db.commit()
        db.refresh(user_role)

        return user_role

    @staticmethod
    def remove_role_from_user(db: Session, user_id: str, role_id: str) -> bool:
        """Remove role from user"""
        user_role = (
            db.query(UserRole)
            .filter(and_(UserRole.user_id == user_id, UserRole.role_id == role_id))
            .first()
        )

        if user_role:
            db.delete(user_role)
            db.commit()
            return True

        return False

    @staticmethod
    def assign_role_to_integration(
        db: Session,
        integration_id: str,
        role_id: str,
        assigned_by: Optional[str] = None,
    ) -> IntegrationRole:
        """Assign role to integration"""
        # Check if already assigned
        existing = (
            db.query(IntegrationRole)
            .filter(
                and_(
                    IntegrationRole.integration_id == integration_id,
                    IntegrationRole.role_id == role_id,
                )
            )
            .first()
        )

        if existing:
            return existing

        integration_role = IntegrationRole(
            integration_id=integration_id, role_id=role_id, assigned_by=assigned_by
        )

        db.add(integration_role)
        db.commit()
        db.refresh(integration_role)

        return integration_role

    @staticmethod
    def remove_role_from_integration(
        db: Session, integration_id: str, role_id: str
    ) -> bool:
        """Remove role from integration"""
        integration_role = (
            db.query(IntegrationRole)
            .filter(
                and_(
                    IntegrationRole.integration_id == integration_id,
                    IntegrationRole.role_id == role_id,
                )
            )
            .first()
        )

        if integration_role:
            db.delete(integration_role)
            db.commit()
            return True

        return False

    @staticmethod
    def get_user_roles(db: Session, user_id: str) -> List[Role]:
        """Get all roles for a user"""
        user_roles = db.query(UserRole).filter(UserRole.user_id == user_id).all()
        role_ids = [ur.role_id for ur in user_roles]
        return db.query(Role).filter(Role.id.in_(role_ids)).all() if role_ids else []

    @staticmethod
    def get_integration_roles(db: Session, integration_id: str) -> List[Role]:
        """Get all roles for an integration"""
        integration_roles = (
            db.query(IntegrationRole)
            .filter(IntegrationRole.integration_id == integration_id)
            .all()
        )
        role_ids = [ir.role_id for ir in integration_roles]
        return db.query(Role).filter(Role.id.in_(role_ids)).all() if role_ids else []
