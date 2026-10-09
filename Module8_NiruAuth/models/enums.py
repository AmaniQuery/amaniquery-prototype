"""
Enums for authentication and authorization
"""

from enum import Enum


class AuthMethod(str, Enum):
    """Authentication methods"""

    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    SESSION = "session"
    SOCIAL = "social"


class RoleType(str, Enum):
    """Role types"""

    USER = "user"
    INTEGRATION = "integration"


class UserStatus(str, Enum):
    """User account status"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"


class IntegrationStatus(str, Enum):
    """Integration status"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_APPROVAL = "pending_approval"


class PermissionResource(str, Enum):
    """Permission resources"""

    QUERY = "query"
    RESEARCH = "research"
    ADMIN = "admin"
    USER = "user"
    INTEGRATION = "integration"
    CONFIG = "config"
    REPORTS = "reports"
    CHAT = "chat"
    SMS = "sms"
    SHARE = "share"
    VOICE = "voice"


class PermissionAction(str, Enum):
    """Permission actions"""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    MANAGE = "manage"
    EXECUTE = "execute"
