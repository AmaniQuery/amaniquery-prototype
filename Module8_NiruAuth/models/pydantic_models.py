"""
Pydantic Models for Authentication API Requests and Responses
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, Field, validator
from .enums import (
    AuthMethod,
    RoleType,
    UserStatus,
    IntegrationStatus,
    PermissionResource,
    PermissionAction,
)

# ==================== User Models ====================


class UserRegister(BaseModel):
    """User registration request"""

    email: EmailStr
    password: str = Field(
        ..., min_length=8, description="Password must be at least 8 characters"
    )
    name: Optional[str] = None
    phone_number: str = Field(
        ..., description="Phone number in format +254712345678 or 0712345678"
    )

    @validator("password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserLogin(BaseModel):
    """User login request"""

    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User response model"""

    id: str
    email: str
    name: Optional[str]
    status: str
    email_verified: bool
    last_login: Optional[datetime]
    profile_image_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    roles: Optional[List[str]] = None  # Add roles field

    class Config:
        from_attributes = True


class UserProfileUpdate(BaseModel):
    """User profile update request"""

    name: Optional[str] = None
    email: Optional[EmailStr] = None
    profile_image_url: Optional[str] = None


class PasswordChange(BaseModel):
    """Password change request"""

    current_password: str
    new_password: str = Field(..., min_length=8)

    @validator("new_password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class PasswordResetRequest(BaseModel):
    """Password reset request"""

    email: EmailStr


class PasswordResetRequestResponse(BaseModel):
    """Password reset request response"""

    message: str
    phone_number: Optional[str] = Field(
        None, description="Masked phone number for OTP verification"
    )


class PasswordReset(BaseModel):
    """Password reset with token"""

    token: str
    new_password: str = Field(..., min_length=8)

    @validator("new_password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class EmailVerificationRequest(BaseModel):
    """Email verification request"""

    token: str


class SessionResponse(BaseModel):
    """Session response"""

    session_token: str
    refresh_token: Optional[str] = None
    expires_at: datetime
    user: UserResponse  # UserResponse now includes roles field


class MFARequiredResponse(BaseModel):
    """Response when MFA challenge is required"""

    mfa_required: bool = True
    mfa_token: str
    message: str = (
        "MFA verification required. Submit TOTP code to /api/v1/auth/mfa/challenge"
    )


class MFAChallengeRequest(BaseModel):
    """MFA challenge completion request"""

    mfa_token: str
    totp_code: str = Field(
        ...,
        min_length=6,
        max_length=6,
        description="6-digit TOTP code from authenticator app",
    )


# ==================== Role Models ====================


class RoleResponse(BaseModel):
    """Role response model"""

    id: str
    name: str
    description: Optional[str]
    role_type: str
    permissions: Optional[List[str]]
    is_system: bool
    created_at: datetime

    class Config:
        from_attributes = True


class RoleCreate(BaseModel):
    """Role creation request"""

    name: str
    description: Optional[str] = None
    role_type: str
    permissions: Optional[List[str]] = []


class RoleUpdate(BaseModel):
    """Role update request"""

    description: Optional[str] = None
    permissions: Optional[List[str]] = None


class UserRoleAssign(BaseModel):
    """Assign role to user"""

    role_id: str


# ==================== Integration Models ====================


class IntegrationCreate(BaseModel):
    """Integration creation request"""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    type: Optional[str] = None
    webhook_url: Optional[str] = None
    ip_whitelist: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class IntegrationUpdate(BaseModel):
    """Integration update request"""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    status: Optional[str] = None
    webhook_url: Optional[str] = None
    ip_whitelist: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class IntegrationResponse(BaseModel):
    """Integration response model"""

    id: str
    name: str
    description: Optional[str]
    type: Optional[str]
    owner_user_id: str
    status: str
    webhook_url: Optional[str]
    ip_whitelist: Optional[List[str]]
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    roles: Optional[List[RoleResponse]] = []

    class Config:
        from_attributes = True


# ==================== API Key Models ====================


class APIKeyCreate(BaseModel):
    """API key creation request"""

    name: Optional[str] = None
    scopes: Optional[List[str]] = []
    rate_limit_per_minute: Optional[int] = Field(60, ge=1, le=10000)
    rate_limit_per_hour: Optional[int] = Field(1000, ge=1, le=100000)
    rate_limit_per_day: Optional[int] = Field(10000, ge=1, le=1000000)
    expires_at: Optional[datetime] = None


class APIKeyResponse(BaseModel):
    """API key response (includes the key only on creation)"""

    id: str
    key: Optional[str] = None  # Only included on creation
    key_prefix: str
    name: Optional[str]
    scopes: Optional[List[str]]
    rate_limit_per_minute: int
    rate_limit_per_hour: int
    rate_limit_per_day: int
    is_active: bool
    last_used: Optional[datetime]
    expires_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class APIKeyListResponse(BaseModel):
    """List of API keys (without actual keys)"""

    id: str
    key_prefix: str
    name: Optional[str]
    scopes: Optional[List[str]]
    is_active: bool
    last_used: Optional[datetime]
    expires_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


# ==================== OAuth Models ====================


class OAuthClientCreate(BaseModel):
    """OAuth client creation request"""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    redirect_uris: Optional[List[str]] = []
    grant_types: Optional[List[str]] = ["authorization_code", "refresh_token"]
    scopes: Optional[List[str]] = []


class OAuthClientResponse(BaseModel):
    """OAuth client response (includes secret only on creation)"""

    id: str
    client_id: str
    client_secret: Optional[str] = None  # Only included on creation
    name: str
    description: Optional[str]
    redirect_uris: Optional[List[str]]
    grant_types: Optional[List[str]]
    scopes: Optional[List[str]]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class OAuthTokenRequest(BaseModel):
    """OAuth token request"""

    grant_type: str
    client_id: str
    client_secret: Optional[str] = None
    code: Optional[str] = None  # For authorization code flow
    redirect_uri: Optional[str] = None
    refresh_token: Optional[str] = None  # For refresh token flow
    scope: Optional[str] = None


class OAuthTokenResponse(BaseModel):
    """OAuth token response"""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int  # Seconds until expiration
    refresh_token: Optional[str] = None
    scope: Optional[str] = None


class OAuthAuthorizeRequest(BaseModel):
    """OAuth authorization request"""

    client_id: str
    redirect_uri: str
    response_type: str = "code"
    scope: Optional[str] = None
    state: Optional[str] = None


# ==================== Usage & Analytics Models ====================


class UsageStats(BaseModel):
    """Usage statistics"""

    total_requests: int
    total_tokens: int
    total_cost: float
    avg_response_time_ms: float
    requests_today: int
    requests_this_week: int
    requests_this_month: int
    top_endpoints: List[Dict[str, Any]]


class UsageLogResponse(BaseModel):
    """Usage log entry"""

    id: str
    endpoint: str
    method: str
    status_code: int
    tokens_used: int
    cost: float
    response_time_ms: Optional[float]
    timestamp: datetime

    class Config:
        from_attributes = True


class AnalyticsDashboard(BaseModel):
    """Analytics dashboard data"""

    total_users: int
    total_integrations: int
    total_api_keys: int
    total_requests_today: int
    total_requests_this_week: int
    total_requests_this_month: int
    total_cost_today: float
    total_cost_this_week: float
    total_cost_this_month: float
    top_users: List[Dict[str, Any]]
    top_integrations: List[Dict[str, Any]]
    top_endpoints: List[Dict[str, Any]]
    requests_over_time: List[Dict[str, Any]]


# ==================== Admin Models ====================


class UserListResponse(BaseModel):
    """User list response"""

    users: List[UserResponse]
    total: int
    page: int
    page_size: int


class UserUpdate(BaseModel):
    """Admin user update request"""

    name: Optional[str] = None
    email: Optional[EmailStr] = None
    status: Optional[str] = None
    email_verified: Optional[bool] = None


# ==================== Auth Context Models ====================


class AuthContext(BaseModel):
    """Authentication context attached to requests"""

    auth_method: str
    user_id: Optional[str] = None
    integration_id: Optional[str] = None
    api_key_id: Optional[str] = None
    roles: List[str] = []
    permissions: List[str] = []
    scopes: Optional[List[str]] = None
    user: Optional[Any] = None  # Cache the user object to avoid re-querying DB
    integration: Optional[Any] = (
        None  # Cache the integration object to avoid re-querying DB
    )

    class Config:
        arbitrary_types_allowed = True


# ==================== Error Models ====================


class ErrorResponse(BaseModel):
    """Error response"""

    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class ValidationErrorResponse(BaseModel):
    """Validation error response"""

    error: str = "Validation Error"
    details: List[Dict[str, Any]]
