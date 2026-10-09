"""
Session Management Router
List active sessions, revoke sessions, concurrent session limits
"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import List

from ..dependencies import get_db, get_current_user
from ..models.auth_models import User, UserSession
from ..models.pydantic_models import SessionResponse
from ..providers.session_provider import SessionProvider
from ..config import config

router = APIRouter(prefix="/api/v1/auth/sessions", tags=["Sessions"])

MAX_CONCURRENT_SESSIONS = 5


@router.get("/", response_model=List[SessionResponse])
async def list_sessions(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all active sessions for the current user"""
    current_token = request.headers.get("X-Session-Token") or request.cookies.get(
        "session_token"
    )
    current_hash = SessionProvider.hash_token(current_token) if current_token else None

    sessions = (
        db.query(UserSession)
        .filter(
            UserSession.user_id == user.id,
            UserSession.is_active == True,
            UserSession.expires_at > datetime.utcnow(),
        )
        .order_by(UserSession.last_activity.desc())
        .all()
    )

    return [
        SessionResponse(
            id=s.id,
            created_at=s.created_at,
            last_activity=s.last_activity,
            expires_at=s.expires_at,
            ip_address=s.ip_address,
            user_agent=s.user_agent,
            is_current=(current_hash is not None and s.session_token == current_hash),
        )
        for s in sessions
    ]


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_session(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Revoke a specific session"""
    session = (
        db.query(UserSession)
        .filter(
            UserSession.id == session_id,
            UserSession.user_id == user.id,
        )
        .first()
    )

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.is_active = False
    db.commit()
    logger = __import__("logging").getLogger(__name__)
    logger.info(f"Session {session_id} revoked for user {user.id}")


@router.delete("/", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_all_sessions(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Revoke all sessions (except current one)"""
    current_token = request.headers.get("X-Session-Token") or request.cookies.get(
        "session_token"
    )
    current_hash = SessionProvider.hash_token(current_token) if current_token else None

    sessions = (
        db.query(UserSession)
        .filter(
            UserSession.user_id == user.id,
            UserSession.is_active == True,
        )
        .all()
    )

    revoked = 0
    for s in sessions:
        if current_hash and s.session_token == current_hash:
            continue
        s.is_active = False
        revoked += 1

    db.commit()
    logger = __import__("logging").getLogger(__name__)
    logger.info(f"Revoked {revoked} sessions for user {user.id} (kept current)")


def enforce_concurrent_sessions(
    db: Session, user_id: str, new_session_token: str
) -> None:
    """Enforce maximum concurrent sessions per user.
    Call this after creating a new session. Revokes oldest if over limit.
    """
    active_sessions = (
        db.query(UserSession)
        .filter(
            UserSession.user_id == user_id,
            UserSession.is_active == True,
            UserSession.expires_at > datetime.utcnow(),
        )
        .order_by(UserSession.last_activity.asc())
        .all()
    )

    if len(active_sessions) > MAX_CONCURRENT_SESSIONS:
        excess = len(active_sessions) - MAX_CONCURRENT_SESSIONS
        for s in active_sessions[:excess]:
            if s.session_token != new_session_token:
                s.is_active = False
                logger = __import__("logging").getLogger(__name__)
                logger.info(f"Revoked excess session {s.id} for user {user_id}")
        db.commit()
