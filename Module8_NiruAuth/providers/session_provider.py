"""
Session Provider
Handles user session management
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..models.auth_models import UserSession, User
from ..config import config


class SessionCache:
    """In-memory cache for validated sessions"""

    def __init__(self, ttl_seconds: int = 300):  # 5 minutes default
        self._cache: Dict[str, Tuple[UserSession, datetime]] = {}
        self._ttl_seconds = ttl_seconds

    def get(self, token_hash: str) -> Optional[UserSession]:
        """Get cached session if valid"""
        if token_hash in self._cache:
            session, cached_at = self._cache[token_hash]
            # Check if cache entry is still valid
            if (datetime.utcnow() - cached_at).total_seconds() < self._ttl_seconds:
                return session
            # Cache expired, remove it
            del self._cache[token_hash]
        return None

    def set(self, token_hash: str, session: UserSession):
        """Cache a validated session"""
        self._cache[token_hash] = (session, datetime.utcnow())

    def invalidate(self, token_hash: str):
        """Remove a session from cache"""
        if token_hash in self._cache:
            del self._cache[token_hash]

    def clear(self):
        """Clear all cached sessions"""
        self._cache.clear()


class SessionProvider:
    """Handles user session operations"""

    # Class-level cache instance
    _session_cache = SessionCache(ttl_seconds=300)  # 5 minutes

    @staticmethod
    def hash_token(token: str) -> str:
        """Hash session token for storage"""
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def generate_session_token() -> str:
        """Generate a secure session token"""
        return secrets.token_urlsafe(config.SESSION_TOKEN_LENGTH)

    @staticmethod
    def create_session(
        db: Session,
        user: User,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Tuple[str, UserSession]:
        """Create a new user session"""
        import logging

        logger = logging.getLogger(__name__)

        try:
            # Generate tokens
            session_token = SessionProvider.generate_session_token()
            refresh_token = SessionProvider.generate_session_token()
            token_hash = SessionProvider.hash_token(session_token)

            # Calculate expiration
            expires_at = datetime.utcnow() + timedelta(
                hours=config.SESSION_EXPIRE_HOURS
            )

            logger.info(f"Creating session for user {user.id}, expires_at={expires_at}")

            # Create session
            session = UserSession(
                user_id=user.id,
                session_token=token_hash,
                refresh_token=SessionProvider.hash_token(refresh_token),
                expires_at=expires_at,
                ip_address=ip_address,
                user_agent=user_agent,
            )

            # Add session to database
            db.add(session)

            # Flush to get the session ID before commit
            db.flush()

            # Commit the transaction
            db.commit()

            # Refresh to ensure we have the latest data from database
            db.refresh(session)

            # Verify session was saved by querying it back
            saved_session = (
                db.query(UserSession)
                .filter(UserSession.session_token == token_hash)
                .first()
            )

            if not saved_session:
                logger.error(
                    f"Session was not saved to database! user_id={user.id}, token_hash={token_hash[:16]}..."
                )
                db.rollback()
                raise Exception("Failed to save session to database")

            logger.info(
                f"Session created successfully: session_id={session.id}, user_id={user.id}, expires_at={session.expires_at}"
            )

            return session_token, session

        except Exception as e:
            logger.error(
                f"Error creating session for user {user.id}: {e}", exc_info=True
            )
            db.rollback()
            raise

    @staticmethod
    def validate_session(db: Session, session_token: str) -> Optional[UserSession]:
        """Validate session token and return session"""
        from datetime import datetime
        from sqlalchemy import and_
        import logging

        logger = logging.getLogger(__name__)
        token_hash = SessionProvider.hash_token(session_token)
        now = datetime.utcnow()

        # Check cache first
        try:
            cached_session = SessionProvider._session_cache.get(token_hash)
            if cached_session:
                # Attach cached session instance to current session to avoid DetachedInstanceError
                cached_session = db.merge(cached_session)
                # Refresh the merged session to ensure it's up to date
                db.refresh(cached_session)
                logger.debug(f"Session found in cache: session_id={cached_session.id}")
                # Still check if session is expired (quick in-memory check)
                if cached_session.expires_at:
                    expires_at = cached_session.expires_at
                    # Handle timezone-aware datetimes
                    if hasattr(expires_at, "tzinfo") and expires_at.tzinfo is not None:
                        from datetime import timezone as tz

                        try:
                            expires_at = expires_at.astimezone(tz.utc).replace(
                                tzinfo=None
                            )
                        except Exception:
                            expires_at = expires_at.replace(tzinfo=None)

                    if hasattr(now, "tzinfo") and now.tzinfo is not None:
                        from datetime import timezone as tz

                        now = now.astimezone(tz.utc).replace(tzinfo=None)

                    try:
                        if expires_at <= now:
                            logger.debug(
                                f"Cached session expired, invalidating cache and checking DB"
                            )
                            SessionProvider._session_cache.invalidate(token_hash)
                            # Do not return None here, fall through to DB check
                            # return None
                    except TypeError:
                        SessionProvider._session_cache.invalidate(token_hash)
                        # return None

                if not cached_session.is_active:
                    logger.debug(f"Cached session not active, invalidating cache")
                    SessionProvider._session_cache.invalidate(token_hash)
                    # return None

                # Only return if we are sure it's valid and active
                if cached_session.is_active and (
                    not cached_session.expires_at or expires_at > now
                ):
                    # Return cached session (skip DB query and last_activity update)
                    return cached_session
        except Exception as e:
            logger.warning(
                f"Error retrieving session from cache: {e}. Falling back to DB."
            )
            # Remove from cache if it caused an error
            SessionProvider._session_cache.invalidate(token_hash)

        logger.debug(f"Validating session from DB: token_hash={token_hash[:16]}...")

        try:
            # Refresh database session to ensure we see latest data
            db.expire_all()

            # First check if session exists with matching token hash
            session = (
                db.query(UserSession)
                .filter(UserSession.session_token == token_hash)
                .first()
            )

            if not session:
                logger.warning(
                    f"Session not found for token hash: {token_hash[:16]}..."
                )
                # Try one more time with explicit refresh
                try:
                    db.commit()  # Commit any pending transactions
                except:
                    pass
                db.expire_all()
                session = (
                    db.query(UserSession)
                    .filter(UserSession.session_token == token_hash)
                    .first()
                )
                if not session:
                    logger.warning(f"Session still not found after refresh")
                    return None

            logger.debug(
                f"Session found: id={session.id}, is_active={session.is_active}, user_id={session.user_id}"
            )

            # Check if session is active
            if not session.is_active:
                logger.warning(f"Session found but not active: session_id={session.id}")
                return None

            # Check if session is expired
            if session.expires_at:
                # Robust timezone handling
                expires_at = session.expires_at

                # Ensure expires_at is timezone-naive UTC
                if hasattr(expires_at, "tzinfo") and expires_at.tzinfo is not None:
                    from datetime import timezone as tz

                    expires_at = expires_at.astimezone(tz.utc).replace(tzinfo=None)

                # Ensure now is timezone-naive UTC
                # datetime.utcnow() returns naive UTC, but let's be safe
                if hasattr(now, "tzinfo") and now.tzinfo is not None:
                    from datetime import timezone as tz

                    now = now.astimezone(tz.utc).replace(tzinfo=None)

                is_expired = expires_at <= now

                if is_expired:
                    logger.warning(
                        f"Session expired: expires_at={expires_at} (UTC naive), now={now} (UTC naive)"
                    )
                    return None
            else:
                logger.warning(f"Session has no expires_at: session_id={session.id}")
                return None

            # Session is valid - update last activity and extend expiration (sliding window)
            try:
                session.last_activity = now
                # Extend expiration to keep session alive for active users
                session.expires_at = now + timedelta(hours=config.SESSION_EXPIRE_HOURS)
                db.commit()
                logger.debug(
                    f"Session validated and extended: session_id={session.id}, user_id={session.user_id}"
                )
            except Exception as e:
                logger.error(
                    f"Error updating session last_activity/expires_at: {e}",
                    exc_info=True,
                )
                db.rollback()
                # Still return the session even if update fails
                logger.debug(
                    f"Returning session despite update failure: session_id={session.id}"
                )

            # Cache the validated session
            SessionProvider._session_cache.set(token_hash, session)
            logger.debug(f"Session cached for future requests: session_id={session.id}")

            return session
        except Exception as e:
            logger.error(f"Error validating session: {e}", exc_info=True)
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            try:
                db.rollback()
            except:
                pass
            return None

    @staticmethod
    def refresh_session(
        db: Session, refresh_token: str
    ) -> Optional[Tuple[str, UserSession]]:
        """Refresh session using refresh token"""
        token_hash = SessionProvider.hash_token(refresh_token)

        session = (
            db.query(UserSession)
            .filter(
                and_(
                    UserSession.refresh_token == token_hash,
                    UserSession.is_active == True,
                    UserSession.expires_at > datetime.utcnow(),
                )
            )
            .first()
        )

        if not session:
            return None

        # Check if refresh is needed (within threshold)
        refresh_threshold = datetime.utcnow() + timedelta(
            hours=config.SESSION_REFRESH_THRESHOLD_HOURS
        )

        if session.expires_at > refresh_threshold:
            # Still valid, just return existing
            return None, session

        # Generate new session token
        new_session_token = SessionProvider.generate_session_token()
        new_expires_at = datetime.utcnow() + timedelta(
            hours=config.SESSION_EXPIRE_HOURS
        )

        # Update session
        session.session_token = SessionProvider.hash_token(new_session_token)
        session.expires_at = new_expires_at
        session.last_activity = datetime.utcnow()

        db.commit()
        db.refresh(session)

        return new_session_token, session

    @staticmethod
    def revoke_session(db: Session, session_token: str) -> bool:
        """Revoke a session"""
        token_hash = SessionProvider.hash_token(session_token)

        session = (
            db.query(UserSession)
            .filter(UserSession.session_token == token_hash)
            .first()
        )

        if session:
            session.is_active = False
            db.commit()
            # Invalidate cache
            SessionProvider._session_cache.invalidate(token_hash)
            return True

        return False

    @staticmethod
    def revoke_all_user_sessions(db: Session, user_id: str) -> int:
        """Revoke all sessions for a user"""
        sessions = (
            db.query(UserSession)
            .filter(and_(UserSession.user_id == user_id, UserSession.is_active == True))
            .all()
        )

        count = len(sessions)
        for session in sessions:
            session.is_active = False
            # Invalidate cache for each session
            SessionProvider._session_cache.invalidate(session.session_token)

        db.commit()
        return count

    @staticmethod
    def cleanup_expired_sessions(db: Session) -> int:
        """Clean up expired sessions"""
        expired = (
            db.query(UserSession)
            .filter(UserSession.expires_at < datetime.utcnow())
            .all()
        )

        count = len(expired)
        for session in expired:
            session.is_active = False
            # Invalidate cache for expired sessions
            SessionProvider._session_cache.invalidate(session.session_token)

        db.commit()
        return count
