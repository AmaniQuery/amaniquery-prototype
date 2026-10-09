"""
Voice Session Manager
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from loguru import logger
import time


@dataclass
class VoiceSession:
    """Represents a voice conversation session"""

    session_id: str
    created_at: datetime
    last_activity: datetime
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    context: Dict = field(default_factory=dict)

    def add_turn(self, user_query: str, agent_response: str):
        """Add a conversation turn"""
        self.conversation_history.append(
            {
                "user": user_query,
                "agent": agent_response,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        self.last_activity = datetime.utcnow()

    def get_recent_context(self, max_turns: int = 3) -> List[str]:
        """Get recent conversation context"""
        recent = (
            self.conversation_history[-max_turns:]
            if len(self.conversation_history) > max_turns
            else self.conversation_history
        )
        return [turn["user"] + " " + turn["agent"] for turn in recent]

    def is_expired(self, timeout_seconds: int) -> bool:
        """Check if session has expired"""
        if not self.last_activity:
            return False

        elapsed = (datetime.utcnow() - self.last_activity).total_seconds()
        return elapsed > timeout_seconds


class VoiceSessionManager:
    """Manages voice conversation sessions"""

    def __init__(self, default_timeout: int = 300):
        """
        Initialize session manager

        Args:
            default_timeout: Default session timeout in seconds
        """
        self.sessions: Dict[str, VoiceSession] = {}
        self.default_timeout = default_timeout
        logger.info("Voice session manager initialized")

    def create_session(self, session_id: str) -> VoiceSession:
        """Create a new voice session"""
        if session_id in self.sessions:
            logger.warning(f"Session {session_id} already exists, reusing")
            return self.sessions[session_id]

        session = VoiceSession(
            session_id=session_id,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
        )

        self.sessions[session_id] = session
        logger.info(f"Created new voice session: {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[VoiceSession]:
        """Get an existing session"""
        session = self.sessions.get(session_id)

        if session and session.is_expired(self.default_timeout):
            logger.info(f"Session {session_id} expired, removing")
            del self.sessions[session_id]
            return None

        return session

    def get_or_create_session(self, session_id: str) -> VoiceSession:
        """Get existing session or create new one"""
        session = self.get_session(session_id)
        if session:
            return session
        return self.create_session(session_id)

    def add_conversation_turn(
        self, session_id: str, user_query: str, agent_response: str
    ):
        """Add a conversation turn to a session"""
        session = self.get_or_create_session(session_id)
        session.add_turn(user_query, agent_response)

    def get_conversation_context(
        self, session_id: str, max_turns: int = 3
    ) -> List[str]:
        """Get conversation context for a session"""
        session = self.get_session(session_id)
        if not session:
            return []
        return session.get_recent_context(max_turns)

    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        expired_ids = [
            sid
            for sid, session in self.sessions.items()
            if session.is_expired(self.default_timeout)
        ]

        for sid in expired_ids:
            logger.info(f"Cleaning up expired session: {sid}")
            del self.sessions[sid]

        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired sessions")

    def delete_session(self, session_id: str):
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
