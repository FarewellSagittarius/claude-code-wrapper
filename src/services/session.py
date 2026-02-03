"""Session management for conversation continuity."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, List, Optional

from ..models.openai import Message

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Represents a conversation session."""

    session_id: str
    messages: List[Message] = field(default_factory=list)
    claude_session_id: Optional[str] = None  # Claude SDK session ID for resumption
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(
        default_factory=lambda: datetime.utcnow() + timedelta(hours=1)
    )

    def touch(self) -> None:
        """Update access time and extend expiration."""
        self.last_accessed = datetime.utcnow()
        self.expires_at = datetime.utcnow() + timedelta(hours=1)

    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.utcnow() > self.expires_at

    def add_message(self, message: Message) -> None:
        """Add a message to the session."""
        self.messages.append(message)
        self.touch()

    def get_message_count(self) -> int:
        """Get total message count."""
        return len(self.messages)


class SessionManager:
    """Manages conversation sessions with automatic cleanup."""

    def __init__(
        self,
        ttl_hours: int = 1,
        cleanup_interval_seconds: int = 300,
    ):
        """
        Initialize session manager.

        Args:
            ttl_hours: Time-to-live for sessions in hours
            cleanup_interval_seconds: Interval between cleanup runs
        """
        self.sessions: Dict[str, Session] = {}
        self.lock = Lock()
        self.ttl_hours = ttl_hours
        self.cleanup_interval = cleanup_interval_seconds
        self._cleanup_task: Optional[asyncio.Task] = None

    def get_or_create(self, session_id: str) -> Session:
        """
        Get existing session or create new one.

        Args:
            session_id: Session identifier

        Returns:
            Session object
        """
        with self.lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                if session.is_expired():
                    logger.debug(f"Session {session_id} expired, creating new")
                    del self.sessions[session_id]
                else:
                    session.touch()
                    return session

            # Create new session
            session = Session(session_id=session_id)
            self.sessions[session_id] = session
            logger.debug(f"Created new session: {session_id}")
            return session

    def get(self, session_id: str) -> Optional[Session]:
        """
        Get session by ID without creating.

        Args:
            session_id: Session identifier

        Returns:
            Session object or None
        """
        with self.lock:
            session = self.sessions.get(session_id)
            if session and not session.is_expired():
                session.touch()
                return session
            return None

    def add_messages(self, session_id: str, messages: List[Message]) -> None:
        """
        Add messages to a session.

        Args:
            session_id: Session identifier
            messages: Messages to add
        """
        session = self.get_or_create(session_id)
        for msg in messages:
            session.add_message(msg)

    def add_response(self, session_id: str, response: Message) -> None:
        """
        Add assistant response to session.

        Args:
            session_id: Session identifier
            response: Assistant response message
        """
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].add_message(response)

    def set_claude_session(self, session_id: str, claude_session_id: str) -> None:
        """
        Set Claude SDK session ID for resumption.

        Args:
            session_id: Our session identifier
            claude_session_id: Claude SDK session ID
        """
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].claude_session_id = claude_session_id
                logger.debug(
                    f"Set Claude session {claude_session_id} for session {session_id}"
                )

    def get_claude_session(self, session_id: str) -> Optional[str]:
        """
        Get Claude SDK session ID.

        Args:
            session_id: Our session identifier

        Returns:
            Claude SDK session ID or None
        """
        with self.lock:
            session = self.sessions.get(session_id)
            if session and not session.is_expired():
                return session.claude_session_id
            return None

    def delete(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.debug(f"Deleted session: {session_id}")
                return True
            return False

    def list_sessions(self) -> List[Dict]:
        """
        List all active sessions.

        Returns:
            List of session info dictionaries
        """
        with self.lock:
            return [
                {
                    "session_id": s.session_id,
                    "message_count": s.get_message_count(),
                    "claude_session_id": s.claude_session_id,
                    "created_at": s.created_at.isoformat(),
                    "last_accessed": s.last_accessed.isoformat(),
                    "expires_at": s.expires_at.isoformat(),
                }
                for s in self.sessions.values()
                if not s.is_expired()
            ]

    def get_stats(self) -> Dict:
        """
        Get session statistics.

        Returns:
            Statistics dictionary
        """
        with self.lock:
            active = [s for s in self.sessions.values() if not s.is_expired()]
            total_messages = sum(s.get_message_count() for s in active)

            return {
                "active_sessions": len(active),
                "total_messages": total_messages,
                "ttl_hours": self.ttl_hours,
            }

    async def start_cleanup(self) -> None:
        """Start background cleanup task."""

        async def cleanup_loop():
            while True:
                await asyncio.sleep(self.cleanup_interval)
                self._cleanup_expired()

        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info(
            f"Started session cleanup task (interval: {self.cleanup_interval}s)"
        )

    def _cleanup_expired(self) -> None:
        """Remove expired sessions."""
        with self.lock:
            expired = [k for k, v in self.sessions.items() if v.is_expired()]
            for k in expired:
                del self.sessions[k]
            if expired:
                logger.debug(f"Cleaned up {len(expired)} expired sessions")

    def shutdown(self) -> None:
        """Shutdown cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            logger.info("Session cleanup task stopped")
