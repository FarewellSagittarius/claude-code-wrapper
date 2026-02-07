"""Session management for conversation continuity."""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from ..models.anthropic import Message

logger = logging.getLogger(__name__)


def compute_messages_hash(messages: List[Dict[str, Any]]) -> str:
    """
    Compute hash of messages list directly (raw format).

    Args:
        messages: List of message dicts (raw format from client)

    Returns:
        20-char hash string
    """
    data = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()[:20]


def compute_session_hash(messages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Compute hash for session lookup.

    For lookup: remove the last user message and last assistant message,
    then hash the remaining messages. This matches what was stored after
    the previous round.

    Args:
        messages: List of message dicts from client request

    Returns:
        20-char hash string, or None if not enough messages
    """
    if not messages:
        return None

    # Need at least: previous messages + assistant reply + new user message
    # So minimum 3 messages for session lookup
    if len(messages) < 3:
        return None

    # Find and remove the last user message
    if messages[-1].get("role") != "user":
        return None

    # Remove last user message
    without_last_user = messages[:-1]

    # Find and remove the last assistant message
    if not without_last_user or without_last_user[-1].get("role") != "assistant":
        return None

    # Remove last assistant message
    base_messages = without_last_user[:-1]

    if not base_messages:
        return None

    # Hash the base messages (should match what was stored in previous round)
    hash_value = compute_messages_hash(base_messages)

    logger.info(f"Lookup hash: {hash_value[:8]}... from {len(base_messages)} base msgs")

    return hash_value


def compute_storage_hash(messages: List[Dict[str, Any]]) -> str:
    """
    Compute hash for session storage.

    Store hash of the raw messages received in this request.
    Next request will strip its last user+assistant to match this.

    Args:
        messages: List of message dicts from client request

    Returns:
        20-char hash string
    """
    hash_value = compute_messages_hash(messages)
    logger.info(f"Storage hash: {hash_value[:8]}... from {len(messages)} msgs")
    return hash_value


def extract_new_messages(
    messages: List[Dict[str, Any]],
    hash_to_session: Dict[str, str],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Extract new messages from conversation, identify session via hash.

    Logic:
    - Compute lookup hash by removing last user msg + last assistant msg
    - This matches the storage hash from the previous round

    Args:
        messages: Full message history from client
        hash_to_session: Mapping of content hash to session_id

    Returns:
        Tuple of (session_id or None, list of new messages)
    """
    if not messages:
        return None, []

    # Try to compute lookup hash (needs at least 3 messages)
    session_hash = compute_session_hash(messages)

    if session_hash is None:
        # Not enough messages for lookup, this is a new or early conversation
        logger.info(f"New conversation ({len(messages)} messages, need 3+ for lookup)")
        return None, messages

    # Look up session
    session_id = hash_to_session.get(session_hash)

    if session_id:
        logger.info(f"Found session by hash: {session_hash[:8]}... → {session_id}")
    else:
        logger.info(f"Hash {session_hash[:8]}... not found in {len(hash_to_session)} mappings")

    # The last user message is the new one
    new_messages = [messages[-1]] if messages else []

    return session_id, new_messages


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
        self.hash_to_session: Dict[str, str] = {}  # content_hash → session_id
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
        # Ensure session exists before setting claude_session_id
        session = self.get_or_create(session_id)
        with self.lock:
            session.claude_session_id = claude_session_id
            logger.info(
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

    def store_hash_mapping(
        self, messages: List[Dict[str, Any]], session_id: str
    ) -> str:
        """
        Store hash → session_id mapping for conversation history.

        Stores hash of the raw messages. Next request will strip its
        last user+assistant messages to match this hash.

        Args:
            messages: Original messages from this request (NOT including response)
            session_id: Session identifier to map to

        Returns:
            The computed hash
        """
        hash_value = compute_storage_hash(messages)
        with self.lock:
            self.hash_to_session[hash_value] = session_id
            logger.info(f"Stored hash mapping: {hash_value[:8]}... → {session_id}")
        return hash_value

    def find_session_by_hash(
        self, messages: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Find session_id by computing hash of message history.

        Args:
            messages: Conversation history from client

        Returns:
            session_id if found, None otherwise
        """
        hash_value = compute_session_hash(messages)
        if not hash_value:
            return None

        with self.lock:
            session_id = self.hash_to_session.get(hash_value)
            if session_id:
                # Verify session still exists and not expired
                session = self.sessions.get(session_id)
                if session and not session.is_expired():
                    logger.debug(
                        f"Found session by hash: {hash_value[:8]}... → {session_id}"
                    )
                    return session_id
                # Clean up stale mapping
                del self.hash_to_session[hash_value]
            return None

    def find_session_and_extract_new(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Find session from messages and extract new messages.

        Convenience method combining hash lookup and message extraction.

        Args:
            messages: Full message history from client

        Returns:
            Tuple of (session_id or None, new messages to process)
        """
        return extract_new_messages(messages, self.hash_to_session)

    def delete(self, session_id: str) -> bool:
        """
        Delete a session and its hash mappings.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                # Clean up hash mappings for this session
                stale_hashes = [
                    h for h, s in self.hash_to_session.items() if s == session_id
                ]
                for h in stale_hashes:
                    del self.hash_to_session[h]
                logger.debug(
                    f"Deleted session: {session_id}, "
                    f"removed {len(stale_hashes)} hash mappings"
                )
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
        """Remove expired sessions and their hash mappings."""
        with self.lock:
            expired = [k for k, v in self.sessions.items() if v.is_expired()]
            for k in expired:
                del self.sessions[k]

            # Clean up hash mappings for expired sessions
            expired_set = set(expired)
            stale_hashes = [
                h for h, s in self.hash_to_session.items() if s in expired_set
            ]
            for h in stale_hashes:
                del self.hash_to_session[h]

            if expired:
                logger.debug(
                    f"Cleaned up {len(expired)} expired sessions, "
                    f"{len(stale_hashes)} hash mappings"
                )

    def shutdown(self) -> None:
        """Shutdown cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            logger.info("Session cleanup task stopped")
