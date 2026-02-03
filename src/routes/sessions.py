"""Session management routes."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from ..services.session import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Sessions"])

# Global session manager (will be set by main.py)
session_manager: Optional[SessionManager] = None


def init_session_manager(manager: SessionManager) -> None:
    """Initialize session manager instance."""
    global session_manager
    session_manager = manager


@router.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")

    sessions = session_manager.list_sessions()
    return {
        "object": "list",
        "data": sessions,
    }


@router.get("/sessions/stats")
async def get_session_stats():
    """Get session statistics."""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")

    return session_manager.get_stats()


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get information about a specific session."""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")

    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session.session_id,
        "message_count": session.get_message_count(),
        "claude_session_id": session.claude_session_id,
        "created_at": session.created_at.isoformat(),
        "last_accessed": session.last_accessed.isoformat(),
        "expires_at": session.expires_at.isoformat(),
        "messages": [
            {"role": m.role, "content": m.content}
            for m in session.messages
        ],
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")

    deleted = session_manager.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"deleted": True, "session_id": session_id}
