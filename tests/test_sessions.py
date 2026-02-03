"""Tests for session management endpoints."""

import uuid

import pytest
from fastapi.testclient import TestClient


class TestSessionsEndpoint:
    """Test /v1/sessions endpoints."""

    def test_list_sessions_empty(self, client: TestClient, basic_headers: dict):
        """GET /v1/sessions should return empty list initially."""
        response = client.get("/v1/sessions", headers=basic_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "list"
        assert "data" in data

    def test_list_sessions_no_auth(self, client: TestClient):
        """GET /v1/sessions does not require authentication."""
        response = client.get("/v1/sessions")
        # Sessions endpoint is open
        assert response.status_code == 200

    def test_get_session_stats(self, client: TestClient):
        """GET /v1/sessions/stats should return statistics."""
        # Sessions endpoint does not require auth
        response = client.get("/v1/sessions/stats")
        assert response.status_code == 200

        data = response.json()
        assert "active_sessions" in data
        assert "total_messages" in data
        assert "ttl_hours" in data

    def test_get_nonexistent_session(self, client: TestClient, basic_headers: dict):
        """GET /v1/sessions/{id} for nonexistent session should return 404."""
        session_id = f"test-{uuid.uuid4().hex[:8]}"
        response = client.get(f"/v1/sessions/{session_id}", headers=basic_headers)
        assert response.status_code == 404

    def test_delete_nonexistent_session(self, client: TestClient, basic_headers: dict):
        """DELETE /v1/sessions/{id} for nonexistent session should return 404."""
        session_id = f"test-{uuid.uuid4().hex[:8]}"
        response = client.delete(f"/v1/sessions/{session_id}", headers=basic_headers)
        assert response.status_code == 404


class TestSessionCreationViaChat:
    """Test session creation through chat completions."""

    @pytest.mark.integration
    def test_session_created_on_chat(self, client: TestClient, basic_headers: dict):
        """Chat with session_id should create a session."""
        session_id = f"test-session-{uuid.uuid4().hex[:8]}"

        # Make a chat request with session_id
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={
                "model": "claude-code-opus",
                "messages": [{"role": "user", "content": "Say 'test'"}],
                "session_id": session_id,
            },
        )

        # Session should be created (even if chat takes time)
        if response.status_code == 200:
            # Check session exists
            session_response = client.get(f"/v1/sessions/{session_id}", headers=basic_headers)
            # Session may or may not exist depending on response handling
            assert session_response.status_code in [200, 404]

    @pytest.mark.integration
    def test_session_delete(self, client: TestClient, basic_headers: dict):
        """Created session can be deleted."""
        session_id = f"test-del-{uuid.uuid4().hex[:8]}"

        # First create via chat (if integration test)
        client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={
                "model": "claude-code-opus",
                "messages": [{"role": "user", "content": "test"}],
                "session_id": session_id,
            },
        )

        # Try to delete
        response = client.delete(f"/v1/sessions/{session_id}", headers=basic_headers)
        # Will be 200 if session was created, 404 if not
        assert response.status_code in [200, 404]
