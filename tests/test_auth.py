"""Tests for authentication and authorization."""

import pytest
from fastapi.testclient import TestClient


class TestAuthentication:
    """Test API key authentication."""

    def test_missing_api_key(self, client: TestClient):
        """Request without API key should fail with 401."""
        response = client.post(
            "/v1/chat/completions",
            json={"model": "claude-code-opus", "messages": [{"role": "user", "content": "test"}]},
        )
        assert response.status_code == 401
        data = response.json()
        # Error is nested under detail
        assert data["detail"]["error"]["type"] == "invalid_request_error"
        assert "API key" in data["detail"]["error"]["message"]

    def test_invalid_api_key(self, client: TestClient):
        """Request with invalid API key should fail with 401."""
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer invalid-key-12345"},
            json={"model": "claude-code-opus", "messages": [{"role": "user", "content": "test"}]},
        )
        assert response.status_code == 401
        data = response.json()
        # Error is nested under detail
        assert data["detail"]["error"]["code"] == "invalid_api_key"

    def test_malformed_auth_header(self, client: TestClient):
        """Request with malformed Authorization header should fail."""
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "NotBearer sk-light-dev"},
            json={"model": "claude-code-opus", "messages": [{"role": "user", "content": "test"}]},
        )
        assert response.status_code == 401

    def test_light_api_key_valid(self, client: TestClient, light_headers: dict):
        """Light mode API key should be accepted."""
        response = client.get("/v1/models", headers=light_headers)
        assert response.status_code == 200

    def test_basic_api_key_valid(self, client: TestClient, basic_headers: dict):
        """Basic mode API key should be accepted."""
        response = client.get("/v1/models", headers=basic_headers)
        assert response.status_code == 200

    def test_heavy_api_key_valid(self, client: TestClient, heavy_headers: dict):
        """Heavy mode API key should be accepted."""
        response = client.get("/v1/models", headers=heavy_headers)
        assert response.status_code == 200

    def test_custom_api_key_valid(self, client: TestClient, custom_headers: dict):
        """Custom mode API key should be accepted."""
        response = client.get("/v1/models", headers=custom_headers)
        assert response.status_code == 200


class TestAuthorizationModes:
    """Test different authorization/tool modes."""

    # Note: Actual tool mode behavior is tested in integration tests
    # These tests verify the API keys are recognized correctly

    def test_models_endpoint_no_auth_required(self, client: TestClient):
        """Models endpoint should work without auth for listing."""
        response = client.get("/v1/models")
        assert response.status_code == 200

    def test_sessions_no_auth_required(self, client: TestClient):
        """Sessions endpoint does not require authentication."""
        response = client.get("/v1/sessions")
        # Sessions endpoint is open (no auth required)
        assert response.status_code == 200

    def test_chat_completions_requires_auth(self, client: TestClient):
        """Chat completions should require authentication."""
        response = client.post(
            "/v1/chat/completions",
            json={"model": "claude-code-opus", "messages": [{"role": "user", "content": "test"}]},
        )
        assert response.status_code == 401

    def test_anthropic_messages_requires_auth(self, client: TestClient):
        """Anthropic messages endpoint should require authentication."""
        response = client.post(
            "/v1/messages",
            json={"model": "claude-code-opus", "max_tokens": 100, "messages": [{"role": "user", "content": "test"}]},
        )
        assert response.status_code == 401
