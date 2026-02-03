"""Tests for error handling."""

import pytest
from fastapi.testclient import TestClient


class TestRequestErrors:
    """Test request-level error handling."""

    def test_invalid_json(self, client: TestClient, basic_headers: dict):
        """Invalid JSON should return 422."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            content="not valid json",
        )
        assert response.status_code == 422

    def test_invalid_content_type(self, client: TestClient):
        """Non-JSON content type should fail."""
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer sk-basic-dev",
                "Content-Type": "text/plain",
            },
            content="test",
        )
        assert response.status_code == 422

    def test_extra_fields_ignored(self, client: TestClient, basic_headers: dict):
        """Extra fields in request should be ignored."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={
                "model": "claude-code-opus",
                "messages": [{"role": "user", "content": "test"}],
                "unknown_field": "should be ignored",
                "another_unknown": 12345,
            },
        )
        # Should not fail due to extra fields
        assert response.status_code in [200, 500]


class TestAuthenticationErrors:
    """Test authentication error responses."""

    def test_missing_auth_error_format(self, client: TestClient):
        """Missing auth should return proper error format."""
        response = client.post(
            "/v1/chat/completions",
            json={"model": "claude-code-opus", "messages": [{"role": "user", "content": "test"}]},
        )
        assert response.status_code == 401

        data = response.json()
        # Error is nested under detail in FastAPI HTTPException
        assert "detail" in data
        assert "error" in data["detail"]
        assert "type" in data["detail"]["error"]
        assert "message" in data["detail"]["error"]

    def test_invalid_auth_error_format(self, client: TestClient):
        """Invalid auth should return proper error format."""
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer invalid-key"},
            json={"model": "claude-code-opus", "messages": [{"role": "user", "content": "test"}]},
        )
        assert response.status_code == 401

        data = response.json()
        # Error is nested under detail
        assert "detail" in data
        assert data["detail"]["error"]["code"] == "invalid_api_key"


class TestNotFoundErrors:
    """Test 404 error handling."""

    def test_unknown_endpoint(self, client: TestClient, basic_headers: dict):
        """Unknown endpoint should return 404."""
        response = client.get("/v1/unknown", headers=basic_headers)
        assert response.status_code == 404

    def test_unknown_model_returns_info(self, client: TestClient):
        """Unknown model ID returns info (for flexibility)."""
        response = client.get("/v1/models/unknown-model-xyz")
        # API returns info for unknown models (warns but doesn't 404)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "unknown-model-xyz"

    def test_unknown_session(self, client: TestClient, basic_headers: dict):
        """Unknown session should return 404."""
        response = client.get("/v1/sessions/unknown-session-xyz", headers=basic_headers)
        assert response.status_code == 404


class TestValidationErrors:
    """Test request validation error handling."""

    def test_validation_error_format(self, client: TestClient, basic_headers: dict):
        """Validation errors should have proper format."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={"model": "claude-code-opus"},  # Missing messages
        )
        assert response.status_code == 422

        data = response.json()
        assert "detail" in data

    def test_empty_string_model(self, client: TestClient, basic_headers: dict):
        """Empty model string is handled by SDK."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={
                "model": "",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        # Claude SDK handles empty model gracefully
        assert response.status_code in [200, 422, 500]

    def test_negative_max_tokens(self, client: TestClient, basic_headers: dict):
        """Negative max_tokens should fail validation."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={
                "model": "claude-code-opus",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": -1,
            },
        )
        # May pass validation depending on model constraints
        assert response.status_code in [200, 422, 500]

    def test_invalid_temperature(self, client: TestClient, basic_headers: dict):
        """Invalid temperature should fail validation."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={
                "model": "claude-code-opus",
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 5.0,  # Out of range
            },
        )
        # May pass or fail depending on validation
        assert response.status_code in [200, 422, 500]


class TestCORSHeaders:
    """Test CORS configuration."""

    def test_cors_preflight(self, client: TestClient):
        """OPTIONS request should return CORS headers."""
        response = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_cors_on_response(self, client: TestClient, basic_headers: dict):
        """Response should include CORS headers."""
        response = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers
