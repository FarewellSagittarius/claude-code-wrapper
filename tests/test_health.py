"""Tests for health and root endpoints."""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Test health and info endpoints."""

    def test_root_endpoint(self, client: TestClient):
        """GET / should return API info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Claude OpenAI Wrapper"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
        assert data["endpoints"]["chat"] == "/v1/chat/completions"
        assert data["endpoints"]["messages"] == "/v1/messages"
        assert data["endpoints"]["models"] == "/v1/models"
        assert data["endpoints"]["sessions"] == "/v1/sessions"
        assert data["endpoints"]["health"] == "/health"

    def test_health_endpoint(self, client: TestClient):
        """GET /health should return healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_openapi_docs(self, client: TestClient):
        """GET /docs should return OpenAPI documentation."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_openapi_json(self, client: TestClient):
        """GET /openapi.json should return OpenAPI spec."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "Claude OpenAI Wrapper"
