"""Tests for models endpoints."""

import pytest
from fastapi.testclient import TestClient


class TestModelsEndpoint:
    """Test /v1/models endpoints."""

    def test_list_models(self, client: TestClient):
        """GET /v1/models should return list of models."""
        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) >= 3

    def test_list_models_structure(self, client: TestClient):
        """Each model should have correct structure."""
        response = client.get("/v1/models")
        data = response.json()

        for model in data["data"]:
            assert "id" in model
            assert model["object"] == "model"
            assert "created" in model
            assert model["owned_by"] == "anthropic"

    def test_model_aliases_exist(self, client: TestClient):
        """All expected model aliases should be present."""
        response = client.get("/v1/models")
        data = response.json()

        model_ids = [m["id"] for m in data["data"]]
        assert "claude-code-opus" in model_ids
        assert "claude-code-sonnet" in model_ids
        assert "claude-code-haiku" in model_ids

    def test_get_specific_model(self, client: TestClient):
        """GET /v1/models/{model_id} should return model info."""
        response = client.get("/v1/models/claude-code-opus")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == "claude-code-opus"
        assert data["object"] == "model"
        assert data["owned_by"] == "anthropic"

    def test_get_nonexistent_model_returns_info(self, client: TestClient):
        """GET /v1/models/{invalid_id} returns info for flexibility."""
        response = client.get("/v1/models/nonexistent-model")
        # API returns info for any model (allows custom models)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "nonexistent-model"

    @pytest.mark.parametrize("model_id", [
        "claude-code-opus",
        "claude-code-sonnet",
        "claude-code-haiku",
    ])
    def test_all_models_accessible(self, client: TestClient, model_id: str):
        """All supported models should be accessible."""
        response = client.get(f"/v1/models/{model_id}")
        assert response.status_code == 200
        assert response.json()["id"] == model_id
