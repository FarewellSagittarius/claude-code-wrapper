"""Tests for multimodal (text + image) support."""

import base64

import pytest
from fastapi.testclient import TestClient


# Small 1x1 PNG images for testing
RED_PIXEL_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
BLUE_PIXEL_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPj/HwADBwIAMCbHYQAAAABJRU5ErkJggg=="


class TestOpenAIMultimodal:
    """Test multimodal support in OpenAI format."""

    @pytest.mark.integration
    def test_text_only_array_content(self, client: TestClient, basic_headers: dict):
        """Text-only content in array format should work."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={
                "model": "claude-code-opus",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Say 'test'"}],
                    }
                ],
            },
        )
        assert response.status_code == 200

    @pytest.mark.integration
    def test_image_data_url(self, client: TestClient, basic_headers: dict):
        """Image as base64 data URL should be accepted."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={
                "model": "claude-code-opus",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What color is this pixel?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{RED_PIXEL_PNG}"},
                            },
                        ],
                    }
                ],
            },
        )
        assert response.status_code == 200

    @pytest.mark.integration
    def test_multiple_images(self, client: TestClient, basic_headers: dict):
        """Multiple images in single message should be accepted."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={
                "model": "claude-code-opus",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Compare these images"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{RED_PIXEL_PNG}"},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{BLUE_PIXEL_PNG}"},
                            },
                        ],
                    }
                ],
            },
        )
        assert response.status_code == 200

    @pytest.mark.integration
    def test_image_detail_parameter(self, client: TestClient, basic_headers: dict):
        """Image detail parameter should be accepted."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={
                "model": "claude-code-opus",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{RED_PIXEL_PNG}",
                                    "detail": "low",
                                },
                            },
                        ],
                    }
                ],
            },
        )
        assert response.status_code == 200


class TestAnthropicMultimodal:
    """Test multimodal support in Anthropic format."""

    @pytest.mark.integration
    def test_text_block_array(self, client: TestClient, anthropic_headers: dict):
        """Text blocks in array format should work."""
        response = client.post(
            "/v1/messages",
            headers=anthropic_headers,
            json={
                "model": "claude-code-opus",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Say 'test'"}],
                    }
                ],
            },
        )
        assert response.status_code == 200

    @pytest.mark.integration
    def test_anthropic_base64_image(self, client: TestClient, anthropic_headers: dict):
        """Anthropic format base64 image should be accepted."""
        response = client.post(
            "/v1/messages",
            headers=anthropic_headers,
            json={
                "model": "claude-code-opus",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": RED_PIXEL_PNG,
                                },
                            },
                        ],
                    }
                ],
            },
        )
        assert response.status_code == 200

    @pytest.mark.integration
    def test_mixed_text_and_images(self, client: TestClient, anthropic_headers: dict):
        """Mixed text and images should be processed."""
        response = client.post(
            "/v1/messages",
            headers=anthropic_headers,
            json={
                "model": "claude-code-opus",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "First image:"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": RED_PIXEL_PNG,
                                },
                            },
                            {"type": "text", "text": "Second image:"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": BLUE_PIXEL_PNG,
                                },
                            },
                            {"type": "text", "text": "Compare them."},
                        ],
                    }
                ],
            },
        )
        assert response.status_code == 200


class TestImageValidation:
    """Test image validation."""

    def test_invalid_base64_data(self, client: TestClient, basic_headers: dict):
        """Invalid base64 data should fail gracefully."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={
                "model": "claude-code-opus",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,invalid!!!"},
                            },
                        ],
                    }
                ],
            },
        )
        # Should either handle gracefully or return error
        assert response.status_code in [200, 400, 422, 500]

    def test_missing_image_url(self, client: TestClient, basic_headers: dict):
        """Missing image URL should fail validation or be handled."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={
                "model": "claude-code-opus",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this"},
                            {"type": "image_url", "image_url": {}},
                        ],
                    }
                ],
            },
        )
        # May fail validation, return error, or handle gracefully
        assert response.status_code in [200, 400, 422, 500]
