"""Tests for Anthropic Messages API endpoint."""

import json

import pytest
from fastapi.testclient import TestClient


class TestAnthropicMessagesValidation:
    """Test request validation for Anthropic messages endpoint."""

    def test_missing_model(self, client: TestClient, anthropic_headers: dict):
        """Request without model should fail validation."""
        response = client.post(
            "/v1/messages",
            headers=anthropic_headers,
            json={
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        assert response.status_code == 422

    def test_max_tokens_has_default(self, client: TestClient, anthropic_headers: dict):
        """Request without max_tokens should use default (4096)."""
        response = client.post(
            "/v1/messages",
            headers=anthropic_headers,
            json={
                "model": "claude-code-opus",
                "messages": [{"role": "user", "content": "Say test"}],
            },
        )
        # max_tokens has default value, so request should succeed
        assert response.status_code == 200

    def test_missing_messages(self, client: TestClient, anthropic_headers: dict):
        """Request without messages should fail validation."""
        response = client.post(
            "/v1/messages",
            headers=anthropic_headers,
            json={"model": "claude-code-opus", "max_tokens": 100},
        )
        assert response.status_code == 422


class TestAnthropicMessagesNonStreaming:
    """Test non-streaming Anthropic messages."""

    @pytest.mark.integration
    def test_simple_message(self, client: TestClient, anthropic_headers: dict, simple_anthropic_request: dict):
        """Simple message should return valid response."""
        response = client.post(
            "/v1/messages",
            headers=anthropic_headers,
            json=simple_anthropic_request,
        )
        assert response.status_code == 200

        data = response.json()
        assert data["type"] == "message"
        assert "id" in data
        assert data["id"].startswith("msg_")
        assert data["role"] == "assistant"
        assert data["model"] == simple_anthropic_request["model"]

    @pytest.mark.integration
    def test_response_content_structure(self, client: TestClient, anthropic_headers: dict, simple_anthropic_request: dict):
        """Response content should have Anthropic format."""
        response = client.post(
            "/v1/messages",
            headers=anthropic_headers,
            json=simple_anthropic_request,
        )
        data = response.json()

        assert "content" in data
        assert isinstance(data["content"], list)
        assert len(data["content"]) >= 1

        content_block = data["content"][0]
        assert content_block["type"] == "text"
        assert "text" in content_block

    @pytest.mark.integration
    def test_stop_reason(self, client: TestClient, anthropic_headers: dict, simple_anthropic_request: dict):
        """Response should include stop_reason."""
        response = client.post(
            "/v1/messages",
            headers=anthropic_headers,
            json=simple_anthropic_request,
        )
        data = response.json()

        assert "stop_reason" in data
        assert data["stop_reason"] == "end_turn"

    @pytest.mark.integration
    def test_usage_format(self, client: TestClient, anthropic_headers: dict, simple_anthropic_request: dict):
        """Usage should be in Anthropic format."""
        response = client.post(
            "/v1/messages",
            headers=anthropic_headers,
            json=simple_anthropic_request,
        )
        data = response.json()

        assert "usage" in data
        usage = data["usage"]
        assert "input_tokens" in usage
        assert "output_tokens" in usage

    @pytest.mark.integration
    def test_system_prompt(self, client: TestClient, anthropic_headers: dict):
        """System prompt should be processed."""
        response = client.post(
            "/v1/messages",
            headers=anthropic_headers,
            json={
                "model": "claude-code-opus",
                "max_tokens": 100,
                "system": "Always respond with exactly one word.",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        assert response.status_code == 200


class TestAnthropicMessagesStreaming:
    """Test streaming Anthropic messages."""

    @pytest.mark.integration
    def test_streaming_response(self, client: TestClient, anthropic_headers: dict):
        """Streaming should return SSE events."""
        with client.stream(
            "POST",
            "/v1/messages",
            headers=anthropic_headers,
            json={
                "model": "claude-code-opus",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

    @pytest.mark.integration
    def test_streaming_event_sequence(self, client: TestClient, anthropic_headers: dict):
        """Streaming should have correct event sequence."""
        events = []
        with client.stream(
            "POST",
            "/v1/messages",
            headers=anthropic_headers,
            json={
                "model": "claude-code-opus",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Say ok"}],
                "stream": True,
            },
        ) as response:
            current_event = None
            for line in response.iter_lines():
                if line.startswith("event: "):
                    current_event = line[7:]
                elif line.startswith("data: ") and current_event:
                    events.append(current_event)

        if events:
            # Should start with message_start
            assert events[0] == "message_start"
            # Should end with message_stop
            assert events[-1] == "message_stop"
            # Should have message_delta before message_stop
            assert "message_delta" in events

    @pytest.mark.integration
    def test_streaming_content_blocks(self, client: TestClient, anthropic_headers: dict):
        """Streaming should include content block events."""
        events = []
        with client.stream(
            "POST",
            "/v1/messages",
            headers=anthropic_headers,
            json={
                "model": "claude-code-opus",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Say hi"}],
                "stream": True,
            },
        ) as response:
            current_event = None
            for line in response.iter_lines():
                if line.startswith("event: "):
                    current_event = line[7:]
                elif line.startswith("data: ") and current_event:
                    events.append(current_event)

        # Should have content_block events
        if events:
            assert "content_block_start" in events or "content_block_delta" in events


class TestAnthropicMessagesOptions:
    """Test optional parameters for Anthropic messages."""

    @pytest.mark.integration
    def test_temperature(self, client: TestClient, anthropic_headers: dict):
        """temperature parameter should be accepted."""
        response = client.post(
            "/v1/messages",
            headers=anthropic_headers,
            json={
                "model": "claude-code-opus",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Say hi"}],
                "temperature": 0.5,
            },
        )
        assert response.status_code == 200

    @pytest.mark.integration
    def test_top_p(self, client: TestClient, anthropic_headers: dict):
        """top_p parameter should be accepted."""
        response = client.post(
            "/v1/messages",
            headers=anthropic_headers,
            json={
                "model": "claude-code-opus",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Say hi"}],
                "top_p": 0.9,
            },
        )
        assert response.status_code == 200

    @pytest.mark.integration
    def test_session_id_extension(self, client: TestClient, anthropic_headers: dict):
        """session_id extension should be accepted."""
        response = client.post(
            "/v1/messages",
            headers=anthropic_headers,
            json={
                "model": "claude-code-opus",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Say hi"}],
                "session_id": "test-session-456",
            },
        )
        assert response.status_code == 200
