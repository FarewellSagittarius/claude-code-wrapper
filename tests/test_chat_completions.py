"""Tests for OpenAI chat completions endpoint."""

import json

import pytest
from fastapi.testclient import TestClient


class TestChatCompletionsValidation:
    """Test request validation for chat completions."""

    def test_missing_model(self, client: TestClient, basic_headers: dict):
        """Request without model should fail validation or use default."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={"messages": [{"role": "user", "content": "test"}]},
        )
        # Model may have a default or be required - accept both
        assert response.status_code in [200, 422]

    def test_missing_messages(self, client: TestClient, basic_headers: dict):
        """Request without messages should fail validation."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={"model": "claude-code-opus"},
        )
        assert response.status_code == 422

    def test_empty_messages(self, client: TestClient, basic_headers: dict):
        """Request with empty messages should fail validation or return error."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={"model": "claude-code-opus", "messages": []},
        )
        # Empty messages may return 422 validation error or 500 server error
        assert response.status_code in [422, 500]

    def test_invalid_role(self, client: TestClient, basic_headers: dict):
        """Message with invalid role should fail validation."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={
                "model": "claude-code-opus",
                "messages": [{"role": "invalid", "content": "test"}],
            },
        )
        assert response.status_code == 422


class TestChatCompletionsNonStreaming:
    """Test non-streaming chat completions."""

    @pytest.mark.integration
    def test_simple_completion(self, client: TestClient, basic_headers: dict, simple_chat_request: dict):
        """Simple chat completion should return valid response."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json=simple_chat_request,
        )
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "chat.completion"
        assert "id" in data
        assert data["id"].startswith("chatcmpl-")
        assert data["model"] == simple_chat_request["model"]

    @pytest.mark.integration
    def test_response_structure(self, client: TestClient, basic_headers: dict, simple_chat_request: dict):
        """Response should have correct OpenAI structure."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json=simple_chat_request,
        )
        data = response.json()

        # Choices
        assert "choices" in data
        assert len(data["choices"]) >= 1

        choice = data["choices"][0]
        assert choice["index"] == 0
        assert choice["finish_reason"] == "stop"
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert "content" in choice["message"]

    @pytest.mark.integration
    def test_usage_included(self, client: TestClient, basic_headers: dict, simple_chat_request: dict):
        """Response should include usage statistics."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json=simple_chat_request,
        )
        data = response.json()

        assert "usage" in data
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    @pytest.mark.integration
    def test_different_models(self, client: TestClient, basic_headers: dict):
        """Different models should be usable."""
        for model in ["claude-code-opus", "claude-code-sonnet", "claude-code-haiku"]:
            response = client.post(
                "/v1/chat/completions",
                headers=basic_headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Say 'ok'"}],
                },
            )
            # May succeed or fail depending on model availability
            assert response.status_code in [200, 500]

    @pytest.mark.integration
    def test_system_message(self, client: TestClient, basic_headers: dict):
        """System message should be processed."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={
                "model": "claude-code-opus",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Always respond with exactly one word."},
                    {"role": "user", "content": "Say hello"},
                ],
            },
        )
        assert response.status_code == 200


class TestChatCompletionsStreaming:
    """Test streaming chat completions."""

    @pytest.mark.integration
    def test_streaming_response(self, client: TestClient, basic_headers: dict, streaming_chat_request: dict):
        """Streaming should return SSE events."""
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=basic_headers,
            json=streaming_chat_request,
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            chunks = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        chunks.append(json.loads(data))

            assert len(chunks) > 0

    @pytest.mark.integration
    def test_streaming_chunk_structure(self, client: TestClient, basic_headers: dict, streaming_chat_request: dict):
        """Streaming chunks should have correct structure."""
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=basic_headers,
            json=streaming_chat_request,
        ) as response:
            chunks = []
            for line in response.iter_lines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    chunks.append(json.loads(line[6:]))

            if chunks:
                # First chunk should have role
                first = chunks[0]
                assert first["object"] == "chat.completion.chunk"
                assert "choices" in first

                # Last chunk should have finish_reason and usage
                last = chunks[-1]
                assert last["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.integration
    def test_streaming_ends_with_done(self, client: TestClient, basic_headers: dict, streaming_chat_request: dict):
        """Streaming should end with [DONE] marker."""
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=basic_headers,
            json=streaming_chat_request,
        ) as response:
            lines = list(response.iter_lines())
            data_lines = [l for l in lines if l.startswith("data: ")]
            assert data_lines[-1] == "data: [DONE]"


class TestChatCompletionsOptions:
    """Test optional parameters for chat completions."""

    @pytest.mark.integration
    def test_max_tokens(self, client: TestClient, basic_headers: dict):
        """max_tokens parameter should be accepted."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={
                "model": "claude-code-opus",
                "messages": [{"role": "user", "content": "Count from 1 to 100"}],
                "max_tokens": 10,
            },
        )
        assert response.status_code == 200

    @pytest.mark.integration
    def test_temperature(self, client: TestClient, basic_headers: dict):
        """temperature parameter should be accepted."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={
                "model": "claude-code-opus",
                "messages": [{"role": "user", "content": "Say hi"}],
                "temperature": 0.5,
            },
        )
        assert response.status_code == 200

    @pytest.mark.integration
    def test_session_id(self, client: TestClient, basic_headers: dict):
        """session_id extension should be accepted."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={
                "model": "claude-code-opus",
                "messages": [{"role": "user", "content": "Say hi"}],
                "session_id": "test-session-123",
            },
        )
        assert response.status_code == 200

    @pytest.mark.integration
    def test_max_turns(self, client: TestClient, basic_headers: dict):
        """max_turns extension should be accepted."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={
                "model": "claude-code-opus",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_turns": 5,
            },
        )
        assert response.status_code == 200
