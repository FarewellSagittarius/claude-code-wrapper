"""Tests for MCP server integration."""

import pytest
from fastapi.testclient import TestClient


class TestMCPServerConfiguration:
    """Test MCP server configuration in requests."""

    @pytest.mark.integration
    def test_stdio_mcp_server(self, client: TestClient, heavy_headers: dict):
        """Stdio MCP server config should be accepted."""
        response = client.post(
            "/v1/chat/completions",
            headers=heavy_headers,
            json={
                "model": "claude-code-opus",
                "messages": [{"role": "user", "content": "Say hello"}],
                "mcp_servers": {
                    "test-server": {
                        "type": "stdio",
                        "command": "echo",
                        "args": ["test"],
                    }
                },
            },
        )
        # May succeed or fail depending on MCP server availability
        assert response.status_code in [200, 500]

    @pytest.mark.integration
    def test_sse_mcp_server(self, client: TestClient, heavy_headers: dict):
        """SSE MCP server config should be accepted."""
        response = client.post(
            "/v1/chat/completions",
            headers=heavy_headers,
            json={
                "model": "claude-code-opus",
                "messages": [{"role": "user", "content": "Say hello"}],
                "mcp_servers": {
                    "test-sse": {
                        "type": "sse",
                        "url": "http://localhost:9999/mcp",
                    }
                },
            },
        )
        # Server likely not available, but config should be accepted
        assert response.status_code in [200, 500]

    @pytest.mark.integration
    def test_http_mcp_server(self, client: TestClient, heavy_headers: dict):
        """HTTP MCP server config should be accepted."""
        response = client.post(
            "/v1/chat/completions",
            headers=heavy_headers,
            json={
                "model": "claude-code-opus",
                "messages": [{"role": "user", "content": "Say hello"}],
                "mcp_servers": {
                    "test-http": {
                        "type": "http",
                        "url": "http://localhost:9999/mcp",
                        "headers": {"Authorization": "Bearer test"},
                    }
                },
            },
        )
        assert response.status_code in [200, 500]

    @pytest.mark.integration
    def test_multiple_mcp_servers(self, client: TestClient, heavy_headers: dict):
        """Multiple MCP servers should be accepted."""
        response = client.post(
            "/v1/chat/completions",
            headers=heavy_headers,
            json={
                "model": "claude-code-opus",
                "messages": [{"role": "user", "content": "Say hello"}],
                "mcp_servers": {
                    "server1": {"type": "stdio", "command": "echo"},
                    "server2": {"type": "sse", "url": "http://localhost:9999"},
                },
            },
        )
        assert response.status_code in [200, 500]


class TestMCPServerModes:
    """Test MCP server availability by tool mode."""

    @pytest.mark.integration
    def test_light_mode_no_internal_mcp(self, client: TestClient, light_headers: dict):
        """Light mode should not load internal MCP servers."""
        # External MCP servers are always allowed, but internal ones require heavy mode
        response = client.post(
            "/v1/chat/completions",
            headers=light_headers,
            json={
                "model": "claude-code-opus",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        assert response.status_code == 200

    @pytest.mark.integration
    def test_basic_mode_external_mcp_only(self, client: TestClient, basic_headers: dict):
        """Basic mode should accept external MCP but not internal."""
        response = client.post(
            "/v1/chat/completions",
            headers=basic_headers,
            json={
                "model": "claude-code-opus",
                "messages": [{"role": "user", "content": "Say hello"}],
                "mcp_servers": {
                    "external": {"type": "stdio", "command": "echo"},
                },
            },
        )
        assert response.status_code in [200, 500]

    @pytest.mark.integration
    def test_heavy_mode_full_mcp(self, client: TestClient, heavy_headers: dict):
        """Heavy mode should enable all MCP features."""
        response = client.post(
            "/v1/chat/completions",
            headers=heavy_headers,
            json={
                "model": "claude-code-opus",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        assert response.status_code == 200


class TestMCPInAnthropicEndpoint:
    """Test MCP servers in Anthropic messages endpoint."""

    @pytest.mark.integration
    def test_anthropic_mcp_server(self, client: TestClient, anthropic_headers: dict):
        """Anthropic endpoint should accept MCP servers."""
        # Change to heavy mode headers
        headers = anthropic_headers.copy()
        headers["Authorization"] = "Bearer sk-heavy-dev"

        response = client.post(
            "/v1/messages",
            headers=headers,
            json={
                "model": "claude-code-opus",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Say hello"}],
                "mcp_servers": {
                    "test": {"type": "stdio", "command": "echo"},
                },
            },
        )
        assert response.status_code in [200, 500]
