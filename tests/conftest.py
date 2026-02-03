"""Pytest configuration and fixtures for wrapper tests."""

import asyncio
import os
import sys
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import app
from src.config import settings


# API Keys for different tool modes
API_KEYS = {
    "light": "sk-light-dev",
    "basic": "sk-basic-dev",
    "heavy": "sk-heavy-dev",
    "custom": "sk-custom-dev",
}

# Test models
TEST_MODELS = [
    "claude-code-opus",
    "claude-code-sonnet",
    "claude-code-haiku",
]

# Base URL for tests
BASE_URL = "http://test"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Create synchronous test client."""
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture(scope="module")
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as ac:
        yield ac


@pytest.fixture
def light_headers() -> dict:
    """Headers with light mode API key (no tools)."""
    return {
        "Authorization": f"Bearer {API_KEYS['light']}",
        "Content-Type": "application/json",
    }


@pytest.fixture
def basic_headers() -> dict:
    """Headers with basic mode API key (built-in tools)."""
    return {
        "Authorization": f"Bearer {API_KEYS['basic']}",
        "Content-Type": "application/json",
    }


@pytest.fixture
def heavy_headers() -> dict:
    """Headers with heavy mode API key (all tools + MCP)."""
    return {
        "Authorization": f"Bearer {API_KEYS['heavy']}",
        "Content-Type": "application/json",
    }


@pytest.fixture
def custom_headers() -> dict:
    """Headers with custom mode API key."""
    return {
        "Authorization": f"Bearer {API_KEYS['custom']}",
        "Content-Type": "application/json",
    }


@pytest.fixture
def anthropic_headers() -> dict:
    """Headers for Anthropic API format."""
    return {
        "Authorization": f"Bearer {API_KEYS['basic']}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }


@pytest.fixture
def simple_chat_request() -> dict:
    """Simple chat completion request."""
    return {
        "model": "claude-code-opus",
        "messages": [{"role": "user", "content": "Say 'test' and nothing else"}],
    }


@pytest.fixture
def simple_anthropic_request() -> dict:
    """Simple Anthropic messages request."""
    return {
        "model": "claude-code-opus",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Say 'test' and nothing else"}],
    }


@pytest.fixture
def streaming_chat_request() -> dict:
    """Streaming chat completion request."""
    return {
        "model": "claude-code-opus",
        "messages": [{"role": "user", "content": "Say 'hello' and nothing else"}],
        "stream": True,
    }


@pytest.fixture
def multimodal_message() -> dict:
    """Multimodal message with text and image placeholder."""
    # Small 1x1 red PNG in base64
    red_pixel_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image briefly"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{red_pixel_png}"},
            },
        ],
    }


@pytest.fixture
def conversation_messages() -> list:
    """Multi-turn conversation messages."""
    return [
        {"role": "user", "content": "My name is TestUser"},
        {"role": "assistant", "content": "Hello TestUser!"},
        {"role": "user", "content": "What is my name?"},
    ]
