"""Middleware for Claude OpenAI Wrapper."""

from .auth import verify_api_key

__all__ = ["verify_api_key"]
