"""Authentication middleware."""

import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import settings

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


@dataclass
class AuthResult:
    """Authentication result with tool mode."""
    api_key: str
    tool_mode: str  # "light", "basic", "heavy", or "custom"


def _auth_error(message: str, code: str) -> HTTPException:
    """Create authentication error response."""
    return HTTPException(
        status_code=401,
        detail={"error": {"message": message, "type": "invalid_request_error", "code": code}},
    )


async def verify_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> AuthResult:
    """
    Verify API key from Authorization header.

    Tool permissions by key prefix:
    - sk-light-*: No tools (chat only)
    - sk-basic-*: All built-in tools
    - sk-heavy-*: All tools + plugins + MCP servers
    - sk-custom-*: Use request settings
    """
    if not settings.API_KEYS:
        return AuthResult(api_key="", tool_mode="default")

    if not credentials:
        raise _auth_error(
            "Missing API key. Include 'Authorization: Bearer YOUR_API_KEY' header.",
            "missing_api_key",
        )

    provided_key = credentials.credentials

    # Match against configured keys
    for prefix, config in settings.API_KEYS.items():
        if provided_key == config["key"]:
            logger.info(f"Authenticated with {prefix} key (tool_mode={config['tools']})")
            return AuthResult(
                api_key=provided_key,
                tool_mode=config["tools"],
            )

    logger.warning(f"Invalid API key attempt from {request.client.host}")
    raise _auth_error("Invalid API key provided.", "invalid_api_key")
