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
    """Authentication result."""
    api_key: str


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
    Verify API key from Authorization header or x-api-key header.

    Uses single internal API key for backend authentication.
    """
    # Skip auth if no internal key configured
    if not settings.INTERNAL_API_KEY:
        return AuthResult(api_key="")

    # Try Authorization: Bearer header first
    provided_key = None
    if credentials:
        provided_key = credentials.credentials

    # Fall back to x-api-key header (used by Anthropic SDK)
    if not provided_key:
        provided_key = request.headers.get("x-api-key")

    if not provided_key:
        raise _auth_error(
            "Missing API key. Include 'Authorization: Bearer YOUR_API_KEY' or 'x-api-key' header.",
            "missing_api_key",
        )

    # Verify against internal API key
    if provided_key != settings.INTERNAL_API_KEY:
        client_host = request.client.host if request.client else "unknown"
        logger.warning(f"Invalid API key attempt from {client_host}")
        raise _auth_error("Invalid API key provided.", "invalid_api_key")

    logger.debug(f"Authenticated with internal key")
    return AuthResult(api_key=provided_key)
