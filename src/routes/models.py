"""Model listing routes."""

from fastapi import APIRouter

router = APIRouter(tags=["Models"])

MODELS = [
    {
        "id": "claude-code",
        "type": "model",
        "display_name": "claude-code",
        "created_at": "2025-01-01T00:00:00Z",
    },
    {
        "id": "claude-code-opus",
        "type": "model",
        "display_name": "claude-code-opus",
        "created_at": "2025-01-01T00:00:00Z",
    },
    {
        "id": "claude-code-sonnet",
        "type": "model",
        "display_name": "claude-code-sonnet",
        "created_at": "2025-01-01T00:00:00Z",
    },
    {
        "id": "claude-code-haiku",
        "type": "model",
        "display_name": "claude-code-haiku",
        "created_at": "2025-01-01T00:00:00Z",
    },
]


@router.get("/models")
async def list_models():
    """List available models."""
    return {
        "data": MODELS,
        "has_more": False,
        "first_id": MODELS[0]["id"],
        "last_id": MODELS[-1]["id"],
    }
