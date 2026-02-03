"""Models listing route."""

import logging

from fastapi import APIRouter

from ..config import settings
from ..models.openai import ModelInfo, ModelListResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Models"])


@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """
    List available Claude models.

    Returns models in OpenAI-compatible format.
    """
    models = [
        ModelInfo(id=model_id, owned_by="anthropic")
        for model_id in settings.SUPPORTED_MODELS
    ]

    return ModelListResponse(data=models)


@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """
    Get information about a specific model.

    Args:
        model_id: The model identifier
    """
    if model_id not in settings.SUPPORTED_MODELS:
        # Still return info for unknown models (they might work)
        logger.warning(f"Unknown model requested: {model_id}")

    return ModelInfo(id=model_id, owned_by="anthropic")
