"""Admin Router for OpenGenAI API - placeholder for administrative operations."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from opengenai.core.config import settings
from opengenai.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _verify_admin():  # placeholder security dependency
    return True


@router.get("/stats", response_model=dict[str, Any], dependencies=[Depends(_verify_admin)])
async def stats() -> dict[str, Any]:
    """Return basic system statistics."""
    return {
        "version": settings.app_version,
        "environment": settings.environment,
    }
