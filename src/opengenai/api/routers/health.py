"""
Health check endpoints for OpenGenAI API.
"""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter

from opengenai.core.config import get_settings
from opengenai.core.logging import get_logger
from opengenai.storage.database import close_db, init_db

router = APIRouter()
logger = get_logger(__name__)


@router.get("/", response_model=dict[str, Any])
async def health_root() -> dict[str, Any]:
    """Basic health endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC),
        "version": get_settings().app_version,
    }


@router.get("/detailed", response_model=dict[str, Any])
async def health_detailed() -> dict[str, Any]:
    """Detailed health endpoint checks database connectivity."""
    db_ok = True
    try:
        await init_db()
    except Exception as exc:  # pragma: no cover
        db_ok = False
        logger.error("Database health check failed", error=str(exc))
    finally:
        try:
            await close_db()
        except Exception:
            pass

    return {
        "status": "healthy" if db_ok else "unhealthy",
        "database": db_ok,
        "version": get_settings().app_version,
        "timestamp": datetime.now(UTC),
    }
