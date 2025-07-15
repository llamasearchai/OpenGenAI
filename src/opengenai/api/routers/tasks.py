"""Tasks Router for OpenGenAI API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from opengenai.core.logging import get_logger
from opengenai.tasks.task_manager import TaskManager

router = APIRouter()
logger = get_logger(__name__)
manager = TaskManager()


@router.get("/", response_model=list[dict[str, Any]])
async def list_tasks() -> list[dict[str, Any]]:
    """List all tasks."""
    return await manager.list_tasks()


@router.post("/", response_model=dict[str, Any])
async def create_task(payload: dict[str, Any]) -> dict[str, Any]:
    """Create a new task."""
    pipeline = payload.get("pipeline", "generic")
    config = payload.get("config", {})
    task_id = await manager.create_task(pipeline, config)
    return {"task_id": task_id, "status": "created"}


@router.get("/{task_id}", response_model=dict[str, Any])
async def get_task(task_id: str) -> dict[str, Any]:
    """Get task status by ID."""
    try:
        return await manager.get_task_status(task_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")


@router.delete("/{task_id}")
async def cancel_task(task_id: str) -> dict[str, Any]:
    """Cancel a task by ID."""
    try:
        await manager.cancel_task(task_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    return {"task_id": task_id, "status": "cancelled"}
