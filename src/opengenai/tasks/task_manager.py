"""
Task Manager for OpenGenAI
Manages task lifecycle, execution, and monitoring.
"""

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from opengenai.core.logging import get_logger
from opengenai.core.types import (
    TaskState,
    TaskStatus,
)

logger = get_logger(__name__)


class TaskManager:
    """A naive in-memory task manager implementation."""

    def __init__(self) -> None:  # noqa: D401
        self._tasks: dict[str, TaskState] = {}
        self._lock = asyncio.Lock()

    # ---------------------------------------------------------------------
    # CRUD
    # ---------------------------------------------------------------------

    async def create_task(self, pipeline: str, config: dict[str, Any]) -> str:
        """Create a new task and enqueue it for execution."""
        task_id = str(uuid.uuid4())
        task_state = TaskState(
            id=task_id,
            name=pipeline,
            status=TaskStatus.PENDING,
            agent_id=None,
            created_at=datetime.now(UTC),
            started_at=None,
            completed_at=None,
            error_message=None,
            progress_percentage=0.0,
            retry_count=0,
            result=None,
            metrics={},
        )
        async with self._lock:
            self._tasks[task_id] = task_state
        logger.info("Task created", task_id=task_id, pipeline=pipeline)
        # For now tasks are not executed automatically.
        return task_id

    async def list_tasks(self) -> list[dict[str, Any]]:
        """Return list of all tasks."""
        async with self._lock:
            return [t.model_dump() for t in self._tasks.values()]

    async def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Retrieve task status."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                raise KeyError(f"Task {task_id} does not exist")
            return task.model_dump()

    async def cancel_task(self, task_id: str) -> None:
        """Cancel a pending/running task."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                raise KeyError(f"Task {task_id} does not exist")
            if task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
                logger.warning("Cannot cancel task in final state", task_id=task_id)
                return
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now(UTC)
        logger.info("Task cancelled", task_id=task_id)

    def get_stats(self) -> dict[str, Any]:
        """Get task manager statistics."""
        total_tasks = len(self._tasks)
        running_tasks = sum(1 for task in self._tasks.values() if task.status == TaskStatus.RUNNING)
        completed_tasks = sum(
            1 for task in self._tasks.values() if task.status == TaskStatus.COMPLETED
        )
        failed_tasks = sum(1 for task in self._tasks.values() if task.status == TaskStatus.FAILED)

        return {
            "total_tasks": total_tasks,
            "running_tasks": running_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "pending_tasks": total_tasks - running_tasks - completed_tasks - failed_tasks,
        }


# Global task manager instance
_task_manager: TaskManager | None = None


def get_task_manager() -> TaskManager:
    """Get the global task manager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
