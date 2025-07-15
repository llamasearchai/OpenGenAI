"""Agents Router for OpenGenAI API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from opengenai.agents.registry import get_agent_registry
from opengenai.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)
registry = get_agent_registry()


@router.get("/", response_model=list[dict[str, Any]])
async def list_agents() -> list[dict[str, Any]]:
    """List registered agent instances."""
    return registry.list_agent_instances()


@router.get("/{agent_id}", response_model=dict[str, Any])
async def get_agent(agent_id: str) -> dict[str, Any]:
    """Retrieve an agent instance by ID."""
    try:
        return registry.get_agent_instance_info(agent_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
