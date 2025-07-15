"""API Routers package for OpenGenAI."""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

__all__ = [
    "health",
    "agents",
    "tasks",
    "admin",
]

# Lazy import to avoid heavy dependencies during startup if not needed
if TYPE_CHECKING:
    from . import admin, agents, health, tasks  # noqa: F401
else:
    globals().update({name: import_module(f"opengenai.api.routers.{name}") for name in __all__})
