"""
Agent Registry for OpenGenAI
Manages agent registration, discovery, and lifecycle.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opengenai.agents.base import BaseAgent
from opengenai.core.exceptions import (
    AgentError,
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ValidationError,
)
from opengenai.core.logging import get_logger
from opengenai.core.types import (
    AgentCapability,
    AgentConfig,
    AgentStatus,
)

logger = get_logger(__name__)


class AgentTypeInfo:
    """Information about an agent type."""

    def __init__(
        self,
        name: str,
        description: str,
        agent_class: type[BaseAgent],
        capabilities: list[AgentCapability],
        default_config: dict[str, Any] | None = None,
        version: str = "1.0.0",
        author: str = "OpenGenAI",
        tags: list[str] | None = None,
    ):
        """Initialize agent type info."""
        self.name = name
        self.description = description
        self.agent_class = agent_class
        self.capabilities = capabilities
        self.default_config = default_config or {}
        self.version = version
        self.author = author
        self.tags = tags or []
        self.created_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": [cap.value for cap in self.capabilities],
            "default_config": self.default_config,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def matches_capability(self, capability: AgentCapability) -> bool:
        """Check if agent type has specific capability."""
        return capability in self.capabilities

    def matches_tag(self, tag: str) -> bool:
        """Check if agent type has specific tag."""
        return tag in self.tags


class AgentRegistry:
    """Registry for agent types and configurations."""

    def __init__(self, storage_path: Path | None = None):
        """Initialize agent registry."""
        self.storage_path = storage_path or Path("./data/agents")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory registry
        self._agent_types: dict[str, AgentTypeInfo] = {}
        self._agent_instances: dict[str, dict[str, Any]] = {}  # agent_id -> info

        # Load from storage
        self._load_from_storage()

        # Register built-in agent types
        self._register_builtin_types()

    def register_agent_type(
        self,
        name: str,
        description: str,
        agent_class: type[BaseAgent],
        capabilities: list[AgentCapability],
        default_config: dict[str, Any] | None = None,
        version: str = "1.0.0",
        author: str = "OpenGenAI",
        tags: list[str] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Register a new agent type."""
        if name in self._agent_types and not overwrite:
            raise ResourceAlreadyExistsError(
                f"Agent type '{name}' already exists",
                resource_type="agent_type",
                resource_id=name,
            )

        # Validate agent class
        if not issubclass(agent_class, BaseAgent):
            raise ValidationError(
                "Agent class must inherit from BaseAgent",
                field="agent_class",
                value=agent_class,
            )

        # Create agent type info
        agent_type_info = AgentTypeInfo(
            name=name,
            description=description,
            agent_class=agent_class,
            capabilities=capabilities,
            default_config=default_config,
            version=version,
            author=author,
            tags=tags,
        )

        self._agent_types[name] = agent_type_info

        # Save to storage
        self._save_to_storage()

        logger.info(
            "Agent type registered",
            name=name,
            capabilities=[cap.value for cap in capabilities],
            version=version,
        )

    def unregister_agent_type(self, name: str) -> None:
        """Unregister an agent type."""
        if name not in self._agent_types:
            raise ResourceNotFoundError(
                f"Agent type '{name}' not found",
                resource_type="agent_type",
                resource_id=name,
            )

        del self._agent_types[name]
        self._save_to_storage()

        logger.info("Agent type unregistered", name=name)

    def get_agent_type(self, name: str) -> AgentTypeInfo:
        """Get agent type by name."""
        if name not in self._agent_types:
            raise ResourceNotFoundError(
                f"Agent type '{name}' not found",
                resource_type="agent_type",
                resource_id=name,
            )

        return self._agent_types[name]

    def list_agent_types(
        self,
        capability: AgentCapability | None = None,
        tag: str | None = None,
        author: str | None = None,
    ) -> list[AgentTypeInfo]:
        """List agent types with optional filtering."""
        agent_types = list(self._agent_types.values())

        if capability:
            agent_types = [
                agent_type
                for agent_type in agent_types
                if agent_type.matches_capability(capability)
            ]

        if tag:
            agent_types = [agent_type for agent_type in agent_types if agent_type.matches_tag(tag)]

        if author:
            agent_types = [agent_type for agent_type in agent_types if agent_type.author == author]

        return agent_types

    def create_agent_config(
        self,
        agent_type: str,
        name: str,
        description: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> AgentConfig:
        """Create agent configuration from type."""
        agent_type_info = self.get_agent_type(agent_type)

        # Start with default config
        config_dict = agent_type_info.default_config.copy()

        # Apply overrides
        if overrides:
            config_dict.update(overrides)

        # Set required fields
        config_dict.update(
            {
                "name": name,
                "description": description or agent_type_info.description,
                "capabilities": agent_type_info.capabilities,
            }
        )

        return AgentConfig(**config_dict)

    def create_agent_instance(
        self,
        agent_type: str,
        name: str,
        description: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> BaseAgent:
        """Create an agent instance from type."""
        agent_type_info = self.get_agent_type(agent_type)

        # Create configuration
        config = self.create_agent_config(
            agent_type=agent_type,
            name=name,
            description=description,
            overrides=overrides,
        )

        # Create agent instance
        try:
            agent = agent_type_info.agent_class(config)

            # Register instance
            self._agent_instances[agent.id] = {
                "agent_type": agent_type,
                "name": name,
                "description": description,
                "config": config.model_dump(),
                "created_at": datetime.now(UTC).isoformat(),
                "status": AgentStatus.IDLE.value,
            }

            logger.info(
                "Agent instance created",
                agent_id=agent.id,
                agent_type=agent_type,
                name=name,
            )

            return agent

        except Exception as e:
            logger.error(
                "Failed to create agent instance",
                agent_type=agent_type,
                name=name,
                error=str(e),
            )
            raise AgentError(
                f"Failed to create agent instance: {str(e)}",
                agent_id=name,
                agent_name=name,
            ) from e

    def register_agent_instance(
        self,
        agent: BaseAgent,
        agent_type: str,
    ) -> None:
        """Register an existing agent instance."""
        self._agent_instances[agent.id] = {
            "agent_type": agent_type,
            "name": agent.config.name,
            "description": agent.config.description,
            "config": agent.config.model_dump(),
            "created_at": datetime.now(UTC).isoformat(),
            "status": agent.status.value,
        }

        logger.info(
            "Agent instance registered",
            agent_id=agent.id,
            agent_type=agent_type,
            name=agent.config.name,
        )

    def unregister_agent_instance(self, agent_id: str) -> None:
        """Unregister an agent instance."""
        if agent_id in self._agent_instances:
            del self._agent_instances[agent_id]
            logger.info("Agent instance unregistered", agent_id=agent_id)

    def get_agent_instance_info(self, agent_id: str) -> dict[str, Any]:
        """Get agent instance information."""
        if agent_id not in self._agent_instances:
            raise ResourceNotFoundError(
                f"Agent instance '{agent_id}' not found",
                resource_type="agent_instance",
                resource_id=agent_id,
            )

        return self._agent_instances[agent_id]

    def list_agent_instances(
        self,
        agent_type: str | None = None,
        status: AgentStatus | None = None,
    ) -> list[dict[str, Any]]:
        """List agent instances with optional filtering."""
        instances = list(self._agent_instances.values())

        if agent_type:
            instances = [instance for instance in instances if instance["agent_type"] == agent_type]

        if status:
            instances = [instance for instance in instances if instance["status"] == status.value]

        return instances

    def update_agent_instance_status(
        self,
        agent_id: str,
        status: AgentStatus,
    ) -> None:
        """Update agent instance status."""
        if agent_id in self._agent_instances:
            self._agent_instances[agent_id]["status"] = status.value
            self._agent_instances[agent_id]["updated_at"] = datetime.now(UTC).isoformat()

    def search_agent_types(
        self,
        query: str,
        limit: int = 10,
    ) -> list[AgentTypeInfo]:
        """Search agent types by name, description, or tags."""
        query_lower = query.lower()
        matches = []

        for agent_type in self._agent_types.values():
            score = 0

            # Check name (highest priority)
            if query_lower in agent_type.name.lower():
                score += 10

            # Check description
            if query_lower in agent_type.description.lower():
                score += 5

            # Check tags
            for tag in agent_type.tags:
                if query_lower in tag.lower():
                    score += 3

            # Check capabilities
            for capability in agent_type.capabilities:
                if query_lower in capability.value.lower():
                    score += 2

            if score > 0:
                matches.append((score, agent_type))

        # Sort by score descending and return top results
        matches.sort(key=lambda x: x[0], reverse=True)
        return [agent_type for _, agent_type in matches[:limit]]

    def validate_agent_config(
        self,
        agent_type: str,
        config: dict[str, Any],
    ) -> bool:
        """Validate agent configuration for a type."""
        try:
            agent_type_info = self.get_agent_type(agent_type)

            # Create a test config to validate
            test_config = agent_type_info.default_config.copy()
            test_config.update(config)

            # Validate by creating AgentConfig
            AgentConfig(**test_config)
            return True

        except Exception as e:
            logger.warning(
                "Agent config validation failed",
                agent_type=agent_type,
                config=config,
                error=str(e),
            )
            return False

    def get_registry_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        stats = {
            "total_types": len(self._agent_types),
            "total_instances": len(self._agent_instances),
            "types_by_capability": {},
            "instances_by_type": {},
            "instances_by_status": {},
        }

        # Count types by capability
        for agent_type in self._agent_types.values():
            for capability in agent_type.capabilities:
                cap_name = capability.value
                stats["types_by_capability"][cap_name] = (
                    stats["types_by_capability"].get(cap_name, 0) + 1
                )

        # Count instances by type and status
        for instance in self._agent_instances.values():
            agent_type = instance["agent_type"]
            status = instance["status"]

            stats["instances_by_type"][agent_type] = (
                stats["instances_by_type"].get(agent_type, 0) + 1
            )
            stats["instances_by_status"][status] = stats["instances_by_status"].get(status, 0) + 1

        return stats

    def _register_builtin_types(self) -> None:
        """Register built-in agent types."""
        # This will be populated with actual agent implementations
        logger.info("Built-in agent types registered")

    def _load_from_storage(self) -> None:
        """Load registry from storage."""
        try:
            registry_file = self.storage_path / "registry.json"
            if registry_file.exists():
                with open(registry_file, encoding='utf-8') as f:
                    data = json.load(f)

                # Load agent types (metadata only, not classes)
                for type_data in data.get("agent_types", []):
                    # Skip loading classes from storage - they need to be registered at runtime
                    pass

                # Load agent instances
                self._agent_instances = data.get("agent_instances", {})

                logger.info(
                    "Registry loaded from storage",
                    types_count=len(self._agent_types),
                    instances_count=len(self._agent_instances),
                )
        except Exception as e:
            logger.warning("Failed to load registry from storage", error=str(e))

    def _save_to_storage(self) -> None:
        """Save registry to storage."""
        try:
            registry_file = self.storage_path / "registry.json"

            data = {
                "agent_types": [agent_type.to_dict() for agent_type in self._agent_types.values()],
                "agent_instances": self._agent_instances,
                "updated_at": datetime.now(UTC).isoformat(),
            }

            with open(registry_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug("Registry saved to storage")
        except Exception as e:
            logger.error("Failed to save registry to storage", error=str(e))

    def export_registry(self, export_path: Path) -> None:
        """Export registry to file."""
        try:
            data = {
                "agent_types": [agent_type.to_dict() for agent_type in self._agent_types.values()],
                "agent_instances": self._agent_instances,
                "stats": self.get_registry_stats(),
                "exported_at": datetime.now(UTC).isoformat(),
            }

            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info("Registry exported", export_path=str(export_path))
        except Exception as e:
            logger.error("Failed to export registry", error=str(e))
            raise

    def clear_registry(self) -> None:
        """Clear all registry data (use with caution)."""
        self._agent_types.clear()
        self._agent_instances.clear()
        self._save_to_storage()
        logger.warning("Registry cleared")


# Global registry instance
_global_registry: AgentRegistry | None = None


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry


def register_agent_type(
    name: str,
    description: str,
    agent_class: type[BaseAgent],
    capabilities: list[AgentCapability],
    default_config: dict[str, Any] | None = None,
    version: str = "1.0.0",
    author: str = "OpenGenAI",
    tags: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """Register an agent type in the global registry."""
    registry = get_agent_registry()
    registry.register_agent_type(
        name=name,
        description=description,
        agent_class=agent_class,
        capabilities=capabilities,
        default_config=default_config,
        version=version,
        author=author,
        tags=tags,
        overwrite=overwrite,
    )


def create_agent(
    agent_type: str,
    name: str,
    description: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> BaseAgent:
    """Create an agent instance from the global registry."""
    registry = get_agent_registry()
    return registry.create_agent_instance(
        agent_type=agent_type,
        name=name,
        description=description,
        overrides=overrides,
    )
