from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Dict, List, Optional

from opengenai.agents.base import BaseAgent
from opengenai.agents.models import AgentMessage, AgentStatus, MessageType

logger = logging.getLogger(__name__)


class AgentRegistry:
    """In-memory registry; swap for Redis etc. if desired."""

    def __init__(self) -> None:
        self._agents: Dict[str, BaseAgent] = {}

    # ----------------------- CRUD -------------------------------------------

    def register_agent(self, agent: BaseAgent) -> None:
        if agent.id in self._agents:
            raise ValueError(f"Agent {agent.id} already registered")
        self._agents[agent.id] = agent

    def unregister_agent(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)

    def get(self, agent_id: str) -> BaseAgent:
        return self._agents[agent_id]

    # ----------------------- broadcast --------------------------------------

    async def broadcast(self, *messages: AgentMessage) -> None:
        await asyncio.gather(
            *(
                a.step(*messages)
                for a in self._agents.values()
                if a.status == AgentStatus.RUNNING
            )
        )


class AgentManager:
    """High-level orchestration of tasks & agents."""

    def __init__(self, registry: AgentRegistry | None = None) -> None:
        self.registry = registry or AgentRegistry()
        self._tasks: Dict[str, asyncio.Task] = {}
        self._logs: Dict[str, List[AgentMessage]] = defaultdict(list)

    # --------------------------------------------------------------------- #

    async def start(self, agent: BaseAgent) -> None:
        await agent.start()
        self.registry.register_agent(agent)

    async def stop(self, agent_id: str) -> None:
        agent = self.registry.get(agent_id)
        await agent.stop()
        self.registry.unregister_agent(agent_id)

    # --------------------------------------------------------------------- #

    async def chat(
        self,
        agent_id: str,
        content: str,
        *,
        recipient_id: Optional[str] = None,
        **meta: str,
    ) -> str:
        agent = self.registry.get(agent_id)
        user_msg = AgentMessage(
            type=MessageType.USER,
            content=content,
            recipient_id=recipient_id,
            metadata=meta,
        )
        reply = await agent.step(user_msg)
        self._logs[agent_id].append(user_msg)
        self._logs[agent_id].append(reply)
        return reply.content
