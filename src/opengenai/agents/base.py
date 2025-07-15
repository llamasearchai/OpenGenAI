"""
Base Agent Implementation for OpenGenAI
Provides the foundation for all AI agents in the system.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from opengenai.agents.models import (
    AgentMemory,
    AgentMessage,
    AgentState,
    AgentStatus,
    MemoryType,
    MessageType,
    to_openai_messages,
)
from opengenai.core.config import settings
from opengenai.core.performance import PerformanceLogger

logger = logging.getLogger(__name__)
openai_client = AsyncOpenAI(api_key=settings.openai.api_key.get_secret_value())


# --------------------------------------------------------------------------- #
#  PUBLIC BASE-CLASS
# --------------------------------------------------------------------------- #


class BaseAgent(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    state: AgentState = Field(default_factory=AgentState)
    status: AgentStatus = AgentStatus.INITIALIZING
    last_error: str | None = None
    memory: List[AgentMemory] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}

    # --------------------- life-cycle ----------------------------------------

    async def start(self) -> None:
        logger.info("[%s] starting …", self.id)
        self.status = AgentStatus.RUNNING
        await self._bootstrap()

    async def _bootstrap(self) -> None:
        """Override for expensive async initialisation (e.g. load embeddings)."""
        await asyncio.sleep(0.0)

    async def stop(self) -> None:
        logger.info("[%s] terminating", self.id)
        self.status = AgentStatus.TERMINATED

    # --------------------- memory helpers -----------------------------------

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        importance: float = 0.5,
        **meta: Any,
    ) -> None:
        self.memory.append(
            AgentMemory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                input_data=meta.get("input_data"),
                output_data=meta.get("output_data"),
                context=meta.get("context"),
                iteration=self.state.iteration_count,
            )
        )

    # --------------------- core loop ----------------------------------------

    async def step(
        self, *messages: AgentMessage, timeout: Optional[int] = None
    ) -> AgentMessage:
        """Single iteration: called by AgentManager."""
        self.state.iteration_count += 1
        self.state.last_activity = datetime.utcnow()

        request_timeout = timeout or settings.agent.agent_timeout

        with PerformanceLogger(
            f"agent:{self.id}:iteration:{self.state.iteration_count}"
        ):
            try:
                response = await openai_client.chat.completions.create(
                    model=settings.openai.model,
                    temperature=settings.openai.temperature,
                    messages=to_openai_messages(messages),
                    max_tokens=1024,
                    timeout=request_timeout,
                )
            except Exception as exc:
                self.status = AgentStatus.FAILED
                self.last_error = str(exc)
                logger.exception("[%s] OpenAI call failed: %s", self.id, exc)
                raise

        content = response.choices[0].message.content
        if content is None:
            content = ""

        if messages:
            correlation_id = messages[-1].correlation_id
            reply = AgentMessage(
                type=MessageType.ASSISTANT,
                content=content,
                correlation_id=correlation_id,
                sender_id=self.id,
            )
        else:
            reply = AgentMessage(
                type=MessageType.ASSISTANT, content=content, sender_id=self.id
            )

        self.remember(
            content, MemoryType.SHORT_TERM, 0.1, output_data={"raw": content}
        )
        return reply
