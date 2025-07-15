"""
Unit tests for OpenGenAI agents functionality.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
import respx
from httpx import Response

from opengenai.agents.base import BaseAgent
from opengenai.agents.manager import AgentManager, AgentRegistry
from opengenai.agents.models import AgentStatus, MessageType, AgentMessage


@pytest.fixture
def registry() -> AgentRegistry:
    """Provides an empty agent registry."""
    return AgentRegistry()


@pytest.fixture
def manager(registry: AgentRegistry) -> AgentManager:
    """Provides an agent manager with a clean registry."""
    return AgentManager(registry=registry)


@pytest.fixture
def agent() -> BaseAgent:
    """Provides a basic agent instance."""
    return BaseAgent(id="test_agent", name="Test Agent", description="A test agent")


@pytest.mark.asyncio
async def test_agent_registration(
    registry: AgentRegistry, agent: BaseAgent
) -> None:
    """Verify that an agent can be registered and retrieved."""
    registry.register_agent(agent)
    retrieved = registry.get("test_agent")
    assert retrieved is agent

    registry.unregister_agent("test_agent")
    with pytest.raises(KeyError):
        registry.get("test_agent")


@pytest.mark.asyncio
async def test_manager_start_stop(manager: AgentManager, agent: BaseAgent) -> None:
    """Ensure the manager can start and stop an agent."""
    await manager.start(agent)
    assert manager.registry.get("test_agent").status == AgentStatus.RUNNING

    await manager.stop("test_agent")
    assert agent.status == AgentStatus.TERMINATED
    with pytest.raises(KeyError):
        manager.registry.get("test_agent")


@respx.mock
@pytest.mark.asyncio
async def test_agent_step_openai_call(agent: BaseAgent) -> None:
    """Test that an agent's step calls the OpenAI API correctly."""
    # Mock the OpenAI API endpoint
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello! I am a test agent.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )

    # Create a valid message to send
    message = AgentMessage(type=MessageType.USER, content="Test prompt")

    # The agent should now be able to execute a step
    reply = await agent.step(message)
    assert reply.type == MessageType.ASSISTANT
    assert "Hello! I am a test agent." in reply.content


@respx.mock
@pytest.mark.asyncio
async def test_manager_chat(manager: AgentManager, agent: BaseAgent) -> None:
    """Verify the manager's chat function orchestrates an agent interaction."""
    await manager.start(agent)

    # Mock the API call that the agent's `step` method will make
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "You said: 'Hello, world!'"},
                    }
                ]
            },
        )
    )

    response = await manager.chat("test_agent", "Hello, world!")
    assert response == "You said: 'Hello, world!'"
    # Check that logs were recorded
    assert len(manager._logs["test_agent"]) == 2 