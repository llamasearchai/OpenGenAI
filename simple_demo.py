"""
Simple Demo for OpenGenAI
Demonstrates basic functionality of the AI agent platform.
"""

import asyncio

from opengenai.agents.base import BaseAgent
from opengenai.agents.manager import AgentManager
from opengenai.agents.models import AgentMessage, MessageType


async def main():
    """Demonstrates creating an agent and having a simple chat."""
    print("[SYSTEM] Initializing agent manager and creating a demo agent...")
    manager = AgentManager()
    agent = BaseAgent(id="demo_agent", name="Demo Agent", description="A simple test agent")

    await manager.start(agent)
    print(f"[SYSTEM] Agent '{agent.name}' ({agent.id}) has started.")

    print("\n[SYSTEM] Starting interactive chat. Type 'exit' to end.")
    while True:
        try:
            user_input = input("[USER] > ")
            if user_input.lower() == "exit":
                break

            response_content = await manager.chat(agent.id, user_input)
            print(f"[{agent.name}] > {response_content}")

        except (KeyboardInterrupt, EOFError):
            break

    print("\n[SYSTEM] Shutting down agent...")
    await manager.stop(agent.id)
    print("[SYSTEM] Demo complete.")


if __name__ == "__main__":
    asyncio.run(main()) 