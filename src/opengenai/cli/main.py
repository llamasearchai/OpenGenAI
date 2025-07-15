import asyncio
from pathlib import Path

import typer

from opengenai.agents.base import BaseAgent
from opengenai.agents.manager import AgentManager

cli = typer.Typer()


@cli.command()
def run(outfile: str | None = None):
    """Quick interactive REPL for a single agent."""

    async def _inner():
        agent = BaseAgent(
            id="demo", name="DemoAgent", description="A minimal demo agent."
        )
        manager = AgentManager()
        await manager.start(agent)

        out_path = Path(outfile) if outfile else Path.cwd() / "transcript.txt"
        with out_path.open("w") as fp:
            while True:
                try:
                    prompt = input("> ")
                except (EOFError, KeyboardInterrupt):
                    break
                reply = await manager.chat(agent.id, prompt)
                print(reply)
                fp.write(f"User: {prompt}\nAgent: {reply}\n\n")

        await manager.stop(agent.id)

    asyncio.run(_inner())


if __name__ == "__main__":
    cli()
