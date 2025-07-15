"""
OpenGenAI FastAPI Application
Main application setup and configuration.
"""

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from opengenai.agents.manager import AgentManager

app = FastAPI(title="OpenGenAI")

manager = AgentManager()


# … endpoints for CRUD omitted …


@app.exception_handler(ValidationError)
async def pydantic_validation_handler(_, exc: ValidationError):
    """Return readable error format instead of accessing .field/.value."""
    errors = [
        {"loc": " → ".join(str(p) for p in err["loc"]), "msg": err["msg"]}
        for err in exc.errors()
    ]
    raise HTTPException(status_code=422, detail=errors)
