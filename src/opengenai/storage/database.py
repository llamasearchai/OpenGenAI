"""OpenGenAI Storage Layer - Database

Provides SQLAlchemy Async engine, Base metadata and helper methods to initialise and dispose the database connection.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TypeVar

from sqlalchemy.ext.asyncio import (  # type: ignore
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeMeta, declarative_base  # type: ignore

from opengenai.core.config import settings
from opengenai.core.logging import get_logger

logger = get_logger(__name__)

Base: DeclarativeMeta = declarative_base()
_engine: AsyncEngine | None = None
AsyncSessionLocal: async_sessionmaker[AsyncSession] | None = None


async def init_db() -> None:
    """Initialise the global SQLAlchemy async engine and session factory."""
    global _engine, AsyncSessionLocal
    if _engine is not None:
        return

    database_url = settings.get_database_url(async_driver=True)
    logger.info("Initialising database engine", url=database_url)
    _engine = create_async_engine(
        database_url,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout,
        echo=settings.database.echo,
    )
    AsyncSessionLocal = async_sessionmaker(_engine, expire_on_commit=False)

    # Create tables if using in-memory SQLite (mainly for tests)
    if settings.database.is_sqlite:
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session for FastAPI dependencies (FastAPI dependency)."""
    if AsyncSessionLocal is None:
        await init_db()
    assert AsyncSessionLocal is not None  # nosec
    async with AsyncSessionLocal() as session:  # type: ignore[arg-type]
        yield session


async def close_db() -> None:
    """Dispose the database engine."""
    global _engine
    if _engine is None:
        return
    logger.info("Disposing database engine")
    await _engine.dispose()
    _engine = None


# Convenience for non-FastAPI usage
T = TypeVar("T")


async def run_in_transaction(coro: Callable[[AsyncSession], Awaitable[T]]) -> T:  # type: ignore[name-match]
    """Run coroutine within a transaction and commit/rollback automatically."""
    if AsyncSessionLocal is None:
        await init_db()
    assert AsyncSessionLocal is not None  # nosec
    async with AsyncSessionLocal() as session:  # type: ignore[arg-type]
        try:
            async with session.begin():
                return await coro(session)
        except Exception:
            await session.rollback()
            raise
