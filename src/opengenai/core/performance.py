from __future__ import annotations

import logging
import time
from types import TracebackType
from typing import Optional, Type

logger = logging.getLogger(__name__)


class PerformanceLogger:
    """Context-manager that logs wall-clock and CPU time."""

    def __init__(self, section: str) -> None:
        self.section = section
        self._start: float = 0.0

    # --- context-manager protocol ------------------------------------------------
    def __enter__(self) -> "PerformanceLogger":
        self._start = time.perf_counter()
        logger.debug("[PERFORMANCE] %s started", self.section)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> bool:  # noqa: D401
        duration = time.perf_counter() - self._start
        if exc:
            logger.exception("%s failed after %.3fs: %s", self.section, duration, exc)
        else:
            logger.info("%s finished in %.3fs", self.section, duration)
        # propagate exception if any
        return False 