"""OpenGenAI API Middleware

Contains HTTP middleware classes for logging, rate-limiting, security headers and Prometheus metrics collection.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Dict, Optional

from fastapi import Request, Response
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from opengenai.core.logging import get_logger
from opengenai.core.security import RateLimiter, SecurityHeaders

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Metrics objects (shared between workers)
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "opengenai_request_total",
    "Total number of HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "opengenai_request_latency_seconds",
    "Latency of HTTP requests in seconds",
    ["method", "path"],
)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Structured request / response logging middleware."""

    def __init__(self, app: ASGIApp) -> None:  # noqa: D401
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:  # type: ignore[override]
        start_time = time.perf_counter()
        response: Response
        path = request.url.path
        method = request.method
        client_ip = request.client.host if request.client else "unknown"
        logger.info("HTTP request started", method=method, path=path, client_ip=client_ip)
        try:
            response = await call_next(request)
        finally:
            duration = time.perf_counter() - start_time
            logger.info(
                "HTTP request completed",
                method=method,
                path=path,
                status_code=getattr(response, "status_code", 500),
                duration_ms=round(duration * 1000, 2),
            )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """IP-based rate-limiting using in-memory RateLimiter from core.security."""

    def __init__(
        self, app: ASGIApp, max_requests: int = 1000, window_seconds: int = 60
    ) -> None:  # noqa: D401
        super().__init__(app)
        self.limiter = RateLimiter(max_requests=max_requests, window_seconds=window_seconds)

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:  # type: ignore[override]
        identifier = request.client.host if request.client else "unknown"
        if not self.limiter.is_allowed(identifier):
            logger.warning("Rate limit exceeded", client_ip=identifier, path=request.url.path)
            return Response(
                content="Rate limit exceeded", status_code=429, media_type="application/json"
            )
        return await call_next(request)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Adds industry-standard security headers to every response."""

    def __init__(self, app: ASGIApp) -> None:  # noqa: D401
        super().__init__(app)
        self.headers: dict[str, str] = SecurityHeaders.get_security_headers()

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:  # type: ignore[override]
        response = await call_next(request)
        for header, value in self.headers.items():
            response.headers.setdefault(header, value)
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collects Prometheus metrics for each HTTP request."""

    def __init__(self, app: ASGIApp) -> None:  # noqa: D401
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:  # type: ignore[override]
        method = request.method
        path = request.url.path
        start_time = time.perf_counter()
        response = await call_next(request)
        latency = time.perf_counter() - start_time
        status = str(response.status_code)
        REQUEST_COUNT.labels(method=method, path=path, status=status).inc()
        REQUEST_LATENCY.labels(method=method, path=path).observe(latency)
        return response


__all__ = [
    "LoggingMiddleware",
    "RateLimitMiddleware",
    "SecurityMiddleware",
    "MetricsMiddleware",
]
