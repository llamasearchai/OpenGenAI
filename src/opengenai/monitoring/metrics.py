"""OpenGenAI Monitoring - Metrics

Provides Prometheus metrics registry and helper utilities.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

# ---------------------------------------------------------------------------
# Prometheus registry and default metrics
# ---------------------------------------------------------------------------
REGISTRY = CollectorRegistry(auto_describe=True)

HTTP_REQUESTS_TOTAL = Counter(
    "opengenai_http_requests_total",
    "Total number of HTTP requests processed.",
    ["method", "path", "status"],
    registry=REGISTRY,
)
HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "opengenai_http_request_latency_seconds",
    "Histogram of HTTP request latency.",
    ["method", "path"],
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def setup_metrics() -> None:
    """Expose /metrics endpoint via FastAPI router and return router for inclusion."""
    # Nothing else required; FastAPI app can include get_metrics_router().


def get_metrics_router(path: str = "/metrics") -> APIRouter:
    """Return a FastAPI router that serves Prometheus metrics."""
    router = APIRouter()

    @router.get(path)
    def metrics_endpoint() -> Response:  # type: ignore[valid-type]
        data = generate_latest(REGISTRY)
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    return router


__all__: list[str] = [
    "REGISTRY",
    "HTTP_REQUESTS_TOTAL",
    "HTTP_REQUEST_LATENCY_SECONDS",
    "setup_metrics",
    "get_metrics_router",
]
