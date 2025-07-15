"""OpenGenAI Monitoring - Tracing

Provides helper to configure OpenTelemetry tracing for FastAPI.
"""

from __future__ import annotations

from opentelemetry import trace  # type: ignore
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore
from opentelemetry.sdk.resources import SERVICE_NAME, Resource  # type: ignore
from opentelemetry.sdk.trace import TracerProvider  # type: ignore
from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore

from opengenai.core.config import settings
from opengenai.core.logging import get_logger

logger = get_logger(__name__)

_tracer_provider_initialized = False


def setup_tracing() -> None:
    """Configure OpenTelemetry TracerProvider if not already set."""
    global _tracer_provider_initialized
    if _tracer_provider_initialized or not settings.monitoring.tracing_enabled:
        return

    resource = Resource(attributes={SERVICE_NAME: "opengenai"})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    exporter = OTLPSpanExporter(endpoint=settings.monitoring.jaeger_endpoint or "")
    provider.add_span_processor(BatchSpanProcessor(exporter))

    _tracer_provider_initialized = True
    logger.info("OpenTelemetry tracing configured")


def get_tracer(name: str = "opengenai"):
    """Return a tracer instance."""
    setup_tracing()
    return trace.get_tracer(name)


__all__ = ["setup_tracing", "get_tracer"]
