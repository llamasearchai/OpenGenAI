"""
OpenGenAI Logging Configuration
Provides structured logging with context management and performance tracking.
"""

import json
import logging
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from structlog.processors import JSONRenderer
from structlog.stdlib import LoggerFactory

from opengenai.core.types import LogLevel

# Global logger configuration
_logger_configured = False
_current_agent_id: str | None = None
_current_request_id: str | None = None


def setup_logging(
    level: LogLevel = LogLevel.INFO,
    format_type: str = "human",
    log_file: Path | None = None,
    enable_performance: bool = True,
) -> None:
    """Setup structured logging configuration."""
    global _logger_configured

    if _logger_configured:
        return

    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        add_context_processor,
    ]

    # Add performance processor if enabled
    if enable_performance:
        processors.append(PerformanceProcessor())

    # Configure output format
    if format_type == "json":
        processors.append(JSONRenderer())
        formatter = JSONFormatter()
    else:
        processors.append(structlog.dev.ConsoleRenderer())
        formatter = HumanReadableFormatter()

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    root_logger = logging.getLogger()
    root_logger.setLevel(level.value)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _logger_configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def set_agent_id(agent_id: str) -> None:
    """Set the current agent ID for logging context."""
    global _current_agent_id
    _current_agent_id = agent_id


def set_request_id(request_id: str) -> None:
    """Set the current request ID for logging context."""
    global _current_request_id
    _current_request_id = request_id


def get_agent_id() -> str | None:
    """Get the current agent ID."""
    return _current_agent_id


def get_request_id() -> str | None:
    """Get the current request ID."""
    return _current_request_id


def add_context_processor(_, __, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Add context information to log events."""
    if _current_agent_id:
        event_dict["agent_id"] = _current_agent_id
    if _current_request_id:
        event_dict["request_id"] = _current_request_id
    return event_dict


class TimestampProcessor:
    """Processor to add timestamps to log records."""

    def __call__(self, _, __, event_dict: dict[str, Any]) -> dict[str, Any]:
        """Add timestamp to event dict."""
        event_dict["timestamp"] = datetime.now(UTC).isoformat()
        return event_dict


class PerformanceProcessor:
    """Processor to add performance metrics to log records."""

    def __init__(self):
        """Initialize performance processor."""
        self.start_times: dict[str, float] = {}

    def __call__(self, _, __, event_dict: dict[str, Any]) -> dict[str, Any]:
        """Add performance metrics to event dict."""
        # Add memory usage
        try:
            import psutil

            process = psutil.Process()
            event_dict["memory_usage_mb"] = process.memory_info().rss / 1024 / 1024
            event_dict["cpu_percent"] = process.cpu_percent()
        except ImportError:
            pass

        return event_dict


class LoggingContext:
    """Context manager for structured logging."""

    def __init__(self, **context):
        """Initialize logging context."""
        self.context = context
        self.original_context = {}

    def __enter__(self):
        """Enter logging context."""
        # Store original values
        self.original_context = {
            "agent_id": _current_agent_id,
            "request_id": _current_request_id,
        }

        # Set new context
        if "agent_id" in self.context:
            set_agent_id(self.context["agent_id"])
        if "request_id" in self.context:
            set_request_id(self.context["request_id"])

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit logging context."""
        # Restore original values
        global _current_agent_id, _current_request_id
        _current_agent_id = self.original_context["agent_id"]
        _current_request_id = self.original_context["request_id"]


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Extract structured data from record
        event_dict = getattr(record, "event_dict", {})
        if not event_dict:
            event_dict = {
                "message": record.getMessage(),
                "level": record.levelname,
                "timestamp": datetime.now(UTC).isoformat(),
                "logger": record.name,
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }

        # Add exception information if present
        if record.exc_info:
            event_dict["exception"] = self.formatException(record.exc_info)

        return json.dumps(event_dict, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human reading."""
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        # Extract structured data
        event_dict = getattr(record, "event_dict", {})
        if event_dict:
            message = event_dict.get("message", str(record.getMessage()))

            # Build context string
            context_parts = []
            for key, value in event_dict.items():
                if key not in ["message", "level", "timestamp", "logger"]:
                    context_parts.append(f"{key}={value}")

            context_str = f" [{', '.join(context_parts)}]" if context_parts else ""
        else:
            message = record.getMessage()
            context_str = ""

        # Format final message
        formatted = f"[{timestamp}] {record.levelname:8} {record.name}: {message}{context_str}"

        # Add exception information if present
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


class PerformanceLogger:
    """Logger for performance metrics and timing."""

    def __init__(self, name: str):
        """Initialize performance logger."""
        self.name = name
        self.logger = get_logger(f"opengenai.performance.{name}")
        self.timers: dict[str, float] = {}

    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.timers[operation] = time.perf_counter()
        self.logger.debug(f"Started timing {operation}")

    def end_timer(self, operation: str, **context) -> float:
        """End timing an operation and log the duration."""
        if operation not in self.timers:
            self.logger.warning(f"No timer found for operation: {operation}")
            return 0.0

        duration = time.perf_counter() - self.timers[operation]
        del self.timers[operation]

        self.logger.info(
            f"Operation completed: {operation}",
            operation=operation,
            duration_ms=duration * 1000,
            **context,
        )

        return duration

    def log_metric(self, metric_name: str, value: int | float, **context) -> None:
        """Log a performance metric."""
        self.logger.info(f"Metric: {metric_name}", metric_name=metric_name, value=value, **context)

    def log_counter(self, counter_name: str, increment: int = 1, **context) -> None:
        """Log a counter increment."""
        self.logger.info(
            f"Counter: {counter_name}", counter_name=counter_name, increment=increment, **context
        )


class AsyncPerformanceTimer:
    """Async context manager for timing operations."""

    def __init__(self, logger: PerformanceLogger, operation: str, **context):
        """Initialize async performance timer."""
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None

    async def __aenter__(self):
        """Enter async context."""
        self.start_time = time.perf_counter()
        self.logger.logger.debug(f"Started timing {self.operation}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time

            # Add exception info if present
            if exc_type:
                self.context["exception"] = str(exc_val)
                self.context["exception_type"] = exc_type.__name__

            self.logger.logger.info(
                f"Async operation completed: {self.operation}",
                operation=self.operation,
                duration_ms=duration * 1000,
                **self.context,
            )


def async_performance_timer(
    operation: str, logger: PerformanceLogger | None = None, **context
) -> AsyncPerformanceTimer:
    """Create an async performance timer."""
    if logger is None:
        logger = PerformanceLogger("default")

    return AsyncPerformanceTimer(logger, operation, **context)


class AuditLogger:
    """Logger for audit events."""

    def __init__(self):
        """Initialize audit logger."""
        self.logger = get_logger("opengenai.audit")

    def log_user_action(
        self,
        user_id: str,
        action: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
        **context,
    ) -> None:
        """Log a user action."""
        self.logger.info(
            f"User action: {action}",
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            **context,
        )

    def log_system_event(
        self, event_type: str, description: str, severity: str = "info", **context
    ) -> None:
        """Log a system event."""
        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method(
            f"System event: {event_type}",
            event_type=event_type,
            description=description,
            severity=severity,
            **context,
        )

    def log_security_event(
        self, event_type: str, user_id: str | None = None, ip_address: str | None = None, **context
    ) -> None:
        """Log a security event."""
        self.logger.warning(
            f"Security event: {event_type}",
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            **context,
        )


class RequestLogger:
    """Logger for HTTP requests."""

    def __init__(self):
        """Initialize request logger."""
        self.logger = get_logger("opengenai.requests")

    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        user_id: str | None = None,
        **context,
    ) -> None:
        """Log an HTTP request."""
        self.logger.info(
            f"{method} {path} {status_code}",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            user_id=user_id,
            **context,
        )


class ErrorLogger:
    """Logger for errors and exceptions."""

    def __init__(self):
        """Initialize error logger."""
        self.logger = get_logger("opengenai.errors")

    def log_error(self, error: Exception, context: dict[str, Any] | None = None, **kwargs) -> None:
        """Log an error with context."""
        error_context = context or {}
        error_context.update(kwargs)

        self.logger.error(
            f"Error occurred: {type(error).__name__}",
            error_type=type(error).__name__,
            error_message=str(error),
            **error_context,
            exc_info=True,
        )

    def log_warning(self, message: str, context: dict[str, Any] | None = None, **kwargs) -> None:
        """Log a warning with context."""
        warning_context = context or {}
        warning_context.update(kwargs)

        self.logger.warning(message, **warning_context)


# Global logger instances
audit_logger = AuditLogger()
request_logger = RequestLogger()
error_logger = ErrorLogger()


def log_function_call(func: Callable) -> Callable:
    """Decorator to log function calls."""

    def wrapper(*args, **kwargs):
        logger = get_logger(f"opengenai.functions.{func.__name__}")

        logger.debug(
            f"Function called: {func.__name__}",
            function=func.__name__,
            args=len(args),
            kwargs=list(kwargs.keys()),
        )

        try:
            result = func(*args, **kwargs)
            logger.debug(
                f"Function completed: {func.__name__}", function=func.__name__, success=True
            )
            return result
        except Exception as e:
            logger.error(
                f"Function failed: {func.__name__}",
                function=func.__name__,
                error=str(e),
                exc_info=True,
            )
            raise

    return wrapper


def log_async_function_call(func: Callable[..., Awaitable]) -> Callable[..., Awaitable]:
    """Decorator to log async function calls."""

    async def wrapper(*args, **kwargs):
        logger = get_logger(f"opengenai.functions.{func.__name__}")

        logger.debug(
            f"Async function called: {func.__name__}",
            function=func.__name__,
            args=len(args),
            kwargs=list(kwargs.keys()),
        )

        try:
            result = await func(*args, **kwargs)
            logger.debug(
                f"Async function completed: {func.__name__}", function=func.__name__, success=True
            )
            return result
        except Exception as e:
            logger.error(
                f"Async function failed: {func.__name__}",
                function=func.__name__,
                error=str(e),
                exc_info=True,
            )
            raise

    return wrapper


class MetricsCollector:
    """Collector for application metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.logger = get_logger("opengenai.metrics")
        self.counters: dict[str, int] = {}
        self.gauges: dict[str, float] = {}
        self.histograms: dict[str, list[float]] = {}

    def increment_counter(self, name: str, value: int = 1, **tags) -> None:
        """Increment a counter metric."""
        self.counters[name] = self.counters.get(name, 0) + value
        self.logger.debug(
            f"Counter incremented: {name}",
            metric_name=name,
            value=value,
            total=self.counters[name],
            **tags,
        )

    def set_gauge(self, name: str, value: float, **tags) -> None:
        """Set a gauge metric."""
        self.gauges[name] = value
        self.logger.debug(f"Gauge set: {name}", metric_name=name, value=value, **tags)

    def record_histogram(self, name: str, value: float, **tags) -> None:
        """Record a histogram value."""
        if name not in self.histograms:
            self.histograms[name] = []
        self.histograms[name].append(value)

        self.logger.debug(f"Histogram recorded: {name}", metric_name=name, value=value, **tags)

    def get_metrics(self) -> dict[str, Any]:
        """Get all collected metrics."""
        return {
            "counters": self.counters.copy(),
            "gauges": self.gauges.copy(),
            "histograms": {
                name: {
                    "count": len(values),
                    "sum": sum(values),
                    "avg": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                }
                for name, values in self.histograms.items()
            },
        }


# Global metrics collector
metrics_collector = MetricsCollector()


class HealthCheckLogger:
    """Logger for health check events."""

    def __init__(self):
        """Initialize health check logger."""
        self.logger = get_logger("opengenai.health")
        self.start_time = datetime.now(UTC)

    def log_health_check(
        self, component: str, status: str, duration_ms: float | None = None, **context
    ) -> None:
        """Log a health check result."""
        end_time = datetime.now(UTC)

        self.logger.info(
            f"Health check: {component} - {status}",
            component=component,
            status=status,
            duration_ms=duration_ms,
            **context,
        )

    def get_uptime(self) -> float:
        """Get application uptime in seconds."""
        return (datetime.now(UTC) - self.start_time).total_seconds()


# Global health check logger
health_logger = HealthCheckLogger()
