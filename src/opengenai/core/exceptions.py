"""
OpenGenAI Exceptions
Comprehensive exception handling for the OpenGenAI system.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any


class OpenGenAIException(Exception):
    """Base exception class for OpenGenAI."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        """Initialize exception with detailed information."""
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary format."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None,
        }

    def __str__(self) -> str:
        """String representation of the exception."""
        return f"{self.error_code}: {self.message}"


class ValidationError(OpenGenAIException):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        field_value: Any | None = None,
        validation_rules: list[str] | None = None,
        **kwargs,
    ):
        """Initialize validation error."""
        details = {
            "field_name": field_name,
            "field_value": field_value,
            "validation_rules": validation_rules or [],
        }
        super().__init__(message, details=details, **kwargs)


class AuthenticationError(OpenGenAIException):
    """Raised when authentication fails."""

    def __init__(
        self, message: str, auth_method: str | None = None, user_id: str | None = None, **kwargs
    ):
        """Initialize authentication error."""
        details = {
            "auth_method": auth_method,
            "user_id": user_id,
        }
        super().__init__(message, details=details, **kwargs)


class AuthorizationError(OpenGenAIException):
    """Raised when authorization fails."""

    def __init__(
        self,
        message: str,
        user_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        required_permissions: list[str] | None = None,
        **kwargs,
    ):
        """Initialize authorization error."""
        details = {
            "user_id": user_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "required_permissions": required_permissions or [],
        }
        super().__init__(message, details=details, **kwargs)


class ResourceNotFoundError(OpenGenAIException):
    """Raised when a requested resource is not found."""

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
        **kwargs,
    ):
        """Initialize resource not found error."""
        details = {
            "resource_type": resource_type,
            "resource_id": resource_id,
        }
        super().__init__(message, details=details, **kwargs)


class ResourceAlreadyExistsError(OpenGenAIException):
    """Raised when trying to create a resource that already exists."""

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
        **kwargs,
    ):
        """Initialize resource already exists error."""
        details = {
            "resource_type": resource_type,
            "resource_id": resource_id,
        }
        super().__init__(message, details=details, **kwargs)


class ResourceError(OpenGenAIException):
    """Raised when a resource-related error occurs."""

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        resource_limit: str | None = None,
        current_usage: str | None = None,
        **kwargs,
    ):
        """Initialize resource error."""
        details = {
            "resource_type": resource_type,
            "resource_limit": resource_limit,
            "current_usage": current_usage,
        }
        super().__init__(message, details=details, **kwargs)


class RateLimitError(OpenGenAIException):
    """Raised when rate limits are exceeded."""

    def __init__(
        self,
        message: str,
        rate_limit: int | None = None,
        current_usage: int | None = None,
        reset_time: datetime | None = None,
        **kwargs,
    ):
        """Initialize rate limit error."""
        details = {
            "rate_limit": rate_limit,
            "current_usage": current_usage,
            "reset_time": reset_time.isoformat() if reset_time else None,
        }
        super().__init__(message, details=details, **kwargs)


class AgentError(OpenGenAIException):
    """Base class for agent-related errors."""

    def __init__(
        self, message: str, agent_id: str | None = None, agent_name: str | None = None, **kwargs
    ):
        """Initialize agent error."""
        details = {
            "agent_id": agent_id,
            "agent_name": agent_name,
        }
        super().__init__(message, details=details, **kwargs)


class AgentNotFoundError(AgentError):
    """Raised when an agent is not found."""

    def __init__(self, agent_id: str, **kwargs):
        """Initialize agent not found error."""
        super().__init__(f"Agent with ID '{agent_id}' not found", agent_id=agent_id, **kwargs)


class AgentInitializationError(AgentError):
    """Raised when agent initialization fails."""

    pass


class AgentTimeoutError(AgentError):
    """Raised when agent operation times out."""

    def __init__(
        self,
        message: str,
        agent_id: str | None = None,
        timeout_seconds: int | None = None,
        **kwargs,
    ):
        """Initialize agent timeout error."""
        details = {
            "agent_id": agent_id,
            "timeout_seconds": timeout_seconds,
        }
        super().__init__(message, details=details, **kwargs)


class AgentMemoryError(AgentError):
    """Raised when agent memory operations fail."""

    pass


class AgentCommunicationError(AgentError):
    """Raised when agent communication fails."""

    pass


class AgentReflectionError(AgentError):
    """Raised when agent reflection operations fail."""

    pass


class TaskError(OpenGenAIException):
    """Base class for task-related errors."""

    def __init__(
        self, message: str, task_id: str | None = None, task_name: str | None = None, **kwargs
    ):
        """Initialize task error."""
        details = {
            "task_id": task_id,
            "task_name": task_name,
        }
        super().__init__(message, details=details, **kwargs)


class TaskNotFoundError(TaskError):
    """Raised when a task is not found."""

    def __init__(self, task_id: str, **kwargs):
        """Initialize task not found error."""
        super().__init__(f"Task with ID '{task_id}' not found", task_id=task_id, **kwargs)


class TaskTimeoutError(TaskError):
    """Raised when task execution times out."""

    def __init__(
        self, message: str, task_id: str | None = None, timeout_seconds: int | None = None, **kwargs
    ):
        """Initialize task timeout error."""
        details = {
            "task_id": task_id,
            "timeout_seconds": timeout_seconds,
        }
        super().__init__(message, details=details, **kwargs)


class MemoryLimitExceededError(ResourceError):
    """Raised when memory limit is exceeded."""

    def __init__(
        self,
        message: str,
        current_usage_mb: float | None = None,
        limit_mb: float | None = None,
        **kwargs,
    ):
        """Initialize memory limit exceeded error."""
        details = {
            "current_usage_mb": current_usage_mb,
            "limit_mb": limit_mb,
        }
        super().__init__(message, details=details, **kwargs)


class CPULimitExceededError(ResourceError):
    """Raised when CPU limit is exceeded."""

    def __init__(
        self,
        message: str,
        current_usage_percent: float | None = None,
        limit_percent: float | None = None,
        **kwargs,
    ):
        """Initialize CPU limit exceeded error."""
        details = {
            "current_usage_percent": current_usage_percent,
            "limit_percent": limit_percent,
        }
        super().__init__(message, details=details, **kwargs)


class ConfigurationError(OpenGenAIException):
    """Raised when configuration is invalid."""

    def __init__(
        self, message: str, config_key: str | None = None, config_value: Any | None = None, **kwargs
    ):
        """Initialize configuration error."""
        details = {
            "config_key": config_key,
            "config_value": config_value,
        }
        super().__init__(message, details=details, **kwargs)


class ExternalServiceError(OpenGenAIException):
    """Raised when external service calls fail."""

    def __init__(
        self,
        message: str,
        service_name: str | None = None,
        service_url: str | None = None,
        status_code: int | None = None,
        **kwargs,
    ):
        """Initialize external service error."""
        details = {
            "service_name": service_name,
            "service_url": service_url,
            "status_code": status_code,
        }
        super().__init__(message, details=details, **kwargs)


class DatabaseError(OpenGenAIException):
    """Raised when database operations fail."""

    def __init__(
        self, message: str, operation: str | None = None, table_name: str | None = None, **kwargs
    ):
        """Initialize database error."""
        details = {
            "operation": operation,
            "table_name": table_name,
        }
        super().__init__(message, details=details, **kwargs)


class SecurityError(OpenGenAIException):
    """Raised when security violations occur."""

    def __init__(
        self,
        message: str,
        security_context: str | None = None,
        user_id: str | None = None,
        **kwargs,
    ):
        """Initialize security error."""
        details = {
            "security_context": security_context,
            "user_id": user_id,
        }
        super().__init__(message, details=details, **kwargs)


class ExceptionHandler:
    """Centralized exception handler for the OpenGenAI system."""

    def __init__(self):
        """Initialize exception handler."""
        self.handlers: dict[type, Callable[[Exception], dict[str, Any]]] = {}
        self.global_handler: Callable[[Exception], None] | None = None

    def register_handler(
        self, exception_type: type, handler: Callable[[Exception], dict[str, Any]]
    ) -> None:
        """Register a handler for a specific exception type."""
        self.handlers[exception_type] = handler

    def set_global_handler(self, handler: Callable[[Exception], None]) -> None:
        """Set a global exception handler."""
        self.global_handler = handler

    def handle(self, exception: Exception) -> dict[str, Any]:
        """Handle an exception using registered handlers."""
        # Try specific handler first
        for exc_type, handler in self.handlers.items():
            if isinstance(exception, exc_type):
                return handler(exception)

        # Fall back to global handler
        if self.global_handler:
            self.global_handler(exception)

        # Default handling
        return handle_exception(exception)


# Global exception handler instance
exception_handler = ExceptionHandler()


def handle_exception(exception: Exception) -> dict[str, Any]:
    """Handle an exception and return error information."""
    if isinstance(exception, OpenGenAIException):
        return exception.to_dict()

    return {
        "error_type": exception.__class__.__name__,
        "error_code": "UNKNOWN_ERROR",
        "message": str(exception),
        "details": {},
        "timestamp": datetime.now(UTC).isoformat(),
        "cause": None,
    }


def format_exception_for_api(exception: Exception) -> dict[str, Any]:
    """Format exception for API response."""
    error_info = handle_exception(exception)

    # Determine HTTP status code based on exception type
    status_code = 500  # Default to internal server error

    if isinstance(exception, ValidationError):
        status_code = 400
    elif isinstance(exception, AuthenticationError):
        status_code = 401
    elif isinstance(exception, AuthorizationError):
        status_code = 403
    elif isinstance(exception, ResourceNotFoundError):
        status_code = 404
    elif isinstance(exception, ResourceAlreadyExistsError):
        status_code = 409
    elif isinstance(exception, RateLimitError):
        status_code = 429
    elif isinstance(exception, AgentTimeoutError):
        status_code = 408
    elif isinstance(exception, TaskTimeoutError):
        status_code = 408

    return {
        "status_code": status_code,
        "error": error_info,
        "success": False,
        "timestamp": datetime.now(UTC).isoformat(),
    }
