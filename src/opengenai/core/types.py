"""
OpenGenAI Type Definitions
Comprehensive type system for all OpenGenAI components.
"""

from collections.abc import Awaitable, Callable
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Environment(str, Enum):
    """Environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class AgentStatus(str, Enum):
    """Agent status enumeration."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    COMPLETED = "completed"


class TaskStatus(str, Enum):
    """Task status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class MessageType(str, Enum):
    """Message types for inter-agent communication."""

    GENERAL = "general"
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    ERROR = "error"
    SYSTEM = "system"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AgentCapability(str, Enum):
    """Agent capabilities enumeration."""

    REASONING = "reasoning"
    PLANNING = "planning"
    EXECUTION = "execution"
    LEARNING = "learning"
    COMMUNICATION = "communication"
    REFLECTION = "reflection"
    MEMORY_MANAGEMENT = "memory_management"
    TOOL_USAGE = "tool_usage"
    CODE_GENERATION = "code_generation"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    ORCHESTRATION = "orchestration"


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    capabilities: list[AgentCapability] = Field(
        default_factory=list, description="Agent capabilities"
    )
    model: str = Field("gpt-4-turbo-preview", description="OpenAI model to use")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(4096, ge=1, le=128000, description="Maximum tokens")
    max_iterations: int = Field(10, ge=1, le=100, description="Maximum iterations")
    max_memory_mb: int = Field(512, ge=64, le=4096, description="Memory limit in MB")
    max_cpu_percent: float = Field(80.0, ge=1.0, le=100.0, description="CPU limit")
    timeout_seconds: int = Field(300, ge=1, le=3600, description="Timeout in seconds")
    enable_reflection: bool = Field(True, description="Enable self-reflection")
    enable_learning: bool = Field(True, description="Enable learning from experience")
    memory_window_size: int = Field(100, ge=1, le=1000, description="Memory window size")
    tools: list[str] = Field(default_factory=list, description="Available tools")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate agent name."""
        if not v or not v.strip():
            raise ValueError("Agent name cannot be empty")
        if len(v) > 100:
            raise ValueError("Agent name cannot exceed 100 characters")
        return v.strip()


class TaskConfig(BaseModel):
    """Configuration for a task."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    id: str = Field(..., description="Task ID")
    name: str = Field(..., description="Task name")
    description: str = Field(..., description="Task description")
    priority: int = Field(0, ge=0, le=10, description="Task priority")
    agent_id: str | None = Field(None, description="Assigned agent ID")
    input_data: dict[str, Any] = Field(default_factory=dict, description="Input data")
    output_data: dict[str, Any] = Field(default_factory=dict, description="Output data")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Task metadata")
    timeout_seconds: int = Field(300, ge=1, le=3600, description="Task timeout")
    retry_count: int = Field(3, ge=0, le=10, description="Maximum retry attempts")
    dependencies: list[str] = Field(default_factory=list, description="Task dependencies")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate task name."""
        if not v or not v.strip():
            raise ValueError("Task name cannot be empty")
        return v.strip()


class AgentState(BaseModel):
    """Agent state representation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    id: str = Field(..., description="Agent ID")
    name: str = Field(..., description="Agent name")
    status: AgentStatus = Field(..., description="Agent status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    current_task_id: str | None = Field(None, description="Current task ID")
    last_error: str | None = Field(None, description="Last error message")
    memory_usage_mb: float = Field(0.0, description="Memory usage in MB")
    cpu_usage_percent: float = Field(0.0, description="CPU usage percentage")
    total_tasks_completed: int = Field(0, description="Total completed tasks")
    total_tasks_failed: int = Field(0, description="Total failed tasks")
    uptime_seconds: float = Field(0.0, description="Uptime in seconds")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Agent metrics")


class TaskState(BaseModel):
    """Task state representation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    id: str = Field(..., description="Task ID")
    name: str = Field(..., description="Task name")
    status: TaskStatus = Field(..., description="Task status")
    agent_id: str | None = Field(None, description="Assigned agent ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    started_at: datetime | None = Field(None, description="Start timestamp")
    completed_at: datetime | None = Field(None, description="Completion timestamp")
    error_message: str | None = Field(None, description="Error message")
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0, description="Progress")
    retry_count: int = Field(0, description="Current retry count")
    result: dict[str, Any] | None = Field(None, description="Task result")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Task metrics")


class AgentMessage(BaseModel):
    """Message for inter-agent communication."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    id: str = Field(..., description="Message ID")
    type: MessageType = Field(..., description="Message type")
    sender_id: str = Field(..., description="Sender agent ID")
    recipient_id: str | None = Field(None, description="Recipient agent ID")
    content: dict[str, Any] = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")
    correlation_id: str | None = Field(None, description="Correlation ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Message metadata")


class AgentMemory(BaseModel):
    """Agent memory representation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    id: str = Field(..., description="Memory ID")
    agent_id: str = Field(..., description="Agent ID")
    content: dict[str, Any] = Field(..., description="Memory content")
    memory_type: str = Field(..., description="Memory type")
    importance: float = Field(0.0, ge=0.0, le=1.0, description="Memory importance")
    timestamp: datetime = Field(..., description="Memory timestamp")
    access_count: int = Field(0, description="Access count")
    last_accessed: datetime | None = Field(None, description="Last access time")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Memory metadata")


class SystemHealthStatus(BaseModel):
    """System health status."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    status: str = Field(..., description="Overall status")
    timestamp: datetime = Field(..., description="Status timestamp")
    uptime_seconds: float = Field(..., description="System uptime")
    version: str = Field(..., description="System version")

    # Component health
    database_healthy: bool = Field(..., description="Database health")
    redis_healthy: bool = Field(..., description="Redis health")
    api_healthy: bool = Field(..., description="API health")
    agents_healthy: bool = Field(..., description="Agent system health")

    # Metrics
    total_agents: int = Field(0, description="Total agents")
    active_agents: int = Field(0, description="Active agents")
    total_tasks: int = Field(0, description="Total tasks")
    running_tasks: int = Field(0, description="Running tasks")

    # Resource usage
    memory_usage_percent: float = Field(0.0, description="Memory usage")
    cpu_usage_percent: float = Field(0.0, description="CPU usage")
    disk_usage_percent: float = Field(0.0, description="Disk usage")

    # Additional metrics
    request_rate: float = Field(0.0, description="Request rate per second")
    error_rate: float = Field(0.0, description="Error rate percentage")
    response_time_ms: float = Field(0.0, description="Average response time")


class APIResponse(BaseModel):
    """Standard API response format."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    success: bool = Field(..., description="Request success status")
    message: str = Field(..., description="Response message")
    data: dict[str, Any] | None = Field(None, description="Response data")
    timestamp: datetime = Field(..., description="Response timestamp")
    request_id: str | None = Field(None, description="Request ID")
    errors: list[str] = Field(default_factory=list, description="Error messages")


class MetricsData(BaseModel):
    """Metrics data structure."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    unit: str = Field(..., description="Metric unit")
    timestamp: datetime = Field(..., description="Metric timestamp")
    labels: dict[str, str] = Field(default_factory=dict, description="Metric labels")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metric metadata")


# Type aliases for convenience
AgentCallback = Callable[[AgentState], Awaitable[None]]
MessageHandler = Callable[[AgentMessage], Awaitable[None]]
ErrorHandler = Callable[[Exception], Awaitable[None]]
TaskCallback = Callable[[TaskState], Awaitable[None]]
HealthChecker = Callable[[], Awaitable[bool]]
MetricsCollector = Callable[[], Awaitable[list[MetricsData]]]
