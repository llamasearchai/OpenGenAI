from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, Iterable, List, Optional

from openai.types.chat import ChatCompletionMessageParam


# --------------------------------------------------------------------------- #
#  ENUMS & CONSTANTS
# --------------------------------------------------------------------------- #
class AgentStatus(Enum):
    INITIALIZING = auto()
    RUNNING = auto()
    TIMEOUT = auto()
    FAILED = auto()
    TERMINATED = auto()


class MessageType(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MemoryType(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


# --------------------------------------------------------------------------- #
#  CORE MODELS
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class AgentState:
    iteration_count: int = 0
    last_activity: datetime = field(default_factory=datetime.utcnow)
    start_time: datetime = field(default_factory=datetime.utcnow)

    def uptime_seconds(self) -> int:
        return int((datetime.utcnow() - self.start_time).total_seconds())


@dataclass(slots=True)
class AgentMemory:
    content: str
    memory_type: MemoryType = MemoryType.SHORT_TERM
    importance: float = 0.5
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    iteration: Optional[int] = None


@dataclass(slots=True)
class AgentMessage:
    type: MessageType
    content: str
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sender_id: Optional[str] = None
    recipient_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# convenience for message lists in OpenAI format
def to_openai_messages(
    msgs: Iterable["AgentMessage"],
) -> List[ChatCompletionMessageParam]:
    return [{"role": m.type.value, "content": m.content} for m in msgs]  # type: ignore[return-value] 