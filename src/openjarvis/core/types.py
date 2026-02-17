"""Canonical data types shared across all OpenJarvis pillars."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence  # noqa: I001

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Role(str, Enum):
    """Chat message roles (OpenAI-compatible)."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Quantization(str, Enum):
    """Model quantization formats."""

    NONE = "none"
    FP8 = "fp8"
    FP4 = "fp4"
    INT8 = "int8"
    INT4 = "int4"
    GGUF_Q4 = "gguf_q4"
    GGUF_Q8 = "gguf_q8"


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolCall:
    """A single tool invocation request embedded in an assistant message."""

    id: str
    name: str
    arguments: str  # JSON string


@dataclass(slots=True)
class Message:
    """A single chat message (OpenAI-compatible structure)."""

    role: Role
    content: str = ""
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Conversation:
    """Ordered list of messages with an optional sliding-window cap."""

    messages: List[Message] = field(default_factory=list)
    max_messages: Optional[int] = None

    def add(self, message: Message) -> None:
        """Append a message, trimming oldest if *max_messages* is set."""
        self.messages.append(message)
        if self.max_messages is not None and len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def window(self, n: int) -> List[Message]:
        """Return the last *n* messages."""
        return self.messages[-n:]


# ---------------------------------------------------------------------------
# Model / tool / telemetry records
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ModelSpec:
    """Metadata describing a language model."""

    model_id: str
    name: str
    parameter_count_b: float
    context_length: int
    active_parameter_count_b: Optional[float] = None  # MoE active params
    quantization: Quantization = Quantization.NONE
    min_vram_gb: float = 0.0
    supported_engines: Sequence[str] = ()
    provider: str = ""
    requires_api_key: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    """Result returned by a tool invocation."""

    tool_name: str
    content: str
    success: bool = True
    usage: Dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TelemetryRecord:
    """Single telemetry observation recorded after an inference call."""

    timestamp: float
    model_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    ttft: float = 0.0  # time to first token
    cost_usd: float = 0.0
    energy_joules: float = 0.0
    power_watts: float = 0.0
    engine: str = ""
    agent: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "Conversation",
    "Message",
    "ModelSpec",
    "Quantization",
    "Role",
    "TelemetryRecord",
    "ToolCall",
    "ToolResult",
]
