"""OpenClaw wire protocol — JSON-line message serialization."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class MessageType(str, Enum):
    """Wire message types for OpenClaw communication."""

    QUERY = "query"
    RESPONSE = "response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    HEALTH = "health"
    HEALTH_OK = "health_ok"


@dataclass(slots=True)
class ProtocolMessage:
    """A single message in the OpenClaw protocol."""

    type: MessageType
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def serialize(msg: ProtocolMessage) -> str:
    """Serialize a ProtocolMessage to a JSON line."""
    obj: Dict[str, Any] = {
        "type": msg.type.value,
        "id": msg.id,
        "content": msg.content,
    }
    if msg.tool_name is not None:
        obj["tool_name"] = msg.tool_name
    if msg.tool_args is not None:
        obj["tool_args"] = msg.tool_args
    if msg.tool_result is not None:
        obj["tool_result"] = msg.tool_result
    if msg.error is not None:
        obj["error"] = msg.error
    if msg.metadata:
        obj["metadata"] = msg.metadata
    return json.dumps(obj)


def deserialize(line: str) -> ProtocolMessage:
    """Deserialize a JSON line into a ProtocolMessage.

    Raises
    ------
    ValueError
        If the JSON is invalid or the message type is unknown.
    """
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object")

    raw_type = obj.get("type", "")
    try:
        msg_type = MessageType(raw_type)
    except ValueError:
        raise ValueError(f"Unknown message type: {raw_type!r}") from None

    return ProtocolMessage(
        type=msg_type,
        id=obj.get("id", str(uuid.uuid4())),
        content=obj.get("content", ""),
        tool_name=obj.get("tool_name"),
        tool_args=obj.get("tool_args"),
        tool_result=obj.get("tool_result"),
        error=obj.get("error"),
        metadata=obj.get("metadata", {}),
    )


__all__ = ["MessageType", "ProtocolMessage", "deserialize", "serialize"]
