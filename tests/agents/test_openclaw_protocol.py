"""Tests for the OpenClaw wire protocol."""

from __future__ import annotations

import pytest

from openjarvis.agents.openclaw_protocol import (
    MessageType,
    ProtocolMessage,
    deserialize,
    serialize,
)


class TestProtocolMessage:
    def test_query_message(self):
        msg = ProtocolMessage(type=MessageType.QUERY, content="Hello")
        assert msg.type == MessageType.QUERY
        assert msg.content == "Hello"

    def test_response_message(self):
        msg = ProtocolMessage(type=MessageType.RESPONSE, content="World")
        assert msg.type == MessageType.RESPONSE

    def test_tool_call_message(self):
        msg = ProtocolMessage(
            type=MessageType.TOOL_CALL,
            tool_name="calculator",
            tool_args={"expression": "2+2"},
        )
        assert msg.tool_name == "calculator"
        assert msg.tool_args == {"expression": "2+2"}

    def test_tool_result_message(self):
        msg = ProtocolMessage(
            type=MessageType.TOOL_RESULT,
            tool_name="calculator",
            tool_result="4",
        )
        assert msg.tool_result == "4"

    def test_error_message(self):
        msg = ProtocolMessage(type=MessageType.ERROR, error="Something went wrong")
        assert msg.error == "Something went wrong"

    def test_health_message(self):
        msg = ProtocolMessage(type=MessageType.HEALTH)
        assert msg.type == MessageType.HEALTH


class TestSerializeDeserialize:
    def test_roundtrip(self):
        original = ProtocolMessage(
            type=MessageType.QUERY,
            content="Hello world",
            metadata={"key": "value"},
        )
        line = serialize(original)
        restored = deserialize(line)
        assert restored.type == MessageType.QUERY
        assert restored.content == "Hello world"
        assert restored.metadata == {"key": "value"}

    def test_serialize_with_metadata(self):
        msg = ProtocolMessage(
            type=MessageType.RESPONSE,
            content="result",
            metadata={"model": "test"},
        )
        line = serialize(msg)
        restored = deserialize(line)
        assert restored.metadata["model"] == "test"

    def test_deserialize_valid_json(self):
        line = '{"type": "query", "id": "123", "content": "test"}'
        msg = deserialize(line)
        assert msg.type == MessageType.QUERY
        assert msg.id == "123"
        assert msg.content == "test"

    def test_deserialize_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            deserialize("not json")

    def test_deserialize_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown message type"):
            deserialize('{"type": "unknown_type", "content": "test"}')

    def test_tool_call_roundtrip(self):
        msg = ProtocolMessage(
            type=MessageType.TOOL_CALL,
            tool_name="calculator",
            tool_args={"expression": "3*7"},
        )
        line = serialize(msg)
        restored = deserialize(line)
        assert restored.type == MessageType.TOOL_CALL
        assert restored.tool_name == "calculator"
        assert restored.tool_args == {"expression": "3*7"}
