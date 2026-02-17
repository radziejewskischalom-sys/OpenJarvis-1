"""Tests for the OpenClawAgent."""

from __future__ import annotations

import pytest

from openjarvis.agents._stubs import AgentResult
from openjarvis.agents.openclaw import OpenClawAgent
from openjarvis.agents.openclaw_protocol import MessageType, ProtocolMessage
from openjarvis.agents.openclaw_transport import OpenClawTransport
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import AgentRegistry


@pytest.fixture(autouse=True)
def _register_openclaw():
    """Re-register openclaw agent after registry clear."""
    if not AgentRegistry.contains("openclaw"):
        AgentRegistry.register_value("openclaw", OpenClawAgent)


class MockTransport(OpenClawTransport):
    """Mock transport for testing."""

    def __init__(self, responses=None, healthy=True):
        self._responses = list(responses or [])
        self._idx = 0
        self._healthy = healthy

    def send(self, msg):
        if self._idx < len(self._responses):
            resp = self._responses[self._idx]
            self._idx += 1
            return resp
        return ProtocolMessage(type=MessageType.RESPONSE, content="default")

    def health(self):
        return self._healthy

    def close(self):
        pass


class TestOpenClawAgent:
    def test_agent_id(self):
        transport = MockTransport()
        agent = OpenClawAgent(transport=transport)
        assert agent.agent_id == "openclaw"

    def test_registration(self):
        assert AgentRegistry.contains("openclaw")
        assert AgentRegistry.get("openclaw") is OpenClawAgent

    def test_run_with_mock_transport(self):
        responses = [
            ProtocolMessage(type=MessageType.RESPONSE, content="Hello from OpenClaw"),
        ]
        transport = MockTransport(responses)
        agent = OpenClawAgent(transport=transport)
        result = agent.run("Hello")
        assert isinstance(result, AgentResult)
        assert result.content == "Hello from OpenClaw"
        assert result.turns == 1

    def test_returns_agent_result(self):
        transport = MockTransport([
            ProtocolMessage(type=MessageType.RESPONSE, content="test"),
        ])
        agent = OpenClawAgent(transport=transport)
        result = agent.run("test")
        assert isinstance(result, AgentResult)

    def test_handles_tool_calls(self):
        responses = [
            ProtocolMessage(
                type=MessageType.TOOL_CALL,
                tool_name="calculator",
                tool_args={"expression": "2+2"},
            ),
            ProtocolMessage(type=MessageType.RESPONSE, content="The answer is 4"),
        ]
        transport = MockTransport(responses)
        agent = OpenClawAgent(transport=transport)
        result = agent.run("What is 2+2?")
        assert result.content == "The answer is 4"
        assert result.turns == 2
        assert len(result.tool_results) == 1

    def test_transport_unhealthy_raises(self):
        transport = MockTransport(healthy=False)
        agent = OpenClawAgent(transport=transport)
        with pytest.raises(RuntimeError, match="not healthy"):
            agent.run("Hello")

    def test_mode_http(self):
        agent = OpenClawAgent(mode="http")
        from openjarvis.agents.openclaw_transport import HttpTransport

        assert isinstance(agent._transport, HttpTransport)

    def test_mode_subprocess(self):
        agent = OpenClawAgent(mode="subprocess")
        from openjarvis.agents.openclaw_transport import SubprocessTransport

        assert isinstance(agent._transport, SubprocessTransport)

    def test_mode_invalid(self):
        with pytest.raises(ValueError, match="Unknown OpenClaw mode"):
            OpenClawAgent(mode="invalid")

    def test_event_bus_events(self):
        responses = [
            ProtocolMessage(type=MessageType.RESPONSE, content="test"),
        ]
        transport = MockTransport(responses)
        bus = EventBus(record_history=True)
        agent = OpenClawAgent(transport=transport, bus=bus)
        agent.run("Hello")

        event_types = [e.event_type for e in bus.history]
        assert EventType.AGENT_TURN_START in event_types
        assert EventType.AGENT_TURN_END in event_types

    def test_custom_transport_injection(self):
        mock = MockTransport([
            ProtocolMessage(type=MessageType.RESPONSE, content="injected"),
        ])
        agent = OpenClawAgent(transport=mock)
        result = agent.run("test")
        assert result.content == "injected"

    def test_run_with_context(self):
        from openjarvis.agents._stubs import AgentContext
        from openjarvis.core.types import Conversation, Message, Role

        transport = MockTransport([
            ProtocolMessage(type=MessageType.RESPONSE, content="contextualized"),
        ])
        agent = OpenClawAgent(transport=transport)
        conv = Conversation()
        conv.add(Message(role=Role.SYSTEM, content="Be helpful"))
        ctx = AgentContext(conversation=conv)
        result = agent.run("Hello", context=ctx)
        assert result.content == "contextualized"

    def test_error_response(self):
        transport = MockTransport([
            ProtocolMessage(type=MessageType.ERROR, error="Something failed"),
        ])
        agent = OpenClawAgent(transport=transport)
        result = agent.run("Hello")
        assert result.content == "Something failed"
