"""OpenClawAgent — wraps the OpenClaw Pi agent via HTTP or subprocess transport."""

from __future__ import annotations

import json
from typing import Any, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.agents.openclaw_protocol import MessageType, ProtocolMessage
from openjarvis.agents.openclaw_transport import (
    HttpTransport,
    OpenClawTransport,
    SubprocessTransport,
)
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import ToolCall, ToolResult


@AgentRegistry.register("openclaw")
class OpenClawAgent(BaseAgent):
    """Wraps the OpenClaw Pi agent via HTTP or subprocess transport.

    Parameters
    ----------
    engine:
        Inference engine (used as fallback or for provider plugin).
    model:
        Model identifier.
    transport:
        Optional pre-configured transport. If None, one is created based on *mode*.
    mode:
        Transport mode: ``"http"`` (default) or ``"subprocess"``.
    bus:
        Optional event bus for telemetry.
    """

    agent_id = "openclaw"

    def __init__(
        self,
        engine: Any = None,
        model: str = "",
        *,
        transport: Optional[OpenClawTransport] = None,
        mode: str = "http",
        bus: Optional[EventBus] = None,
        **kwargs: Any,
    ) -> None:
        self._engine = engine
        self._model = model
        self._bus = bus

        if transport is not None:
            self._transport = transport
        elif mode == "http":
            self._transport = HttpTransport()
        elif mode == "subprocess":
            self._transport = SubprocessTransport()
        else:
            raise ValueError(
                f"Unknown OpenClaw mode: {mode!r}. "
                "Use 'http' or 'subprocess'."
            )

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Send a query through the OpenClaw transport and handle tool calls."""
        if self._bus:
            self._bus.publish(EventType.AGENT_TURN_START, {"agent": self.agent_id})

        # Check transport health
        if not self._transport.health():
            raise RuntimeError(
                "OpenClaw transport is not healthy. "
                "Ensure the OpenClaw server is running."
            )

        # Build and send query message
        query_msg = ProtocolMessage(
            type=MessageType.QUERY,
            content=input,
            metadata={"model": self._model},
        )

        tool_results: list[ToolResult] = []
        turns = 0
        max_turns = 10

        response = self._transport.send(query_msg)
        turns += 1

        # Handle tool-call loop
        while response.type == MessageType.TOOL_CALL and turns < max_turns:
            if self._bus:
                self._bus.publish(EventType.TOOL_CALL_START, {
                    "tool": response.tool_name,
                    "arguments": response.tool_args,
                })

            # Execute tool locally
            tool_result = self._execute_tool(
                response.tool_name or "",
                response.tool_args or {},
            )
            tool_results.append(tool_result)

            if self._bus:
                self._bus.publish(EventType.TOOL_CALL_END, {
                    "tool": response.tool_name,
                    "success": tool_result.success,
                })

            # Send tool result back
            result_msg = ProtocolMessage(
                type=MessageType.TOOL_RESULT,
                tool_name=response.tool_name,
                tool_result=tool_result.content,
                metadata={"tool_call_id": response.id},
            )
            response = self._transport.send(result_msg)
            turns += 1

        # Handle error response
        if response.type == MessageType.ERROR:
            content = (
                response.error or response.content
                or "Unknown error from OpenClaw"
            )
        else:
            content = response.content

        if self._bus:
            self._bus.publish(EventType.AGENT_TURN_END, {
                "agent": self.agent_id,
                "turns": turns,
            })

        return AgentResult(
            content=content,
            tool_results=tool_results,
            turns=turns,
        )

    def _execute_tool(self, tool_name: str, tool_args: dict) -> ToolResult:
        """Dispatch a tool call to the local tool executor."""
        try:
            from openjarvis.core.registry import ToolRegistry

            if not ToolRegistry.contains(tool_name):
                return ToolResult(
                    tool_name=tool_name,
                    content=f"Unknown tool: {tool_name}",
                    success=False,
                )

            tool_cls = ToolRegistry.get(tool_name)
            tool = tool_cls() if callable(tool_cls) else tool_cls
            tc = ToolCall(
                id="openclaw-tool",
                name=tool_name,
                arguments=json.dumps(tool_args),
            )

            from openjarvis.tools._stubs import ToolExecutor

            executor = ToolExecutor([tool], bus=self._bus)
            return executor.execute(tc)
        except Exception as exc:
            return ToolResult(
                tool_name=tool_name,
                content=f"Tool execution error: {exc}",
                success=False,
            )


__all__ = ["OpenClawAgent"]
