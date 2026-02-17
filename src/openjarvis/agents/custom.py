"""CustomAgent — template for user-defined agents."""

from __future__ import annotations

from typing import Any, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.core.registry import AgentRegistry


@AgentRegistry.register("custom")
class CustomAgent(BaseAgent):
    """Template for user-defined agents.

    Subclass this agent and override ``run()`` to implement custom behavior.
    Register your subclass with ``@AgentRegistry.register("my-agent")``.
    """

    agent_id = "custom"

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        raise NotImplementedError(
            "CustomAgent is a template. Subclass it and override run() "
            "to implement your custom agent logic. Register with "
            "@AgentRegistry.register('your-agent-name')."
        )


__all__ = ["CustomAgent"]
