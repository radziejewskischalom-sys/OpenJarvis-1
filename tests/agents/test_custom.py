"""Tests for the CustomAgent stub."""

from __future__ import annotations

import pytest

from openjarvis.agents.custom import CustomAgent


class TestCustomAgent:
    def test_agent_id(self):
        agent = CustomAgent()
        assert agent.agent_id == "custom"

    def test_run_raises(self):
        agent = CustomAgent()
        with pytest.raises(NotImplementedError, match="template"):
            agent.run("Hello")

    def test_error_message_helpful(self):
        agent = CustomAgent()
        with pytest.raises(NotImplementedError, match="Subclass"):
            agent.run("test")

    def test_error_mentions_register(self):
        agent = CustomAgent()
        with pytest.raises(NotImplementedError, match="register"):
            agent.run("test")
