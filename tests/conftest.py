"""Shared fixtures — clear all registries and the event bus between tests."""

from __future__ import annotations

import pytest

from openjarvis.core.events import reset_event_bus
from openjarvis.core.registry import (
    AgentRegistry,
    BenchmarkRegistry,
    EngineRegistry,
    MemoryRegistry,
    ModelRegistry,
    RouterPolicyRegistry,
    ToolRegistry,
)


@pytest.fixture(autouse=True)
def _clean_registries() -> None:
    """Ensure each test starts with empty registries and a fresh event bus."""
    ModelRegistry.clear()
    EngineRegistry.clear()
    MemoryRegistry.clear()
    AgentRegistry.clear()
    ToolRegistry.clear()
    RouterPolicyRegistry.clear()
    BenchmarkRegistry.clear()
    reset_event_bus()
