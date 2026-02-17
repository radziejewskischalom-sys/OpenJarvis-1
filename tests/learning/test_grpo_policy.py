"""Tests for GRPORouterPolicy stub."""

from __future__ import annotations

import pytest

from openjarvis.learning._stubs import RoutingContext
from openjarvis.learning.grpo_policy import GRPORouterPolicy, ensure_registered


class TestGRPORouterPolicy:
    def test_raises_not_implemented(self) -> None:
        policy = GRPORouterPolicy()
        with pytest.raises(NotImplementedError):
            policy.select_model(RoutingContext(query="test"))

    def test_error_mentions_phase5(self) -> None:
        policy = GRPORouterPolicy()
        with pytest.raises(NotImplementedError, match="Phase 5"):
            policy.select_model(RoutingContext())

    def test_registered_in_registry(self) -> None:
        from openjarvis.core.registry import RouterPolicyRegistry

        ensure_registered()
        assert RouterPolicyRegistry.contains("grpo")
        assert RouterPolicyRegistry.get("grpo") is GRPORouterPolicy

    def test_accepts_kwargs(self) -> None:
        policy = GRPORouterPolicy(learning_rate=0.001, batch_size=32)
        assert policy._kwargs == {"learning_rate": 0.001, "batch_size": 32}
