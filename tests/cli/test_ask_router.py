"""Tests for the --router CLI option in jarvis ask."""

from __future__ import annotations

import importlib
from unittest import mock

from click.testing import CliRunner

from openjarvis.cli import cli

_ask_mod = importlib.import_module("openjarvis.cli.ask")


def _mock_engine():
    """Create a mock engine that returns a simple response."""
    engine = mock.MagicMock()
    engine.engine_id = "mock"
    engine.health.return_value = True
    engine.list_models.return_value = ["test-model"]
    engine.generate.return_value = {
        "content": "Hello!",
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "model": "test-model",
        "finish_reason": "stop",
    }
    return engine


def _patch_engine(engine):
    """Return context managers that patch engine discovery to use our mock."""
    return (
        mock.patch.object(
            _ask_mod, "get_engine",
            return_value=("mock", engine),
        ),
        mock.patch.object(
            _ask_mod, "discover_engines",
            return_value={"mock": engine},
        ),
        mock.patch.object(
            _ask_mod, "discover_models",
            return_value={"mock": ["test-model"]},
        ),
        mock.patch.object(_ask_mod, "register_builtin_models"),
        mock.patch.object(_ask_mod, "merge_discovered_models"),
        mock.patch.object(_ask_mod, "TelemetryStore"),
    )


class TestAskRouter:
    def test_default_uses_heuristic(self) -> None:
        engine = _mock_engine()
        patches = _patch_engine(engine)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = CliRunner().invoke(cli, ["ask", "Hello"])
        assert result.exit_code == 0
        assert "Hello!" in result.output

    def test_explicit_heuristic(self) -> None:
        engine = _mock_engine()
        patches = _patch_engine(engine)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = CliRunner().invoke(cli, ["ask", "--router", "heuristic", "Hello"])
        assert result.exit_code == 0
        assert "Hello!" in result.output

    def test_grpo_raises_error(self) -> None:
        engine = _mock_engine()
        patches = _patch_engine(engine)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = CliRunner().invoke(cli, ["ask", "--router", "grpo", "Hello"])
        # GRPO raises NotImplementedError which should surface as an error
        assert result.exit_code != 0

    def test_unknown_router_falls_back(self) -> None:
        engine = _mock_engine()
        patches = _patch_engine(engine)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = CliRunner().invoke(
                cli, ["ask", "--router", "nonexistent", "Hello"],
            )
        assert result.exit_code == 0
        assert "Hello!" in result.output

    def test_model_flag_bypasses_router(self) -> None:
        engine = _mock_engine()
        patches = _patch_engine(engine)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = CliRunner().invoke(
                cli, ["ask", "-m", "test-model", "Hello"],
            )
        assert result.exit_code == 0
        assert "Hello!" in result.output

    def test_config_default_policy_respected(self) -> None:
        engine = _mock_engine()
        patches = _patch_engine(engine)
        with (
            patches[0], patches[1], patches[2], patches[3], patches[4], patches[5],
            mock.patch.object(
                _ask_mod, "load_config",
            ) as mock_config,
        ):
            cfg = mock_config.return_value
            cfg.telemetry.enabled = False
            cfg.intelligence.default_model = "test-model"
            cfg.intelligence.fallback_model = ""
            cfg.learning.default_policy = "heuristic"
            cfg.memory.context_injection = False
            result = CliRunner().invoke(cli, ["ask", "Hello"])
        assert result.exit_code == 0
