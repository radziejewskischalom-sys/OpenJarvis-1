"""Tests for the OpenClaw plugin skeleton."""

from __future__ import annotations

from unittest.mock import MagicMock

from openjarvis.agents.openclaw_plugin import (
    MemorySearchManager,
    ProviderPlugin,
    register,
)


class TestRegister:
    def test_register_returns_dict(self):
        result = register()
        assert isinstance(result, dict)
        assert result["name"] == "openjarvis"
        assert "provider_class" in result
        assert "memory_class" in result


class TestProviderPlugin:
    def test_has_generate(self):
        assert hasattr(ProviderPlugin, "generate")

    def test_has_list_models(self):
        assert hasattr(ProviderPlugin, "list_models")

    def test_generate_with_engine(self):
        engine = MagicMock()
        engine.generate.return_value = {"content": "test"}
        plugin = ProviderPlugin(engine=engine, model="test-model")
        result = plugin.generate("Hello")
        assert result["content"] == "test"

    def test_list_models_with_engine(self):
        engine = MagicMock()
        engine.list_models.return_value = ["model-1", "model-2"]
        plugin = ProviderPlugin(engine=engine)
        assert plugin.list_models() == ["model-1", "model-2"]


class TestMemorySearchManager:
    def test_has_search(self):
        assert hasattr(MemorySearchManager, "search")

    def test_has_sync(self):
        assert hasattr(MemorySearchManager, "sync")

    def test_has_status(self):
        assert hasattr(MemorySearchManager, "status")

    def test_search_with_backend(self):
        backend = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "found"
        mock_result.score = 0.9
        mock_result.source = "test.txt"
        backend.retrieve.return_value = [mock_result]

        mgr = MemorySearchManager(backend=backend)
        results = mgr.search("test query")
        assert len(results) == 1
        assert results[0]["content"] == "found"

    def test_search_no_backend(self):
        mgr = MemorySearchManager()
        results = mgr.search("test")
        assert results == []

    def test_status_no_backend(self):
        mgr = MemorySearchManager()
        assert mgr.status()["available"] is False

    def test_sync(self):
        mgr = MemorySearchManager()
        result = mgr.sync()
        assert result["status"] == "ok"
