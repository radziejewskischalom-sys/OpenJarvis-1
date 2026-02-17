"""llama.cpp inference engine backend (OpenAI-compatible API)."""

from __future__ import annotations

from openjarvis.core.registry import EngineRegistry
from openjarvis.engine._openai_compat import _OpenAICompatibleEngine


@EngineRegistry.register("llamacpp")
class LlamaCppEngine(_OpenAICompatibleEngine):
    """llama.cpp server — OpenAI-compatible base."""

    engine_id = "llamacpp"
    _default_host = "http://localhost:8080"


__all__ = ["LlamaCppEngine"]
