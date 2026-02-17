"""vLLM inference engine backend (OpenAI-compatible API)."""

from __future__ import annotations

from openjarvis.core.registry import EngineRegistry
from openjarvis.engine._openai_compat import _OpenAICompatibleEngine


@EngineRegistry.register("vllm")
class VLLMEngine(_OpenAICompatibleEngine):
    """vLLM backend — thin wrapper over the shared OpenAI-compatible base."""

    engine_id = "vllm"
    _default_host = "http://localhost:8000"


__all__ = ["VLLMEngine"]
