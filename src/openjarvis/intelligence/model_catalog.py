"""Built-in model catalog with well-known ModelSpec entries."""

from __future__ import annotations

from typing import List

from openjarvis.core.registry import ModelRegistry
from openjarvis.core.types import ModelSpec

BUILTIN_MODELS: List[ModelSpec] = [
    ModelSpec(
        model_id="qwen3:8b",
        name="Qwen3 8B",
        parameter_count_b=8.0,
        context_length=32768,
        supported_engines=("ollama", "vllm", "llamacpp"),
        provider="alibaba",
    ),
    ModelSpec(
        model_id="qwen3:32b",
        name="Qwen3 32B",
        parameter_count_b=32.0,
        context_length=32768,
        min_vram_gb=20.0,
        supported_engines=("ollama", "vllm"),
        provider="alibaba",
    ),
    ModelSpec(
        model_id="llama3.3:70b",
        name="Llama 3.3 70B",
        parameter_count_b=70.0,
        context_length=131072,
        min_vram_gb=40.0,
        supported_engines=("ollama", "vllm"),
        provider="meta",
    ),
    ModelSpec(
        model_id="llama3.2:3b",
        name="Llama 3.2 3B",
        parameter_count_b=3.0,
        context_length=131072,
        supported_engines=("ollama", "vllm", "llamacpp"),
        provider="meta",
    ),
    ModelSpec(
        model_id="deepseek-coder-v2:16b",
        name="DeepSeek Coder V2 16B",
        parameter_count_b=16.0,
        context_length=131072,
        supported_engines=("ollama", "vllm"),
        provider="deepseek",
    ),
    ModelSpec(
        model_id="mistral:7b",
        name="Mistral 7B",
        parameter_count_b=7.0,
        context_length=32768,
        supported_engines=("ollama", "vllm", "llamacpp"),
        provider="mistral",
    ),
    ModelSpec(
        model_id="gpt-4o",
        name="GPT-4o",
        parameter_count_b=0.0,
        context_length=128000,
        supported_engines=("cloud",),
        provider="openai",
        requires_api_key=True,
    ),
    ModelSpec(
        model_id="gpt-4o-mini",
        name="GPT-4o Mini",
        parameter_count_b=0.0,
        context_length=128000,
        supported_engines=("cloud",),
        provider="openai",
        requires_api_key=True,
    ),
    ModelSpec(
        model_id="claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        parameter_count_b=0.0,
        context_length=200000,
        supported_engines=("cloud",),
        provider="anthropic",
        requires_api_key=True,
    ),
    ModelSpec(
        model_id="claude-opus-4-20250514",
        name="Claude Opus 4",
        parameter_count_b=0.0,
        context_length=200000,
        supported_engines=("cloud",),
        provider="anthropic",
        requires_api_key=True,
    ),
]


def register_builtin_models() -> None:
    """Populate ``ModelRegistry`` with well-known models."""
    for spec in BUILTIN_MODELS:
        if not ModelRegistry.contains(spec.model_id):
            ModelRegistry.register_value(spec.model_id, spec)


def merge_discovered_models(engine_key: str, model_ids: List[str]) -> None:
    """Create minimal ``ModelSpec`` entries for models not already in the registry."""
    for model_id in model_ids:
        if not ModelRegistry.contains(model_id):
            spec = ModelSpec(
                model_id=model_id,
                name=model_id,
                parameter_count_b=0.0,
                context_length=0,
                supported_engines=(engine_key,),
            )
            ModelRegistry.register_value(model_id, spec)


__all__ = ["BUILTIN_MODELS", "merge_discovered_models", "register_builtin_models"]
