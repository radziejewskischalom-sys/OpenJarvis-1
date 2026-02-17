"""Cloud inference engine — OpenAI and Anthropic API backends."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence
from typing import Any, Dict, List

from openjarvis.core.registry import EngineRegistry
from openjarvis.core.types import Message
from openjarvis.engine._base import (
    EngineConnectionError,
    InferenceEngine,
    messages_to_dicts,
)

# Pricing per million tokens (input, output)
PRICING: Dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5": (10.00, 30.00),
    "o3-mini": (1.10, 4.40),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-haiku-3-5-20241022": (0.80, 4.00),
}

# Well-known model IDs per provider
_OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-5", "o3-mini"]
_ANTHROPIC_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-haiku-3-5-20241022",
]


def _is_anthropic_model(model: str) -> bool:
    return "claude" in model.lower()


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost based on the hardcoded pricing table."""
    # Try exact match first, then prefix match
    prices = PRICING.get(model)
    if prices is None:
        for key, val in PRICING.items():
            if model.startswith(key):
                prices = val
                break
    if prices is None:
        return 0.0
    input_cost = (prompt_tokens / 1_000_000) * prices[0]
    output_cost = (completion_tokens / 1_000_000) * prices[1]
    return input_cost + output_cost


@EngineRegistry.register("cloud")
class CloudEngine(InferenceEngine):
    """Cloud inference via OpenAI and Anthropic SDKs."""

    engine_id = "cloud"

    def __init__(self) -> None:
        self._openai_client: Any = None
        self._anthropic_client: Any = None
        self._init_clients()

    def _init_clients(self) -> None:
        if os.environ.get("OPENAI_API_KEY"):
            try:
                import openai
                self._openai_client = openai.OpenAI()
            except ImportError:
                pass
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic()
            except ImportError:
                pass

    def _generate_openai(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if self._openai_client is None:
            raise EngineConnectionError(
                "OpenAI client not available — set "
                "OPENAI_API_KEY and install "
                "openjarvis[inference-cloud]"
            )
        resp = self._openai_client.chat.completions.create(
            model=model,
            messages=messages_to_dicts(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        choice = resp.choices[0]
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        return {
            "content": choice.message.content or "",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": (usage.total_tokens if usage else 0),
            },
            "model": resp.model,
            "finish_reason": choice.finish_reason or "stop",
            "cost_usd": estimate_cost(model, prompt_tokens, completion_tokens),
        }

    def _generate_anthropic(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if self._anthropic_client is None:
            raise EngineConnectionError(
                "Anthropic client not available — set "
                "ANTHROPIC_API_KEY and install "
                "openjarvis[inference-cloud]"
            )
        # Separate system message from conversation messages
        system_text = ""
        chat_msgs: List[Dict[str, Any]] = []
        for m in messages:
            if m.role.value == "system":
                system_text = m.content
            else:
                chat_msgs.append({"role": m.role.value, "content": m.content})
        create_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": chat_msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_text:
            create_kwargs["system"] = system_text
        resp = self._anthropic_client.messages.create(**create_kwargs)
        content = resp.content[0].text if resp.content else ""
        prompt_tokens = resp.usage.input_tokens if resp.usage else 0
        completion_tokens = resp.usage.output_tokens if resp.usage else 0
        return {
            "content": content,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "model": resp.model,
            "finish_reason": resp.stop_reason or "stop",
            "cost_usd": estimate_cost(model, prompt_tokens, completion_tokens),
        }

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        kw = dict(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        if _is_anthropic_model(model):
            return self._generate_anthropic(messages, **kw)
        return self._generate_openai(messages, **kw)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        kw = dict(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        if _is_anthropic_model(model):
            async for token in self._stream_anthropic(
                messages, **kw
            ):
                yield token
        else:
            async for token in self._stream_openai(
                messages, **kw
            ):
                yield token

    async def _stream_openai(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        if self._openai_client is None:
            raise EngineConnectionError("OpenAI client not available")
        resp = self._openai_client.chat.completions.create(
            model=model,
            messages=messages_to_dicts(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    async def _stream_anthropic(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        if self._anthropic_client is None:
            raise EngineConnectionError("Anthropic client not available")
        system_text = ""
        chat_msgs: List[Dict[str, Any]] = []
        for m in messages:
            if m.role.value == "system":
                system_text = m.content
            else:
                chat_msgs.append({"role": m.role.value, "content": m.content})
        create_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": chat_msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_text:
            create_kwargs["system"] = system_text
        with self._anthropic_client.messages.stream(**create_kwargs) as stream:
            for text in stream.text_stream:
                yield text

    def list_models(self) -> List[str]:
        models: List[str] = []
        if self._openai_client is not None:
            models.extend(_OPENAI_MODELS)
        if self._anthropic_client is not None:
            models.extend(_ANTHROPIC_MODELS)
        return models

    def health(self) -> bool:
        return self._openai_client is not None or self._anthropic_client is not None


__all__ = ["CloudEngine", "PRICING", "estimate_cost"]
