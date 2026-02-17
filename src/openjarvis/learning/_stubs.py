"""ABCs for the Learning pillar — router policies and reward functions.

Phase 4 will provide heuristic and GRPO-based implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(slots=True)
class RoutingContext:
    """Inputs available to the router when selecting a model."""

    query: str = ""
    query_length: int = 0
    has_code: bool = False
    has_math: bool = False
    language: str = "en"
    urgency: float = 0.5  # 0 = low priority, 1 = real-time
    metadata: Dict[str, Any] = field(default_factory=dict)


class RouterPolicy(ABC):
    """Selects a model key given a ``RoutingContext``."""

    @abstractmethod
    def select_model(self, context: RoutingContext) -> str:
        """Return the model registry key best suited for *context*."""


class RewardFunction(ABC):
    """Computes a scalar reward for a completed inference, used by GRPO training."""

    @abstractmethod
    def compute(
        self,
        context: RoutingContext,
        model_key: str,
        response: str,
        **kwargs: Any,
    ) -> float:
        """Return a reward in ``[0, 1]``."""


__all__ = ["RewardFunction", "RouterPolicy", "RoutingContext"]
