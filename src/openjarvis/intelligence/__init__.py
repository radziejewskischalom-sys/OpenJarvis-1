"""Intelligence pillar — model management and query routing."""

from __future__ import annotations

from openjarvis.intelligence.model_catalog import (
    BUILTIN_MODELS,
    merge_discovered_models,
    register_builtin_models,
)
from openjarvis.intelligence.router import HeuristicRouter, build_routing_context

__all__ = [
    "BUILTIN_MODELS",
    "HeuristicRouter",
    "build_routing_context",
    "merge_discovered_models",
    "register_builtin_models",
]
