"""Versioned relation semantics and bounded expansion settings."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

EXPLICIT = frozenset(
    {
        "CITES",
        "AMENDS",
        "ADDS_TO",
        "REPEALS",
        "REPLACES",
        "DETAILS",
        "IMPLEMENTS",
        "GUIDES_IMPLEMENTATION_OF",
    }
)
SEMANTIC = frozenset(
    {"SPECIFIES", "SETS_CONDITION_FOR", "EXCEPTION_TO", "PENALTY_FOR", "DEFINES", "REGULATES"}
)
PROVISION_KINDS = EXPLICIT | (SEMANTIC - {"DEFINES", "REGULATES"})
VERSION = "vn-legal-relations/v2"


@dataclass(frozen=True)
class ExpansionConfig:
    mode: str = "M4"
    depth: int = 2
    confidence: float = 0.8
    seed_limit: int = 5
    candidate_budget: int = 60
    per_node_limit: int = 12
    decay: float = 0.65
    graph_weight: float = 0.15
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "STRUCTURAL": 0.15,
            "CITES": 0.35,
            "AMENDS": 0.9,
            "ADDS_TO": 0.9,
            "REPEALS": 0.9,
            "REPLACES": 0.9,
            "DETAILS": 0.9,
            "IMPLEMENTS": 0.8,
            "GUIDES_IMPLEMENTATION_OF": 0.8,
            "SPECIFIES": 0.9,
            "SETS_CONDITION_FOR": 1.0,
            "EXCEPTION_TO": 1.0,
            "PENALTY_FOR": 1.0,
        }
    )

    def __post_init__(self):
        if self.mode not in {"M0", "M1", "M2", "M3", "M4"}:
            raise ValueError("mode must be M0..M4")
        if not 1 <= self.depth <= 4:
            raise ValueError("depth must be between 1 and 4")
        if min(self.seed_limit, self.candidate_budget, self.per_node_limit) <= 0:
            raise ValueError("candidate limits must be positive")
        for value in (self.confidence, self.decay, self.graph_weight, *self.weights.values()):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("confidence, decay and weights must be finite in [0, 1]")

    @property
    def kinds(self) -> list[str]:
        return sorted(
            (EXPLICIT if self.mode in {"M2", "M4"} else frozenset())
            | (SEMANTIC if self.mode in {"M3", "M4"} else frozenset())
        )
