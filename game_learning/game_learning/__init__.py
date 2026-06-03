"""Gymnasium environments for learning multi-attacker MTD policies."""

from typing import TYPE_CHECKING

from .belief import BeliefUpdater, enumerate_binary_states, node_marginals
from .fictitious_play import FictitiousPlayConfig, run_fictitious_play
from .policies import (
    AttackerEnsemble,
    FocusedAttackerPolicy,
    RandomDefenderPolicy,
    ThresholdDefenderPolicy,
    UniformAttackerPolicy,
)

if TYPE_CHECKING:
    from .env import CyberGraphDefenseEnv, GameConfig

__all__ = [
    "AttackerEnsemble",
    "BeliefUpdater",
    "CyberGraphDefenseEnv",
    "FictitiousPlayConfig",
    "FocusedAttackerPolicy",
    "GameConfig",
    "RandomDefenderPolicy",
    "ThresholdDefenderPolicy",
    "UniformAttackerPolicy",
    "enumerate_binary_states",
    "node_marginals",
    "run_fictitious_play",
]


def __getattr__(name: str):
    if name in {"CyberGraphDefenseEnv", "GameConfig"}:
        from .env import CyberGraphDefenseEnv, GameConfig

        return {
            "CyberGraphDefenseEnv": CyberGraphDefenseEnv,
            "GameConfig": GameConfig,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
