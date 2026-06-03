"""Gymnasium environments for learning graph cybersecurity policies."""

from typing import TYPE_CHECKING

from .belief import BeliefUpdater, enumerate_binary_states
from .policies import UniformAttackerPolicy

if TYPE_CHECKING:
    from .env import CyberGraphDefenseEnv, GameConfig

__all__ = [
    "BeliefUpdater",
    "CyberGraphDefenseEnv",
    "GameConfig",
    "UniformAttackerPolicy",
    "enumerate_binary_states",
]


def __getattr__(name: str):
    if name in {"CyberGraphDefenseEnv", "GameConfig"}:
        from .env import CyberGraphDefenseEnv, GameConfig

        return {
            "CyberGraphDefenseEnv": CyberGraphDefenseEnv,
            "GameConfig": GameConfig,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

