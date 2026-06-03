"""Gymnasium environments for learning graph cybersecurity policies."""

from typing import TYPE_CHECKING

from .belief import BeliefUpdater, enumerate_binary_states
from .policies import UniformAttackerPolicy

if TYPE_CHECKING:
    from .env import (
        BasicCyberGraphAttackEnv,
        BasicCyberGraphDefenseConfig,
        BasicCyberGraphDefenseEnv,
    )
    from .policy_adapters import SB3AttackerPolicy, SB3DefenderPolicy

__all__ = [
    "BasicCyberGraphAttackEnv",
    "BasicCyberGraphDefenseConfig",
    "BasicCyberGraphDefenseEnv",
    "BeliefUpdater",
    "SB3AttackerPolicy",
    "SB3DefenderPolicy",
    "UniformAttackerPolicy",
    "enumerate_binary_states",
]


def __getattr__(name: str):
    if name in {
        "BasicCyberGraphAttackEnv",
        "BasicCyberGraphDefenseConfig",
        "BasicCyberGraphDefenseEnv",
    }:
        from .env import (
            BasicCyberGraphAttackEnv,
            BasicCyberGraphDefenseConfig,
            BasicCyberGraphDefenseEnv,
        )

        return {
            "BasicCyberGraphAttackEnv": BasicCyberGraphAttackEnv,
            "BasicCyberGraphDefenseConfig": BasicCyberGraphDefenseConfig,
            "BasicCyberGraphDefenseEnv": BasicCyberGraphDefenseEnv,
        }[name]
    if name in {"SB3AttackerPolicy", "SB3DefenderPolicy"}:
        from .policy_adapters import SB3AttackerPolicy, SB3DefenderPolicy

        return {
            "SB3AttackerPolicy": SB3AttackerPolicy,
            "SB3DefenderPolicy": SB3DefenderPolicy,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
