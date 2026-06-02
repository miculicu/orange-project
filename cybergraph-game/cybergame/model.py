"""Shared model types for cybergraph-game."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

CONTROL_STATE = "control_state"
IS_ENTRY_POINT = "is_entry_point"
SECURITY_LEVEL = "security_level"


class ControlState(str, Enum):
    DEFENDED = "defended"
    CAPTURED = "captured"


class Actor(str, Enum):
    ATTACKER = "attacker"
    DEFENDER = "defender"


@dataclass(frozen=True)
class RuleConfig:
    attack_success_probability: float
    security_level_min: int = 1
    security_level_max: int = 3
    default_security_level: int = 1
    added_node_security_level: int = 3


@dataclass(frozen=True)
class NoAction:
    reason: str = "no action"


@dataclass(frozen=True)
class AttackNode:
    target: Any


@dataclass(frozen=True)
class RestoreNode:
    target: Any


@dataclass(frozen=True)
class AddNode:
    neighbors: list
    security_level: int | None = None
    node: Any | None = None


Action = NoAction | AttackNode | RestoreNode | AddNode


@dataclass(frozen=True)
class ActionResult:
    actor: Actor
    action: Action
    success: bool
    changed: bool
    reason: str
    target: Any | None = None
    probability: float | None = None
    roll: float | None = None


@dataclass(frozen=True)
class AttackerKnowledge:
    known_nodes: set
    known_edges: set


@dataclass
class GameState:
    graph: Any
    attacker_knowledge: AttackerKnowledge
    time_step: int = 0


@dataclass(frozen=True)
class StepResult:
    time_step: int
    attacker_result: ActionResult
    defender_result: ActionResult
