"""Shared model types for cybergraph-game."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import networkx as nx

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
    attack_seen_probability: float
    security_level_min: int
    security_level_max: int
    added_node_security_level: int


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


@dataclass(frozen=True)
class AdjustSecurityLevel:
    target: Any
    delta: int


Action = NoAction | AttackNode | RestoreNode | AddNode | AdjustSecurityLevel


@dataclass
class ActionResult:
    actor: Actor
    action: Action
    success: bool
    reason: str
    target: Any | None = None
    probability: float | None = None
    roll: float | None = None
    seen_by_defender: bool = False


@dataclass(frozen=True)
class AttackerKnowledge:
    known_nodes: set = field(default_factory=set)
    known_edges: set = field(default_factory=set)


@dataclass(frozen=True)
class DefenderKnowledge:
    known_captured_nodes: set = field(default_factory=set)
    seen_attack_nodes: set = field(default_factory=set)


@dataclass
class GameState:
    graph: Any
    attacker_knowledge: AttackerKnowledge
    defender_knowledge: DefenderKnowledge
    time_step: int = 0


@dataclass(frozen=True)
class StepResult:
    time_step: int
    attacker_result: ActionResult
    defender_result: ActionResult


def set_node_state(
    graph: nx.Graph,
    node,
    control_state: ControlState,
) -> None:
    validate_node_exists(graph, node)
    graph.nodes[node][CONTROL_STATE] = control_state.value


def get_node_state(graph: nx.Graph, node) -> ControlState | None:
    validate_node_exists(graph, node)
    value = graph.nodes[node].get(CONTROL_STATE)
    try:
        return ControlState(value)
    except ValueError:
        return None


def is_defended(graph: nx.Graph, node) -> bool:
    return get_node_state(graph, node) == ControlState.DEFENDED


def is_captured(graph: nx.Graph, node) -> bool:
    return get_node_state(graph, node) == ControlState.CAPTURED


def is_entry_point(graph: nx.Graph, node) -> bool:
    validate_node_exists(graph, node)
    return bool(graph.nodes[node].get(IS_ENTRY_POINT))


def get_security_level(
    graph: nx.Graph,
    node,
    security_level_min: int,
    security_level_max: int,
) -> int:
    validate_node_exists(graph, node)
    security_level = graph.nodes[node].get(SECURITY_LEVEL)
    if (
        not isinstance(security_level, int)
        or not security_level_min <= security_level <= security_level_max
    ):
        raise ValueError(
            f"Node {node!r} has invalid security_level: {security_level!r}"
        )
    return security_level


def set_security_level(
    graph: nx.Graph,
    node,
    security_level: int,
    security_level_min: int,
    security_level_max: int,
) -> None:
    validate_node_exists(graph, node)
    if not security_level_min <= security_level <= security_level_max:
        raise ValueError(
            f"security_level must be between {security_level_min} and "
            f"{security_level_max}."
        )
    graph.nodes[node][SECURITY_LEVEL] = security_level


def apply_initial_node_attributes(
    graph: nx.Graph,
    entry_points: Iterable,
    security_levels: dict | None = None,
    default_security_level: int | None = None,
) -> None:
    if security_levels is None and default_security_level is None:
        raise ValueError("Provide security_levels or default_security_level.")

    entry_point_set = set(entry_points)
    security_levels = security_levels or {}
    for node in graph.nodes:
        if node not in security_levels and default_security_level is None:
            raise ValueError(f"Missing security level for node: {node!r}")
        graph.nodes[node][CONTROL_STATE] = ControlState.DEFENDED.value
        graph.nodes[node][IS_ENTRY_POINT] = node in entry_point_set
        graph.nodes[node][SECURITY_LEVEL] = security_levels.get(
            node,
            default_security_level,
        )


def validate_node_exists(graph: nx.Graph, node) -> None:
    if node not in graph:
        raise ValueError(f"Node is not in the graph: {node!r}")
