"""Graph state helpers for cybergraph-game."""

from typing import Iterable

import networkx as nx

from .model import CONTROL_STATE, IS_ENTRY_POINT, SECURITY_LEVEL, ControlState


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


def apply_initial_node_attributes(
    graph: nx.Graph,
    entry_points: Iterable,
    security_levels: dict | None = None,
    default_security_level: int = 1,
) -> None:
    entry_point_set = set(entry_points)
    security_levels = security_levels or {}
    for node in graph.nodes:
        graph.nodes[node][CONTROL_STATE] = ControlState.DEFENDED.value
        graph.nodes[node][IS_ENTRY_POINT] = node in entry_point_set
        graph.nodes[node][SECURITY_LEVEL] = security_levels.get(
            node,
            default_security_level,
        )


def validate_node_exists(graph: nx.Graph, node) -> None:
    if node not in graph:
        raise ValueError(f"Node is not in the graph: {node!r}")
