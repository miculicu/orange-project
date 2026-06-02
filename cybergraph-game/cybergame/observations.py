"""Observation and knowledge models for policies."""

from dataclasses import dataclass

import networkx as nx

from .model import AttackerKnowledge, GameState
from .rules import get_attackable_nodes, get_defendable_nodes
from .state import is_captured, is_defended, is_entry_point


@dataclass(frozen=True)
class AttackerObservation:
    time_step: int
    known_nodes: set
    known_edges: set
    known_entry_points: set
    known_captured_nodes: set
    known_defended_nodes: set
    legal_attack_targets: list


@dataclass(frozen=True)
class DefenderObservation:
    time_step: int
    graph: nx.Graph
    legal_restore_targets: list
    add_node_candidates: list


def initial_attacker_knowledge(graph: nx.Graph) -> AttackerKnowledge:
    """Reveal entry points and their immediate neighbors to the attacker."""
    known_nodes = {
        node for node, data in graph.nodes(data=True) if data.get("is_entry_point")
    }
    for entry_point in list(known_nodes):
        known_nodes.update(graph.neighbors(entry_point))
    return _knowledge_from_nodes(graph, known_nodes)


def update_attacker_knowledge_after_result(
    state: GameState,
    attack_result,
) -> None:
    """Reveal neighbors of successfully captured nodes and preserve old knowledge."""
    if not attack_result.success or attack_result.target is None:
        return

    known_nodes = set(state.attacker_knowledge.known_nodes)
    known_nodes.add(attack_result.target)
    known_nodes.update(state.graph.neighbors(attack_result.target))
    state.attacker_knowledge = _knowledge_from_nodes(state.graph, known_nodes)


def make_attacker_observation(state: GameState) -> AttackerObservation:
    graph = state.graph
    knowledge = state.attacker_knowledge
    known_nodes = set(knowledge.known_nodes)
    legal_targets = [
        node for node in get_attackable_nodes(graph) if node in known_nodes
    ]

    return AttackerObservation(
        time_step=state.time_step,
        known_nodes=known_nodes,
        known_edges=set(knowledge.known_edges),
        known_entry_points={
            node for node in known_nodes if is_entry_point(graph, node)
        },
        known_captured_nodes={
            node for node in known_nodes if is_captured(graph, node)
        },
        known_defended_nodes={
            node for node in known_nodes if is_defended(graph, node)
        },
        legal_attack_targets=legal_targets,
    )


def make_defender_observation(state: GameState) -> DefenderObservation:
    graph = state.graph
    return DefenderObservation(
        time_step=state.time_step,
        graph=graph.copy(),
        legal_restore_targets=get_defendable_nodes(graph),
        add_node_candidates=list(graph.nodes),
    )


def _knowledge_from_nodes(graph: nx.Graph, known_nodes: set) -> AttackerKnowledge:
    known_edges = {
        tuple(sorted(edge, key=str))
        for edge in graph.edges
        if edge[0] in known_nodes and edge[1] in known_nodes
    }
    return AttackerKnowledge(known_nodes=known_nodes, known_edges=known_edges)
