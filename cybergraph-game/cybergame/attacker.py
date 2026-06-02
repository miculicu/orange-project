"""Attacker knowledge, observations, and policies."""

from dataclasses import dataclass
import random

import networkx as nx

from .model import AttackNode, AttackerKnowledge, GameState, NoAction
from .model import is_captured, is_defended, is_entry_point


@dataclass(frozen=True)
class AttackerObservation:
    time_step: int
    known_nodes: set
    known_edges: set
    known_entry_points: set
    known_captured_nodes: set
    known_defended_nodes: set
    legal_attack_targets: list


def initial_attacker_knowledge(
    graph: nx.Graph,
) -> AttackerKnowledge:
    """Reveal only entry points to the attacker."""
    known_nodes = {
        node for node, data in graph.nodes(data=True) if data.get("is_entry_point")
    }
    return _knowledge_from_nodes(graph, known_nodes)


def update_attacker_knowledge_after_attack(
    state: GameState,
    captured_node,
) -> None:
    """Reveal neighbors of a captured node while preserving old knowledge."""
    known_nodes = set(state.attacker_knowledge.known_nodes)
    known_nodes.add(captured_node)
    known_nodes.update(state.graph.neighbors(captured_node))
    state.attacker_knowledge = _knowledge_from_nodes(state.graph, known_nodes)


def make_attacker_observation(
    state: GameState,
    legal_attack_targets: list,
) -> AttackerObservation:
    """Build the attacker's limited view from attacker knowledge."""
    graph = state.graph
    known_nodes = set(state.attacker_knowledge.known_nodes)
    known_legal_targets = [
        node for node in legal_attack_targets if node in known_nodes
    ]

    return AttackerObservation(
        time_step=state.time_step,
        known_nodes=known_nodes,
        known_edges=set(state.attacker_knowledge.known_edges),
        known_entry_points={
            node for node in known_nodes if is_entry_point(graph, node)
        },
        known_captured_nodes={
            node for node in known_nodes if is_captured(graph, node)
        },
        known_defended_nodes={
            node for node in known_nodes if is_defended(graph, node)
        },
        legal_attack_targets=known_legal_targets,
    )


def always_attack_policy(
    observation: AttackerObservation,
    rng: random.Random,
):
    """Always attack a known legal target, choosing stably by node label."""
    if not observation.legal_attack_targets:
        return NoAction("no known attackable nodes")
    return AttackNode(target=sorted(observation.legal_attack_targets, key=str)[0])


def random_attacker_policy(
    observation: AttackerObservation,
    rng: random.Random,
):
    """Attack a random known legal target."""
    if not observation.legal_attack_targets:
        return NoAction("no known attackable nodes")
    return AttackNode(target=rng.choice(observation.legal_attack_targets))


def do_nothing_attacker_policy(
    observation: AttackerObservation,
    rng: random.Random,
):
    """Choose no attacker action."""
    return NoAction("attacker policy chose no action")


def _knowledge_from_nodes(graph: nx.Graph, known_nodes: set) -> AttackerKnowledge:
    known_edges = {
        tuple(sorted(edge, key=str))
        for edge in graph.edges
        if edge[0] in known_nodes and edge[1] in known_nodes
    }
    return AttackerKnowledge(known_nodes=known_nodes, known_edges=known_edges)
