"""Defender knowledge, observations, and policies."""

from dataclasses import dataclass
import random

import networkx as nx

from .model import (
    CONTROL_STATE,
    AddNode,
    AdjustSecurityLevel,
    ControlState,
    DefenderKnowledge,
    GameState,
    NoAction,
    RestoreNode,
)


@dataclass(frozen=True)
class DefenderObservation:
    time_step: int
    graph: nx.Graph
    known_captured_nodes: set
    seen_attack_nodes: set
    add_node_candidates: list
    security_adjust_candidates: list


def initial_defender_knowledge() -> DefenderKnowledge:
    """The defender starts with topology/security knowledge but no capture alerts."""
    return DefenderKnowledge()


def make_defender_observation(state: GameState) -> DefenderObservation:
    """Build the defender view: full graph existence, hidden capture state."""
    visible_graph = state.graph.copy()
    for node in visible_graph.nodes:
        visible_graph.nodes[node][CONTROL_STATE] = ControlState.DEFENDED.value

    return DefenderObservation(
        time_step=state.time_step,
        graph=visible_graph,
        known_captured_nodes=set(state.defender_knowledge.known_captured_nodes),
        seen_attack_nodes=set(state.defender_knowledge.seen_attack_nodes),
        add_node_candidates=list(visible_graph.nodes),
        security_adjust_candidates=list(visible_graph.nodes),
    )


def record_seen_attack(
    state: GameState,
    target,
    attack_succeeded: bool,
) -> None:
    seen_attack_nodes = set(state.defender_knowledge.seen_attack_nodes)
    known_captured_nodes = set(state.defender_knowledge.known_captured_nodes)
    seen_attack_nodes.add(target)
    if attack_succeeded:
        known_captured_nodes.add(target)
    state.defender_knowledge = DefenderKnowledge(
        known_captured_nodes=known_captured_nodes,
        seen_attack_nodes=seen_attack_nodes,
    )


def record_defender_restore_result(state: GameState, target, success: bool) -> None:
    if not success:
        return
    known_captured_nodes = set(state.defender_knowledge.known_captured_nodes)
    known_captured_nodes.discard(target)
    state.defender_knowledge = DefenderKnowledge(
        known_captured_nodes=known_captured_nodes,
        seen_attack_nodes=set(state.defender_knowledge.seen_attack_nodes),
    )


def random_defender_policy_factory(
    action_probability: float,
    add_node_probability: float,
    adjust_security_probability: float,
    added_node_edge_count: int,
    added_node_security_level: int,
    security_adjust_delta: int,
):
    """Create a random defender policy with scenario-local settings."""

    def policy(observation: DefenderObservation, rng: random.Random):
        if rng.random() >= action_probability:
            return NoAction("defender chose no action")

        should_add_node = rng.random() < add_node_probability
        if observation.known_captured_nodes and not should_add_node:
            return RestoreNode(
                target=rng.choice(sorted(observation.known_captured_nodes, key=str))
            )

        should_adjust_security = rng.random() < adjust_security_probability
        if observation.security_adjust_candidates and should_adjust_security:
            return AdjustSecurityLevel(
                target=rng.choice(observation.security_adjust_candidates),
                delta=security_adjust_delta,
            )

        edge_count = min(added_node_edge_count, len(observation.add_node_candidates))
        neighbors = rng.sample(observation.add_node_candidates, edge_count)
        return AddNode(
            neighbors=neighbors,
            security_level=added_node_security_level,
        )

    return policy


def random_reimage_policy_factory(action_probability: float):
    """Create a static-graph defender policy: reimage known captures or do nothing."""

    def policy(observation: DefenderObservation, rng: random.Random):
        if rng.random() >= action_probability:
            return NoAction("defender chose no action")
        if not observation.known_captured_nodes:
            return NoAction("no known captured nodes to reimage")
        return RestoreNode(
            target=rng.choice(sorted(observation.known_captured_nodes, key=str))
        )

    return policy


def do_nothing_defender_policy(
    observation: DefenderObservation,
    rng: random.Random,
):
    """Choose no defender action."""
    return NoAction("defender policy chose no action")
