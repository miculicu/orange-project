"""Simple policies that consume observations and return actions."""

import random

from .config import (
    ADDED_NODE_EDGE_COUNT,
    ADDED_NODE_SECURITY_LEVEL,
    DEFENDER_ACTION_PROBABILITY,
    DEFENDER_ADD_NODE_PROBABILITY,
)
from .model import AddNode, AttackNode, NoAction, RestoreNode
from .observations import AttackerObservation, DefenderObservation


def random_attacker_policy(
    observation: AttackerObservation,
    rng: random.Random,
):
    """Attack a random legal target known to the attacker."""
    if not observation.legal_attack_targets:
        return NoAction("no known attackable nodes")
    return AttackNode(target=rng.choice(observation.legal_attack_targets))


def always_attack_policy(
    observation: AttackerObservation,
    rng: random.Random,
):
    """Always attack a known legal target, choosing stably by node label."""
    if not observation.legal_attack_targets:
        return NoAction("no known attackable nodes")
    return AttackNode(target=sorted(observation.legal_attack_targets, key=str)[0])


def random_defender_policy(
    observation: DefenderObservation,
    rng: random.Random,
):
    """Randomly restore a captured node, add a node, or do nothing."""
    policy = make_random_defender_policy(
        action_probability=DEFENDER_ACTION_PROBABILITY,
        add_node_probability=DEFENDER_ADD_NODE_PROBABILITY,
        added_node_edge_count=ADDED_NODE_EDGE_COUNT,
        added_node_security_level=ADDED_NODE_SECURITY_LEVEL,
    )
    return policy(observation, rng)


def make_random_defender_policy(
    action_probability: float,
    add_node_probability: float,
    added_node_edge_count: int,
    added_node_security_level: int,
):
    """Create a random defender policy with scenario-local settings."""

    def policy(observation: DefenderObservation, rng: random.Random):
        if rng.random() >= action_probability:
            return NoAction("defender chose no action")

        should_add_node = rng.random() < add_node_probability
        if observation.legal_restore_targets and not should_add_node:
            return RestoreNode(target=rng.choice(observation.legal_restore_targets))

        edge_count = min(added_node_edge_count, len(observation.add_node_candidates))
        neighbors = rng.sample(observation.add_node_candidates, edge_count)
        return AddNode(
            neighbors=neighbors,
            security_level=added_node_security_level,
        )

    return policy


def never_defend_policy(
    observation: DefenderObservation,
    rng: random.Random,
):
    """Choose no defender action."""
    return NoAction("defender never defends")
