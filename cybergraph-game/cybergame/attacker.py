"""Compatibility helpers for attacker actions.

New policy code should prefer observations, actions, and the engine.
"""

import random

import networkx as nx

from .model import ActionResult as AttackResult
from .model import AttackNode, GameState, RuleConfig
from .observations import initial_attacker_knowledge
from .rules import (
    apply_attacker_action,
    calculate_attack_success_probability as _calculate_probability,
    can_attack_node,
    get_attackable_nodes,
)


def calculate_attack_success_probability(
    graph: nx.Graph,
    node,
    global_success_probability: float,
) -> float:
    return _calculate_probability(
        graph,
        node,
        RuleConfig(attack_success_probability=global_success_probability),
    )


def attack_node(
    graph: nx.Graph,
    node,
    success_probability: float,
    rng: random.Random | None = None,
) -> AttackResult:
    state = GameState(
        graph=graph,
        attacker_knowledge=initial_attacker_knowledge(graph),
    )
    return apply_attacker_action(
        state,
        AttackNode(target=node),
        RuleConfig(attack_success_probability=success_probability),
        rng or random.Random(),
    )
