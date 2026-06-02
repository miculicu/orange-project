"""Compatibility helpers for defender actions.

New policy code should prefer observations, actions, and the engine.
"""

import networkx as nx

from .model import ActionResult as DefenderResult
from .model import AddNode, GameState, RestoreNode, RuleConfig
from .observations import initial_attacker_knowledge
from .rules import apply_defender_action, get_defendable_nodes
from .state import is_captured, validate_node_exists


def can_defend_node(graph: nx.Graph, node) -> bool:
    validate_node_exists(graph, node)
    return is_captured(graph, node)


def defend_node(graph: nx.Graph, node) -> DefenderResult:
    state = GameState(
        graph=graph,
        attacker_knowledge=initial_attacker_knowledge(graph),
    )
    return apply_defender_action(
        state,
        RestoreNode(target=node),
        RuleConfig(attack_success_probability=1.0),
    )


def add_defended_node(
    graph: nx.Graph,
    neighbors: list | set,
    security_level: int = 3,
    node=None,
) -> DefenderResult:
    state = GameState(
        graph=graph,
        attacker_knowledge=initial_attacker_knowledge(graph),
    )
    return apply_defender_action(
        state,
        AddNode(
            neighbors=list(neighbors),
            security_level=security_level,
            node=node,
        ),
        RuleConfig(
            attack_success_probability=1.0,
            added_node_security_level=security_level,
        ),
    )
