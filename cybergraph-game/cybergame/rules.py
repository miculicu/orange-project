"""Game rules and state transitions."""

import random

import networkx as nx

from .model import (
    Actor,
    ActionResult,
    AddNode,
    AttackNode,
    ControlState,
    GameState,
    NoAction,
    RestoreNode,
    RuleConfig,
)
from .state import (
    get_security_level,
    is_captured,
    is_defended,
    is_entry_point,
    set_node_state,
    validate_node_exists,
)


def get_attackable_nodes(graph: nx.Graph) -> list:
    """Return defended nodes that can be attacked in the true graph."""
    return [node for node in graph.nodes if can_attack_node(graph, node)]


def can_attack_node(graph: nx.Graph, node) -> bool:
    """A node is attackable if defended and entry-adjacent to attacker foothold."""
    validate_node_exists(graph, node)
    if not is_defended(graph, node):
        return False
    if is_entry_point(graph, node):
        return True
    return any(is_captured(graph, neighbor) for neighbor in graph.neighbors(node))


def get_defendable_nodes(graph: nx.Graph) -> list:
    """Return captured nodes that the defender can restore."""
    return [node for node in graph.nodes if is_captured(graph, node)]


def calculate_attack_success_probability(
    graph: nx.Graph,
    node,
    rules: RuleConfig,
) -> float:
    """Compute target-specific success probability from global probability."""
    validate_probability(rules.attack_success_probability)
    security_level = get_security_level(
        graph,
        node,
        rules.security_level_min,
        rules.security_level_max,
    )
    return rules.attack_success_probability / security_level


def apply_attacker_action(
    state: GameState,
    action: AttackNode | NoAction,
    rules: RuleConfig,
    rng: random.Random,
) -> ActionResult:
    if isinstance(action, NoAction):
        return ActionResult(
            actor=Actor.ATTACKER,
            action=action,
            success=False,
            changed=False,
            reason=action.reason,
        )

    graph = state.graph
    node = action.target
    validate_node_exists(graph, node)
    probability = calculate_attack_success_probability(graph, node, rules)

    if is_captured(graph, node):
        return ActionResult(
            actor=Actor.ATTACKER,
            action=action,
            target=node,
            success=False,
            changed=False,
            probability=probability,
            reason="node is already captured",
        )

    if not can_attack_node(graph, node):
        return ActionResult(
            actor=Actor.ATTACKER,
            action=action,
            target=node,
            success=False,
            changed=False,
            probability=probability,
            reason="node is not currently attackable",
        )

    roll = rng.random()
    success = roll < probability
    if success:
        set_node_state(graph, node, ControlState.CAPTURED)

    return ActionResult(
        actor=Actor.ATTACKER,
        action=action,
        target=node,
        success=success,
        changed=success,
        probability=probability,
        roll=roll,
        reason="attack succeeded" if success else "attack failed",
    )


def apply_defender_action(
    state: GameState,
    action: RestoreNode | AddNode | NoAction,
    rules: RuleConfig,
) -> ActionResult:
    if isinstance(action, NoAction):
        return ActionResult(
            actor=Actor.DEFENDER,
            action=action,
            success=False,
            changed=False,
            reason=action.reason,
        )
    if isinstance(action, RestoreNode):
        return _restore_node(state.graph, action)
    return _add_node(state.graph, action, rules)


def validate_probability(probability: float) -> None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("attack_success_probability must be between 0.0 and 1.0.")


def _restore_node(graph: nx.Graph, action: RestoreNode) -> ActionResult:
    node = action.target
    validate_node_exists(graph, node)
    if not is_captured(graph, node):
        return ActionResult(
            actor=Actor.DEFENDER,
            action=action,
            target=node,
            success=False,
            changed=False,
            reason="node is not captured",
        )

    set_node_state(graph, node, ControlState.DEFENDED)
    return ActionResult(
        actor=Actor.DEFENDER,
        action=action,
        target=node,
        success=True,
        changed=True,
        reason="node restored to defended",
    )


def _add_node(graph: nx.Graph, action: AddNode, rules: RuleConfig) -> ActionResult:
    neighbor_list = list(action.neighbors)
    if not neighbor_list:
        raise ValueError("neighbors must contain at least one existing node.")

    invalid_neighbors = [neighbor for neighbor in neighbor_list if neighbor not in graph]
    if invalid_neighbors:
        raise ValueError(f"Neighbors are not in the graph: {invalid_neighbors!r}")

    security_level = action.security_level or rules.added_node_security_level
    if not rules.security_level_min <= security_level <= rules.security_level_max:
        raise ValueError(
            f"security_level must be between {rules.security_level_min} and "
            f"{rules.security_level_max}."
        )

    new_node = _next_node_id(graph) if action.node is None else action.node
    if new_node in graph:
        raise ValueError(f"New node already exists in the graph: {new_node!r}")

    graph.add_node(
        new_node,
        control_state=ControlState.DEFENDED.value,
        is_entry_point=False,
        security_level=security_level,
    )
    graph.add_edges_from((new_node, neighbor) for neighbor in neighbor_list)

    return ActionResult(
        actor=Actor.DEFENDER,
        action=action,
        target=new_node,
        success=True,
        changed=True,
        reason=f"added defended node connected to {neighbor_list!r}",
    )


def _next_node_id(graph: nx.Graph):
    if all(isinstance(node, int) for node in graph.nodes):
        return max(graph.nodes, default=-1) + 1
    index = graph.number_of_nodes()
    candidate = f"defender_node_{index}"
    while candidate in graph:
        index += 1
        candidate = f"defender_node_{index}"
    return candidate
