"""Game engine and transition rules for cybergraph-game."""

import random
from typing import Callable

import networkx as nx

from .attacker import (
    AttackerObservation,
    initial_attacker_knowledge,
    make_attacker_observation,
    update_attacker_knowledge_after_attack,
)
from .defender import (
    DefenderObservation,
    initial_defender_knowledge,
    make_defender_observation,
    record_defender_restore_result,
    record_seen_attack,
)
from .model import (
    Actor,
    Action,
    ActionResult,
    AddNode,
    AdjustSecurityLevel,
    AttackNode,
    ControlState,
    GameState,
    NoAction,
    RestoreNode,
    RuleConfig,
    StepResult,
    get_security_level,
    is_captured,
    is_defended,
    is_entry_point,
    set_node_state,
    set_security_level,
    validate_node_exists,
)

AttackerPolicy = Callable[[AttackerObservation, random.Random], Action]
DefenderPolicy = Callable[[DefenderObservation, random.Random], Action]


def initialize_game_state(
    graph: nx.Graph,
    rules: RuleConfig,
) -> GameState:
    """Create initial hidden game state and player knowledge."""
    return GameState(
        graph=graph,
        attacker_knowledge=initial_attacker_knowledge(graph),
        defender_knowledge=initial_defender_knowledge(),
    )


def advance_game(
    state: GameState,
    attacker_policy: AttackerPolicy,
    defender_policy: DefenderPolicy,
    rules: RuleConfig,
    rng: random.Random,
) -> StepResult:
    """Advance one time step by asking both players for actions."""
    attacker_observation = make_attacker_observation(
        state,
        legal_attack_targets=get_attackable_nodes(state.graph),
    )
    defender_observation = make_defender_observation(state)

    attacker_action = attacker_policy(attacker_observation, rng)
    defender_action = defender_policy(defender_observation, rng)

    attacker_result = apply_attacker_action(state, attacker_action, rules, rng)
    _update_knowledge_after_attacker_action(state, attacker_result, rules, rng)

    defender_result = apply_defender_action(state, defender_action, rules)
    if isinstance(defender_action, RestoreNode):
        record_defender_restore_result(
            state,
            defender_action.target,
            defender_result.success,
        )

    state.time_step += 1
    return StepResult(
        time_step=state.time_step,
        attacker_result=attacker_result,
        defender_result=defender_result,
    )


def get_attackable_nodes(graph: nx.Graph) -> list:
    """Return defended nodes that can be attacked in the true graph."""
    return [node for node in graph.nodes if can_attack_node(graph, node)]


def can_attack_node(graph: nx.Graph, node) -> bool:
    """A node is attackable if defended and reachable from an attacker foothold."""
    validate_node_exists(graph, node)
    if not is_defended(graph, node):
        return False
    if is_entry_point(graph, node):
        return True
    return any(is_captured(graph, neighbor) for neighbor in graph.neighbors(node))


def calculate_attack_success_probability(
    graph: nx.Graph,
    node,
    rules: RuleConfig,
) -> float:
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
    action: Action,
    rules: RuleConfig,
    rng: random.Random,
) -> ActionResult:
    if isinstance(action, NoAction):
        return ActionResult(
            actor=Actor.ATTACKER,
            action=action,
            success=True,
            reason="",
        )
    if not isinstance(action, AttackNode):
        return ActionResult(
            actor=Actor.ATTACKER,
            action=action,
            success=False,
            reason="invalid attacker action",
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
            probability=probability,
            reason="node is already captured",
        )

    if not can_attack_node(graph, node):
        return ActionResult(
            actor=Actor.ATTACKER,
            action=action,
            target=node,
            success=False,
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
        probability=probability,
        roll=roll,
        reason="",
    )


def apply_defender_action(
    state: GameState,
    action: Action,
    rules: RuleConfig,
) -> ActionResult:
    if isinstance(action, NoAction):
        return ActionResult(
            actor=Actor.DEFENDER,
            action=action,
            success=True,
            reason="",
        )
    if isinstance(action, RestoreNode):
        return _restore_node(state.graph, action)
    if isinstance(action, AddNode):
        return _add_node(state.graph, action, rules)
    if isinstance(action, AdjustSecurityLevel):
        return _adjust_security_level(state.graph, action, rules)
    return ActionResult(
        actor=Actor.DEFENDER,
        action=action,
        success=False,
        reason="invalid defender action",
    )


def validate_probability(probability: float) -> None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0.0 and 1.0.")


def _update_knowledge_after_attacker_action(
    state: GameState,
    result: ActionResult,
    rules: RuleConfig,
    rng: random.Random,
) -> None:
    if isinstance(result.action, AttackNode) and result.success and result.target is not None:
        update_attacker_knowledge_after_attack(state, result.target)

    if not isinstance(result.action, AttackNode) or result.target is None:
        return

    validate_probability(rules.attack_seen_probability)
    seen_by_defender = rng.random() < rules.attack_seen_probability
    if seen_by_defender:
        record_seen_attack(state, result.target, attack_succeeded=result.success)

    result.seen_by_defender = seen_by_defender


def _restore_node(graph: nx.Graph, action: RestoreNode) -> ActionResult:
    node = action.target
    validate_node_exists(graph, node)
    if not is_captured(graph, node):
        return ActionResult(
            actor=Actor.DEFENDER,
            action=action,
            target=node,
            success=False,
            reason="node is not captured",
        )

    set_node_state(graph, node, ControlState.DEFENDED)
    return ActionResult(
        actor=Actor.DEFENDER,
        action=action,
        target=node,
        success=True,
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
        reason=f"added defended node connected to {neighbor_list!r}",
    )


def _adjust_security_level(
    graph: nx.Graph,
    action: AdjustSecurityLevel,
    rules: RuleConfig,
) -> ActionResult:
    node = action.target
    validate_node_exists(graph, node)
    old_level = get_security_level(
        graph,
        node,
        rules.security_level_min,
        rules.security_level_max,
    )
    new_level = old_level + action.delta
    if not rules.security_level_min <= new_level <= rules.security_level_max:
        return ActionResult(
            actor=Actor.DEFENDER,
            action=action,
            target=node,
            success=False,
            reason=(
                f"security level would move from {old_level} to {new_level}, "
                "outside allowed range"
            ),
        )

    set_security_level(
        graph,
        node,
        new_level,
        rules.security_level_min,
        rules.security_level_max,
    )
    return ActionResult(
        actor=Actor.DEFENDER,
        action=action,
        target=node,
        success=True,
        reason=f"security level changed from {old_level} to {new_level}",
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
