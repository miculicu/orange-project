"""Compatibility time-step helper for direct action execution."""

import random
from typing import Any

import networkx as nx

from .model import AddNode, AttackNode, GameState, NoAction, RestoreNode, RuleConfig
from .model import StepResult as TimeStepResult
from .observations import initial_attacker_knowledge, update_attacker_knowledge_after_result
from .rules import apply_attacker_action, apply_defender_action


def advance_time_step(
    graph: nx.Graph,
    time_step: int,
    attack_success_probability: float,
    attacker_target: Any | None = None,
    defender_action: str | None = None,
    defender_target: Any | None = None,
    defender_neighbors: list | set | None = None,
    defender_security_level: int | None = None,
    rng: random.Random | None = None,
) -> TimeStepResult:
    """Apply one attacker action and one defender action for a time step."""
    if time_step < 0:
        raise ValueError("time_step must be non-negative.")
    if defender_action not in {None, "restore", "add_node"}:
        raise ValueError("defender_action must be None, 'restore', or 'add_node'.")

    state = GameState(
        graph=graph,
        attacker_knowledge=initial_attacker_knowledge(graph),
        time_step=time_step - 1,
    )
    rules = RuleConfig(attack_success_probability=attack_success_probability)
    rng = rng or random.Random()

    attacker_action = (
        AttackNode(target=attacker_target)
        if attacker_target is not None
        else NoAction("attacker chose no action")
    )

    defender_action_object = NoAction("defender chose no action")
    if defender_action == "restore" and defender_target is not None:
        defender_action_object = RestoreNode(target=defender_target)
    elif defender_action == "add_node" and defender_neighbors is not None:
        defender_action_object = AddNode(
            neighbors=list(defender_neighbors),
            security_level=defender_security_level,
        )

    attacker_result = apply_attacker_action(state, attacker_action, rules, rng)
    defender_result = apply_defender_action(state, defender_action_object, rules)
    state.time_step = time_step
    update_attacker_knowledge_after_result(state, attacker_result)

    return TimeStepResult(
        time_step=time_step,
        attacker_result=attacker_result,
        defender_result=defender_result,
    )
