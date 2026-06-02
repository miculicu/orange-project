"""Simulation engine for cybergraph-game."""

import random
from typing import Callable

from .model import GameState, StepResult
from .observations import (
    AttackerObservation,
    DefenderObservation,
    make_attacker_observation,
    make_defender_observation,
    update_attacker_knowledge_after_result,
)
from .rules import apply_attacker_action, apply_defender_action
from .model import Action, RuleConfig

AttackerPolicy = Callable[[AttackerObservation, random.Random], Action]
DefenderPolicy = Callable[[DefenderObservation, random.Random], Action]


def advance_with_policies(
    state: GameState,
    attacker_policy: AttackerPolicy,
    defender_policy: DefenderPolicy,
    rules: RuleConfig,
    rng: random.Random,
) -> StepResult:
    """Advance one time step using policy decisions from limited observations."""
    attacker_observation = make_attacker_observation(state)
    defender_observation = make_defender_observation(state)

    attacker_action = attacker_policy(attacker_observation, rng)
    defender_action = defender_policy(defender_observation, rng)

    next_time_step = state.time_step + 1
    attacker_result = apply_attacker_action(state, attacker_action, rules, rng)
    defender_result = apply_defender_action(state, defender_action, rules)

    state.time_step = next_time_step
    update_attacker_knowledge_after_result(state, attacker_result)

    return StepResult(
        time_step=state.time_step,
        attacker_result=attacker_result,
        defender_result=defender_result,
    )
