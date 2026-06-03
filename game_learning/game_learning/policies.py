"""Fixed attacker policies for defender-side RL experiments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class UniformAttackerPolicy:
    """Uniformly choose a subset of currently allowed nodes up to a budget."""

    num_nodes: int
    max_attack_nodes: int = 1
    clean_only: bool = True

    def __post_init__(self) -> None:
        if self.num_nodes < 1:
            raise ValueError("num_nodes must be at least 1.")
        if self.max_attack_nodes < 0:
            raise ValueError("max_attack_nodes cannot be negative.")

    def sample(
        self,
        state: np.ndarray,
        defense: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        allowed = self._allowed_nodes(state, defense)
        candidates = self._candidate_actions(allowed)
        index = int(rng.integers(0, len(candidates)))
        return candidates[index].copy()

    def probability(
        self,
        attack: np.ndarray,
        state: np.ndarray,
        defense: np.ndarray,
    ) -> float:
        attack = np.asarray(attack, dtype=np.int8)
        allowed = self._allowed_nodes(state, defense)
        if not self._is_valid_attack(attack, allowed):
            return 0.0
        return 1.0 / len(self._candidate_actions(allowed))

    def _allowed_nodes(self, state: np.ndarray, defense: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.int8)
        defense = np.asarray(defense, dtype=np.int8)
        allowed = defense == 0
        if self.clean_only:
            allowed &= state == 0
        return np.flatnonzero(allowed)

    def _candidate_actions(self, allowed: np.ndarray) -> list[np.ndarray]:
        max_size = min(self.max_attack_nodes, len(allowed))
        candidates: list[np.ndarray] = []
        for size in range(max_size + 1):
            for nodes in combinations(allowed.tolist(), size):
                action = np.zeros(self.num_nodes, dtype=np.int8)
                action[list(nodes)] = 1
                candidates.append(action)
        return candidates

    def _is_valid_attack(self, attack: np.ndarray, allowed: np.ndarray) -> bool:
        if attack.shape != (self.num_nodes,):
            return False
        if np.any((attack != 0) & (attack != 1)):
            return False
        if int(attack.sum()) > self.max_attack_nodes:
            return False
        disallowed = np.ones(self.num_nodes, dtype=bool)
        disallowed[allowed] = False
        return not bool(np.any((attack == 1) & disallowed))


@dataclass(frozen=True)
class RandomDefenderPolicy:
    """Small baseline policy for examples outside RL training."""

    num_nodes: int
    max_defend_nodes: int = 1

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        action = np.zeros(self.num_nodes, dtype=np.int8)
        budget = min(self.max_defend_nodes, self.num_nodes)
        if budget == 0:
            return action
        size = int(rng.integers(0, budget + 1))
        if size:
            nodes = rng.choice(self.num_nodes, size=size, replace=False)
            action[nodes] = 1
        return action

