"""Adapters that turn trained PPO models into fixed game policies."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass
class SB3DefenderPolicy:
    """Fixed defender policy backed by a Stable-Baselines3 model."""

    model: object
    num_nodes: int
    max_defend_nodes: int | None = None
    deterministic: bool = True

    def act(self, belief: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(belief.astype(np.float32), deterministic=self.deterministic)
        return _sanitize_binary_budget(action, self.num_nodes, self.max_defend_nodes)


@dataclass
class SB3AttackerPolicy:
    """Fixed attacker policy backed by a Stable-Baselines3 MultiBinary model.

    The attacker observes the true binary state only. It does not observe the
    defender action; the `defense` argument is accepted for the belief-update
    protocol and intentionally ignored.
    """

    model: object
    num_nodes: int
    max_attack_nodes: int | None = None
    deterministic: bool = True

    def sample(
        self,
        state: np.ndarray,
        defense: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del defense, rng
        action, _ = self.model.predict(state.astype(np.float32), deterministic=self.deterministic)
        return _sanitize_binary_budget(action, self.num_nodes, self.max_attack_nodes)

    def probability(
        self,
        attack: np.ndarray,
        state: np.ndarray,
        defense: np.ndarray,
    ) -> float:
        del defense
        attack = _sanitize_binary_budget(attack, self.num_nodes, self.max_attack_nodes)
        bit_probs = self._bit_probabilities(state)
        probability = 0.0
        for raw_action in _all_binary_actions(self.num_nodes):
            sanitized = _sanitize_binary_budget(raw_action, self.num_nodes, self.max_attack_nodes)
            if np.array_equal(sanitized, attack):
                probability += _independent_bernoulli_probability(raw_action, bit_probs)
        return float(probability)

    def _bit_probabilities(self, state: np.ndarray) -> np.ndarray:
        import torch

        observation = state.astype(np.float32)
        obs_tensor, _ = self.model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            bernoulli = distribution.distribution
            probs = bernoulli.probs.detach().cpu().numpy()
        return probs.reshape(-1)[: self.num_nodes]


def _sanitize_binary_budget(
    action: np.ndarray,
    num_nodes: int,
    budget: int | None,
) -> np.ndarray:
    sanitized = np.asarray(action, dtype=np.int8).reshape(num_nodes)
    sanitized = (sanitized > 0).astype(np.int8)
    if budget is None:
        return sanitized
    budget = max(0, budget)
    active = np.flatnonzero(sanitized)
    if len(active) > budget:
        sanitized[:] = 0
        sanitized[active[:budget]] = 1
    return sanitized


def _all_binary_actions(num_nodes: int):
    for bits in product([0, 1], repeat=num_nodes):
        yield np.array(bits, dtype=np.int8)


def _independent_bernoulli_probability(action: np.ndarray, bit_probs: np.ndarray) -> float:
    probs = np.where(action == 1, bit_probs, 1.0 - bit_probs)
    return float(np.prod(probs, dtype=np.float64))
