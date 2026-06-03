"""Exact defender belief updates for the binary graph game."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Protocol

import numpy as np


class AttackerPolicy(Protocol):
    """Fixed attacker policy pi_A(A | s, D)."""

    def probability(self, attack: np.ndarray, state: np.ndarray, defense: np.ndarray) -> float:
        ...


def enumerate_binary_states(num_nodes: int) -> np.ndarray:
    """Return all vectors in {0, 1}^n as an array of shape (2^n, n)."""
    if num_nodes < 1:
        raise ValueError("num_nodes must be at least 1.")
    return np.array(list(product([0, 1], repeat=num_nodes)), dtype=np.int8)


@dataclass(frozen=True)
class BeliefUpdater:
    """Bayes filter for the defender's belief over all binary graph states."""

    num_nodes: int
    beta: np.ndarray
    probe_miss_probability: float
    attacker_policy: AttackerPolicy
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        beta = np.asarray(self.beta, dtype=np.float64)
        if beta.shape != (self.num_nodes,):
            raise ValueError("beta must have shape (num_nodes,).")
        if np.any((beta < 0.0) | (beta > 1.0)):
            raise ValueError("beta entries must be between 0 and 1.")
        if not 0.0 <= self.probe_miss_probability <= 1.0:
            raise ValueError("probe_miss_probability must be between 0 and 1.")

    @property
    def states(self) -> np.ndarray:
        return enumerate_binary_states(self.num_nodes)

    @property
    def actions(self) -> np.ndarray:
        return enumerate_binary_states(self.num_nodes)

    def update(
        self,
        belief: np.ndarray,
        observation: np.ndarray,
        defense: np.ndarray,
    ) -> np.ndarray:
        """Return b_{k+1} after observing detected probes Y and action D."""
        belief = _as_binary_distribution(belief, 2**self.num_nodes, "belief")
        observation = _as_binary_vector(observation, self.num_nodes, "observation")
        defense = _as_binary_vector(defense, self.num_nodes, "defense")

        prior_states = self.states
        next_states = self.states
        attacks = self.actions
        posterior = np.zeros(len(next_states), dtype=np.float64)

        for state_index, state in enumerate(prior_states):
            prior_mass = belief[state_index]
            if prior_mass <= 0.0:
                continue

            for attack in attacks:
                policy_prob = self.attacker_policy.probability(attack, state, defense)
                if policy_prob <= 0.0:
                    continue

                likelihood = observation_likelihood(
                    observation,
                    attack,
                    self.probe_miss_probability,
                )
                if likelihood <= 0.0:
                    continue

                q = node_compromise_probabilities(state, attack, defense, self.beta)
                transition_probs = factorized_transition_probabilities(next_states, q)
                posterior += prior_mass * policy_prob * likelihood * transition_probs

        normalizer = posterior.sum()
        if normalizer <= self.epsilon:
            return np.full_like(posterior, 1.0 / len(posterior))
        return posterior / normalizer


def node_compromise_probabilities(
    state: np.ndarray,
    attack: np.ndarray,
    defense: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Compute q_v(s, A, D) for every node v."""
    state = np.asarray(state, dtype=np.int8)
    attack = np.asarray(attack, dtype=np.int8)
    defense = np.asarray(defense, dtype=np.int8)
    beta = np.asarray(beta, dtype=np.float64)

    q = np.zeros_like(beta, dtype=np.float64)
    not_defended = defense == 0
    q[not_defended & (state == 1)] = 1.0
    q[not_defended & (state == 0) & (attack == 1)] = beta[
        not_defended & (state == 0) & (attack == 1)
    ]
    return q


def factorized_transition_probabilities(
    next_states: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Compute K^{A,D}(s' | s) for all candidate next states."""
    return np.prod(
        np.where(next_states == 1, q, 1.0 - q),
        axis=1,
        dtype=np.float64,
    )


def observation_likelihood(
    observation: np.ndarray,
    attack: np.ndarray,
    probe_miss_probability: float,
) -> float:
    """Compute L(Y | A) under no false positives and independent misses."""
    observation = np.asarray(observation, dtype=np.int8)
    attack = np.asarray(attack, dtype=np.int8)
    if np.any((observation == 1) & (attack == 0)):
        return 0.0
    detected = int(observation.sum())
    missed = int(attack.sum() - detected)
    return (1.0 - probe_miss_probability) ** detected * probe_miss_probability**missed


def _as_binary_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.int8)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    if np.any((array != 0) & (array != 1)):
        raise ValueError(f"{name} must be binary.")
    return array


def _as_binary_distribution(value: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    if np.any(array < 0.0):
        raise ValueError(f"{name} cannot contain negative probabilities.")
    total = array.sum()
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass.")
    return array / total

