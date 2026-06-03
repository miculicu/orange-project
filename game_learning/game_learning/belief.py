"""Exact defender belief updates for the edge-free multi-attacker graph game.

The dynamics follow the Moving Target Defence model of Datar & Dujardin
("Adaptive Learning for Moving Target Defence", CoDIT 2025): probing a clean
node compromises it with probability ``1 - exp(-alpha * rho)`` for ``rho``
probes, and the process is memoryless so simultaneous probes simply add up.
With several attackers, ``rho`` is the number of probes landing on a node in a
single step, so coordinated attacks on the same node are strictly more likely
to succeed than spread-out ones.

Nodes are independent: there are no edges, so the transition and observation
kernels both factorise over nodes given the joint attack.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import comb
from typing import Protocol

import numpy as np


class AttackerModel(Protocol):
    """Joint attacker model pi_A(rho | s, D) over per-node probe counts."""

    def joint_action_distribution(
        self, state: np.ndarray, defense: np.ndarray
    ) -> list[tuple[np.ndarray, float]]:
        """Return ``[(probe_counts, probability), ...]`` with positive mass.

        ``probe_counts`` is an integer array of shape ``(num_nodes,)`` giving the
        number of attackers probing each node; ``probability`` is the chance the
        attacker ensemble plays it given ``state`` and ``defense``.
        """
        ...


def enumerate_binary_states(num_nodes: int) -> np.ndarray:
    """Return all vectors in {0, 1}^n as an array of shape (2^n, n)."""
    if num_nodes < 1:
        raise ValueError("num_nodes must be at least 1.")
    return np.array(list(product([0, 1], repeat=num_nodes)), dtype=np.int8)


@dataclass(frozen=True)
class BeliefUpdater:
    """Bayes filter for the defender's belief over all binary node states."""

    num_nodes: int
    alpha: np.ndarray
    probe_miss_probability: float
    attacker_model: AttackerModel
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        alpha = np.asarray(self.alpha, dtype=np.float64)
        if alpha.shape != (self.num_nodes,):
            raise ValueError("alpha must have shape (num_nodes,).")
        if np.any(alpha < 0.0):
            raise ValueError("alpha entries must be non-negative.")
        if not 0.0 <= self.probe_miss_probability <= 1.0:
            raise ValueError("probe_miss_probability must be between 0 and 1.")

    @property
    def states(self) -> np.ndarray:
        return enumerate_binary_states(self.num_nodes)

    def update(
        self,
        belief: np.ndarray,
        observation: np.ndarray,
        defense: np.ndarray,
    ) -> np.ndarray:
        """Return b_{k+1} after observing detected probe counts Y and action D."""
        belief = _as_distribution(belief, 2**self.num_nodes, "belief")
        observation = _as_count_vector(observation, self.num_nodes, "observation")
        defense = _as_binary_vector(defense, self.num_nodes, "defense")

        states = self.states
        posterior = np.zeros(len(states), dtype=np.float64)

        for state_index, state in enumerate(states):
            prior_mass = belief[state_index]
            if prior_mass <= 0.0:
                continue

            for probe_counts, policy_prob in self.attacker_model.joint_action_distribution(
                state, defense
            ):
                if policy_prob <= 0.0:
                    continue

                likelihood = observation_likelihood(
                    observation,
                    probe_counts,
                    self.probe_miss_probability,
                )
                if likelihood <= 0.0:
                    continue

                q = node_compromise_probabilities(state, probe_counts, defense, self.alpha)
                transition_probs = factorized_transition_probabilities(states, q)
                posterior += prior_mass * policy_prob * likelihood * transition_probs

        normalizer = posterior.sum()
        if normalizer <= self.epsilon:
            return np.full_like(posterior, 1.0 / len(posterior))
        return posterior / normalizer


def node_marginals(belief: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Per-node probability that the node is compromised under ``belief``."""
    belief = np.asarray(belief, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64)
    return belief @ states


def node_compromise_probabilities(
    state: np.ndarray,
    probe_counts: np.ndarray,
    defense: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Compute q_v(s, rho, D), the chance each node is compromised next step.

    - A reimaged node (defense == 1) is reset to clean: q = 0.
    - An undefended, already-compromised node stays compromised: q = 1.
    - An undefended clean node probed ``rho`` times is compromised with
      probability ``1 - exp(-alpha * rho)`` (0 when unprobed).
    """
    state = np.asarray(state, dtype=np.int8)
    probe_counts = np.asarray(probe_counts, dtype=np.float64)
    defense = np.asarray(defense, dtype=np.int8)
    alpha = np.asarray(alpha, dtype=np.float64)

    q = np.zeros_like(alpha, dtype=np.float64)
    not_defended = defense == 0
    q[not_defended & (state == 1)] = 1.0
    clean = not_defended & (state == 0)
    q[clean] = 1.0 - np.exp(-alpha[clean] * probe_counts[clean])
    return q


def factorized_transition_probabilities(
    next_states: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Compute K^{rho,D}(s' | s) for all candidate next states."""
    return np.prod(
        np.where(next_states == 1, q, 1.0 - q),
        axis=1,
        dtype=np.float64,
    )


def observation_likelihood(
    observation: np.ndarray,
    probe_counts: np.ndarray,
    probe_miss_probability: float,
) -> float:
    """Compute L(Y | rho): each probe is detected independently w.p. 1 - nu.

    With ``rho_v`` probes on node ``v``, the number detected is
    Binomial(rho_v, 1 - nu). There are no false positives, so observing more
    detections than probes has zero likelihood.
    """
    observation = np.asarray(observation, dtype=np.int64)
    probe_counts = np.asarray(probe_counts, dtype=np.int64)
    if np.any(observation > probe_counts) or np.any(observation < 0):
        return 0.0
    detect = 1.0 - probe_miss_probability
    miss = probe_miss_probability
    likelihood = 1.0
    for detected, probes in zip(observation.tolist(), probe_counts.tolist()):
        likelihood *= comb(probes, detected) * detect**detected * miss ** (probes - detected)
    return likelihood


def _as_binary_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.int8)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    if np.any((array != 0) & (array != 1)):
        raise ValueError(f"{name} must be binary.")
    return array


def _as_count_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative.")
    return array


def _as_distribution(value: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    if np.any(array < 0.0):
        raise ValueError(f"{name} cannot contain negative probabilities.")
    total = array.sum()
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass.")
    return array / total
