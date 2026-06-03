"""Attacker and defender policies for the multi-attacker MTD game.

Each attacker probes at most one node per step (or stays idle), so an attacker
action is a node index in ``range(num_nodes)`` or ``-1`` for "idle". An
:class:`AttackerEnsemble` bundles several such attackers and exposes both a
sampler and the exact joint distribution over per-node probe counts, which the
defender's belief filter needs to stay Bayes-exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

IDLE = -1


class SingleAttackerPolicy(Protocol):
    """One attacker's policy pi(choice | s, D) over {-1 (idle)} u nodes."""

    def choice_distribution(
        self, state: np.ndarray, defense: np.ndarray
    ) -> dict[int, float]:
        ...

    def sample(
        self, state: np.ndarray, defense: np.ndarray, rng: np.random.Generator
    ) -> int:
        ...


def _allowed_nodes(state: np.ndarray, defense: np.ndarray, clean_only: bool) -> np.ndarray:
    state = np.asarray(state, dtype=np.int8)
    defense = np.asarray(defense, dtype=np.int8)
    allowed = defense == 0
    if clean_only:
        allowed &= state == 0
    return np.flatnonzero(allowed)


@dataclass(frozen=True)
class UniformAttackerPolicy:
    """Probe a uniformly random allowed node, or stay idle with fixed chance.

    A node is "allowed" if it is undefended (and, when ``clean_only``, not
    already compromised). With no allowed node the attacker is idle.
    """

    num_nodes: int
    idle_probability: float = 0.0
    clean_only: bool = True

    def __post_init__(self) -> None:
        if self.num_nodes < 1:
            raise ValueError("num_nodes must be at least 1.")
        if not 0.0 <= self.idle_probability <= 1.0:
            raise ValueError("idle_probability must be between 0 and 1.")

    def choice_distribution(
        self, state: np.ndarray, defense: np.ndarray
    ) -> dict[int, float]:
        allowed = _allowed_nodes(state, defense, self.clean_only)
        if len(allowed) == 0:
            return {IDLE: 1.0}
        active = (1.0 - self.idle_probability) / len(allowed)
        dist = {int(node): active for node in allowed}
        if self.idle_probability > 0.0:
            dist[IDLE] = self.idle_probability
        return dist

    def sample(
        self, state: np.ndarray, defense: np.ndarray, rng: np.random.Generator
    ) -> int:
        dist = self.choice_distribution(state, defense)
        choices = list(dist.keys())
        probs = np.array(list(dist.values()), dtype=np.float64)
        return int(rng.choice(choices, p=probs / probs.sum()))


@dataclass(frozen=True)
class FocusedAttackerPolicy:
    """Softmax-over-nodes attacker with learnable per-node preferences.

    ``logits`` are per-node scores; probing mass is the softmax over *allowed*
    nodes, scaled by ``1 - idle_probability``. Sharper logits make attackers
    pile onto the same node, which (via the exponential compromise model) is
    exactly the coordinated-attack regime. This closed form keeps the belief
    filter exact while the logits can be tuned by policy gradient.
    """

    num_nodes: int
    logits: np.ndarray
    idle_probability: float = 0.0
    clean_only: bool = True

    def __post_init__(self) -> None:
        logits = np.asarray(self.logits, dtype=np.float64)
        if logits.shape != (self.num_nodes,):
            raise ValueError("logits must have shape (num_nodes,).")
        if not 0.0 <= self.idle_probability <= 1.0:
            raise ValueError("idle_probability must be between 0 and 1.")

    def choice_distribution(
        self, state: np.ndarray, defense: np.ndarray
    ) -> dict[int, float]:
        allowed = _allowed_nodes(state, defense, self.clean_only)
        if len(allowed) == 0:
            return {IDLE: 1.0}
        logits = np.asarray(self.logits, dtype=np.float64)[allowed]
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        active = 1.0 - self.idle_probability
        dist = {int(node): active * w for node, w in zip(allowed.tolist(), weights)}
        if self.idle_probability > 0.0:
            dist[IDLE] = self.idle_probability
        return dist

    def sample(
        self, state: np.ndarray, defense: np.ndarray, rng: np.random.Generator
    ) -> int:
        dist = self.choice_distribution(state, defense)
        choices = list(dist.keys())
        probs = np.array(list(dist.values()), dtype=np.float64)
        return int(rng.choice(choices, p=probs / probs.sum()))


@dataclass
class AttackerEnsemble:
    """A fixed roster of single attackers acting independently each step."""

    num_nodes: int
    attackers: list[SingleAttackerPolicy] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.num_nodes < 1:
            raise ValueError("num_nodes must be at least 1.")

    @property
    def num_attackers(self) -> int:
        return len(self.attackers)

    def sample(
        self, state: np.ndarray, defense: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Return per-node probe counts for one step (shape (num_nodes,))."""
        counts = np.zeros(self.num_nodes, dtype=np.int64)
        for attacker in self.attackers:
            choice = attacker.sample(state, defense, rng)
            if choice != IDLE:
                counts[choice] += 1
        return counts

    def joint_action_distribution(
        self, state: np.ndarray, defense: np.ndarray
    ) -> list[tuple[np.ndarray, float]]:
        """Exact distribution over per-node probe-count vectors."""
        accumulator: dict[tuple[int, ...], float] = {(0,) * self.num_nodes: 1.0}
        for attacker in self.attackers:
            dist = attacker.choice_distribution(state, defense)
            updated: dict[tuple[int, ...], float] = {}
            for counts, prob in accumulator.items():
                for choice, choice_prob in dist.items():
                    if choice_prob <= 0.0:
                        continue
                    new_counts = list(counts)
                    if choice != IDLE:
                        new_counts[choice] += 1
                    key = tuple(new_counts)
                    updated[key] = updated.get(key, 0.0) + prob * choice_prob
            accumulator = updated
        return [
            (np.array(counts, dtype=np.int64), prob)
            for counts, prob in accumulator.items()
            if prob > 0.0
        ]


@dataclass(frozen=True)
class ThresholdDefenderPolicy:
    """Reimage nodes whose belief of being compromised exceeds a threshold.

    This is the threshold-in-belief structure proven optimal in Datar &
    Dujardin (CoDIT 2025), generalised to independent nodes: act on each node's
    marginal posterior, reimaging the most-suspect nodes up to ``max_defend_nodes``.
    """

    num_nodes: int
    threshold: float = 0.5
    max_defend_nodes: int | None = 1

    def sample(self, marginals: np.ndarray) -> np.ndarray:
        marginals = np.asarray(marginals, dtype=np.float64)
        defense = (marginals > self.threshold).astype(np.int8)
        if self.max_defend_nodes is not None:
            budget = max(0, self.max_defend_nodes)
            active = np.flatnonzero(defense)
            if len(active) > budget:
                # keep the most-suspect nodes within budget
                order = active[np.argsort(marginals[active])[::-1]]
                defense[:] = 0
                defense[order[:budget]] = 1
        return defense


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
