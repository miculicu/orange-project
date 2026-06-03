"""Gymnasium environment for defender-side learning on an edge-free node game.

There are ``n`` independent nodes (no graph edges). Several attackers each
probe at most one node per step; the more attackers that hit the same clean,
undefended node in a step, the more likely it is compromised, following the
memoryless exponential model ``q = 1 - exp(-alpha * rho)`` from Datar &
Dujardin (CoDIT 2025). The defender acts on a Bayes-exact belief over the
hidden node states, updated from detected probe counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np

from .belief import (
    BeliefUpdater,
    enumerate_binary_states,
    node_compromise_probabilities,
)
from .policies import IDLE, AttackerEnsemble, SingleAttackerPolicy, UniformAttackerPolicy


@dataclass
class GameConfig:
    """Parameters for the edge-free multi-attacker node game."""

    num_nodes: int | None = None
    graph: nx.Graph | None = None  # optional, only used for node labels
    alpha: float | list[float] | np.ndarray = 0.5
    probe_miss_probability: float = 0.2
    num_attackers: int = 2
    attacker_cost: float = 0.05
    defender_cost: float = 0.1
    control_reward: float = 1.0
    max_steps: int = 50
    max_defend_nodes: int | None = 1
    attacker_idle_probability: float = 0.0
    initial_compromised_probability: float = 0.0

    def resolved_nodes(self) -> list:
        if self.graph is not None:
            return list(self.graph.nodes)
        if self.num_nodes is None:
            raise ValueError("provide either num_nodes or graph.")
        if self.num_nodes < 1:
            raise ValueError("num_nodes must be at least 1.")
        return list(range(self.num_nodes))


class CyberGraphDefenseEnv(gym.Env):
    """Gymnasium environment where the defender learns from exact beliefs.

    Observation:
        ``Box(0, 1, shape=(2^n,))``, the defender belief over hidden states.

    Action:
        ``MultiBinary(n)``, the subset of nodes to reimage.

    Reward:
        defender control reward minus reimage cost. Attacker rewards are
        reported in ``info["attacker_reward"]`` (team) and
        ``info["attacker_rewards"]`` (per attacker).
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        config: GameConfig,
        attackers: list[SingleAttackerPolicy] | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.nodes = config.resolved_nodes()
        self.num_nodes = len(self.nodes)
        if config.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")

        self.alpha = _expand_alpha(config.alpha, self.num_nodes)

        if attackers is None:
            attackers = [
                UniformAttackerPolicy(
                    num_nodes=self.num_nodes,
                    idle_probability=config.attacker_idle_probability,
                )
                for _ in range(config.num_attackers)
            ]
        self.ensemble = AttackerEnsemble(num_nodes=self.num_nodes, attackers=list(attackers))
        self.num_attackers = self.ensemble.num_attackers

        self.belief_updater = BeliefUpdater(
            num_nodes=self.num_nodes,
            alpha=self.alpha,
            probe_miss_probability=config.probe_miss_probability,
            attacker_model=self.ensemble,
        )
        self.state_space = enumerate_binary_states(self.num_nodes)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.state_space),),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiBinary(self.num_nodes)
        self.render_mode = render_mode

        self._rng = np.random.default_rng()
        self._state = np.zeros(self.num_nodes, dtype=np.int8)
        self._belief = np.zeros(len(self.state_space), dtype=np.float64)
        self._step_count = 0
        self._last_probe_counts = np.zeros(self.num_nodes, dtype=np.int64)
        self._last_observation = np.zeros(self.num_nodes, dtype=np.int64)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

        initial_probability = self.config.initial_compromised_probability
        if not 0.0 <= initial_probability <= 1.0:
            raise ValueError("initial_compromised_probability must be between 0 and 1.")
        self._state = (
            self._rng.random(self.num_nodes) < initial_probability
        ).astype(np.int8)
        self._belief = self._belief_from_known_state(self._state)
        self._last_probe_counts = np.zeros(self.num_nodes, dtype=np.int64)
        self._last_observation = np.zeros(self.num_nodes, dtype=np.int64)
        return self._observation(), self._info(defense=np.zeros(self.num_nodes, dtype=np.int8))

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        defense = self._sanitize_defense(action)

        choices = [a.sample(self._state, defense, self._rng) for a in self.ensemble.attackers]
        probe_counts = np.zeros(self.num_nodes, dtype=np.int64)
        for choice in choices:
            if choice != IDLE:
                probe_counts[choice] += 1

        q = node_compromise_probabilities(self._state, probe_counts, defense, self.alpha)
        next_state = (self._rng.random(self.num_nodes) < q).astype(np.int8)
        detected = self._sample_detected_counts(probe_counts)

        self._belief = self.belief_updater.update(self._belief, detected, defense)
        self._state = next_state
        self._step_count += 1
        self._last_probe_counts = probe_counts
        self._last_observation = detected

        compromised = int(next_state.sum())
        defender_reward = float(
            self.config.control_reward * (self.num_nodes - compromised)
            - self.config.defender_cost * int(defense.sum())
        )
        attacker_rewards = [
            float(
                self.config.control_reward * compromised
                - self.config.attacker_cost * (1 if choice != IDLE else 0)
            )
            for choice in choices
        ]

        terminated = False
        truncated = self._step_count >= self.config.max_steps
        info = self._info(defense=defense)
        info["attacker_rewards"] = attacker_rewards
        info["attacker_reward"] = float(sum(attacker_rewards))
        info["attacker_choices"] = list(choices)
        return self._observation(), defender_reward, terminated, truncated, info

    def render(self) -> str | None:
        compromised = [self.nodes[i] for i, v in enumerate(self._state) if v == 1]
        text = (
            f"step={self._step_count} compromised={compromised} "
            f"probe_counts={self._last_probe_counts.tolist()} "
            f"detected={self._last_observation.tolist()}"
        )
        if self.render_mode == "ansi":
            return text
        print(text)
        return None

    def _observation(self) -> np.ndarray:
        return self._belief.astype(np.float32)

    def _info(self, defense: np.ndarray) -> dict:
        return {
            "state": self._state.copy(),
            "state_nodes": {self.nodes[i]: int(v) for i, v in enumerate(self._state)},
            "defense": defense.copy(),
            "probe_counts": self._last_probe_counts.copy(),
            "detected_probes": self._last_observation.copy(),
            "belief": self._belief.copy(),
            "step": self._step_count,
        }

    def _sanitize_defense(self, action: np.ndarray) -> np.ndarray:
        defense = np.asarray(action, dtype=np.int8).reshape(self.num_nodes)
        defense = (defense > 0).astype(np.int8)
        if self.config.max_defend_nodes is None:
            return defense
        budget = max(0, self.config.max_defend_nodes)
        active = np.flatnonzero(defense)
        if len(active) > budget:
            defense[:] = 0
            defense[active[:budget]] = 1
        return defense

    def _sample_detected_counts(self, probe_counts: np.ndarray) -> np.ndarray:
        detect_probability = 1.0 - self.config.probe_miss_probability
        return self._rng.binomial(probe_counts, detect_probability).astype(np.int64)

    def _belief_from_known_state(self, state: np.ndarray) -> np.ndarray:
        matches = np.all(self.state_space == state, axis=1)
        belief = np.zeros(len(self.state_space), dtype=np.float64)
        belief[np.flatnonzero(matches)[0]] = 1.0
        return belief


def _expand_alpha(alpha: float | list[float] | np.ndarray, num_nodes: int) -> np.ndarray:
    if np.isscalar(alpha):
        values = np.full(num_nodes, float(alpha), dtype=np.float64)
    else:
        values = np.asarray(alpha, dtype=np.float64)
    if values.shape != (num_nodes,):
        raise ValueError("alpha must be a scalar or have one entry per node.")
    if np.any(values < 0.0):
        raise ValueError("alpha values must be non-negative.")
    return values
