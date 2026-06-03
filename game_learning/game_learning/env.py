"""Gymnasium environment for defender-side learning on a binary graph game."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np

from .belief import BeliefUpdater, enumerate_binary_states, node_compromise_probabilities
from .policies import UniformAttackerPolicy


@dataclass(frozen=True)
class GameConfig:
    """Parameters for the binary graph cybersecurity game."""

    graph: nx.Graph
    beta: float | list[float] | np.ndarray = 0.4
    probe_miss_probability: float = 0.2
    attacker_cost: float = 0.05
    defender_cost: float = 0.1
    max_steps: int = 50
    max_attack_nodes: int = 1
    max_defend_nodes: int | None = 1
    initial_compromised_probability: float = 0.0


class CyberGraphDefenseEnv(gym.Env):
    """A Gymnasium environment where the defender learns from exact beliefs.

    Observation:
        `Box(0, 1, shape=(2^n,))`, the defender belief over hidden states.

    Action:
        `MultiBinary(n)`, the subset of nodes to reimage.

    Reward:
        defender-controlled nodes after transition minus reimage cost. The
        attacker's reward is reported in `info["attacker_reward"]`.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        config: GameConfig,
        attacker_policy: UniformAttackerPolicy | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.graph = config.graph.copy()
        self.nodes = list(self.graph.nodes)
        self.num_nodes = len(self.nodes)
        if self.num_nodes < 1:
            raise ValueError("graph must contain at least one node.")
        if config.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")

        self.beta = _expand_beta(config.beta, self.num_nodes)
        self.attacker_policy = attacker_policy or UniformAttackerPolicy(
            num_nodes=self.num_nodes,
            max_attack_nodes=config.max_attack_nodes,
        )
        self.belief_updater = BeliefUpdater(
            num_nodes=self.num_nodes,
            beta=self.beta,
            probe_miss_probability=config.probe_miss_probability,
            attacker_policy=self.attacker_policy,
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
        self._last_attack = np.zeros(self.num_nodes, dtype=np.int8)
        self._last_observation = np.zeros(self.num_nodes, dtype=np.int8)

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
        self._last_attack = np.zeros(self.num_nodes, dtype=np.int8)
        self._last_observation = np.zeros(self.num_nodes, dtype=np.int8)
        return self._observation(), self._info(defense=np.zeros(self.num_nodes, dtype=np.int8))

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        defense = self._sanitize_defense(action)
        attack = self.attacker_policy.sample(self._state, defense, self._rng)
        q = node_compromise_probabilities(self._state, attack, defense, self.beta)
        next_state = (self._rng.random(self.num_nodes) < q).astype(np.int8)
        detected = self._sample_probe_observation(attack)

        self._belief = self.belief_updater.update(self._belief, detected, defense)
        self._state = next_state
        self._step_count += 1
        self._last_attack = attack
        self._last_observation = detected

        defender_reward = float(
            self.num_nodes - int(next_state.sum()) - self.config.defender_cost * int(defense.sum())
        )
        attacker_reward = float(
            int(next_state.sum()) - self.config.attacker_cost * int(attack.sum())
        )
        terminated = False
        truncated = self._step_count >= self.config.max_steps
        info = self._info(defense=defense)
        info["attacker_reward"] = attacker_reward
        return self._observation(), defender_reward, terminated, truncated, info

    def render(self) -> str | None:
        compromised = [
            self.nodes[index]
            for index, value in enumerate(self._state)
            if value == 1
        ]
        text = (
            f"step={self._step_count} compromised={compromised} "
            f"last_attack={self._last_attack.tolist()} "
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
            "state_nodes": {
                self.nodes[index]: int(value)
                for index, value in enumerate(self._state)
            },
            "defense": defense.copy(),
            "attack": self._last_attack.copy(),
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

    def _sample_probe_observation(self, attack: np.ndarray) -> np.ndarray:
        detection_probability = 1.0 - self.config.probe_miss_probability
        detected = np.zeros(self.num_nodes, dtype=np.int8)
        attacked = np.flatnonzero(attack)
        rolls = self._rng.random(len(attacked))
        detected[attacked[rolls < detection_probability]] = 1
        return detected

    def _belief_from_known_state(self, state: np.ndarray) -> np.ndarray:
        matches = np.all(self.state_space == state, axis=1)
        belief = np.zeros(len(self.state_space), dtype=np.float64)
        belief[np.flatnonzero(matches)[0]] = 1.0
        return belief


def _expand_beta(beta: float | list[float] | np.ndarray, num_nodes: int) -> np.ndarray:
    if np.isscalar(beta):
        values = np.full(num_nodes, float(beta), dtype=np.float64)
    else:
        values = np.asarray(beta, dtype=np.float64)
    if values.shape != (num_nodes,):
        raise ValueError("beta must be a scalar or have one entry per node.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("beta values must be between 0 and 1.")
    return values

