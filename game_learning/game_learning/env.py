"""Gymnasium environments for the binary graph cybersecurity game."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np

from .belief import BeliefUpdater, enumerate_binary_states, node_compromise_probabilities
from .policies import UniformAttackerPolicy


class DefenderPolicy(Protocol):
    """Fixed defender policy used while training the attacker."""

    def act(self, belief: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class BasicCyberGraphDefenseConfig:
    """Parameters for the binary graph cybersecurity game."""

    graph: nx.Graph
    beta: float | list[float] | np.ndarray = 0.1
    probe_miss_probability: float = 0.2
    attacker_cost: float = 0.05
    defender_cost: float = 0.1
    max_steps: int = 50
    max_attack_nodes: int = 1
    max_defend_nodes: int | None = 1
    initial_compromised_probability: float = 0.0


class BasicCyberGraphDefenseEnv(gym.Env):
    """Defender-side env: PPO observes defender belief and chooses reimages."""

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        config: BasicCyberGraphDefenseConfig,
        attacker_policy: UniformAttackerPolicy | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.graph = config.graph.copy()
        self.nodes = list(self.graph.nodes)
        self.num_nodes = len(self.nodes)
        _validate_config(config, self.num_nodes)

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
        del options
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._state = _sample_initial_state(
            self.num_nodes,
            self.config.initial_compromised_probability,
            self._rng,
        )
        self._belief = _belief_from_known_state(self.state_space, self._state)
        self._last_attack = np.zeros(self.num_nodes, dtype=np.int8)
        self._last_observation = np.zeros(self.num_nodes, dtype=np.int8)
        return self._observation(), self._info(defense=np.zeros(self.num_nodes, dtype=np.int8))

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        defense = _sanitize_binary_budget(action, self.num_nodes, self.config.max_defend_nodes)
        attack = self.attacker_policy.sample(self._state, defense, self._rng)
        next_state, detected = _transition(
            state=self._state,
            attack=attack,
            defense=defense,
            beta=self.beta,
            probe_miss_probability=self.config.probe_miss_probability,
            rng=self._rng,
        )

        self._belief = self.belief_updater.update(self._belief, detected, defense)
        self._state = next_state
        self._step_count += 1
        self._last_attack = attack
        self._last_observation = detected

        defender_reward, attacker_reward = _rewards(
            next_state,
            attack,
            defense,
            self.config.attacker_cost,
            self.config.defender_cost,
        )
        truncated = self._step_count >= self.config.max_steps
        info = self._info(defense=defense)
        info["attacker_reward"] = attacker_reward
        return self._observation(), defender_reward, False, truncated, info

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
        return _info(
            nodes=self.nodes,
            state=self._state,
            defense=defense,
            attack=self._last_attack,
            detected=self._last_observation,
            belief=self._belief,
            step_count=self._step_count,
        )


class BasicCyberGraphAttackEnv(gym.Env):
    """Attacker-side env: PPO observes the true state and chooses probes.

    The attacker has full state knowledge but does not observe the defender's
    current reimage action. Reimages are only visible indirectly through the
    next true state when previously controlled nodes become clean.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        config: BasicCyberGraphDefenseConfig,
        defender_policy: DefenderPolicy,
        belief_attacker_policy,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.defender_policy = defender_policy
        self.graph = config.graph.copy()
        self.nodes = list(self.graph.nodes)
        self.num_nodes = len(self.nodes)
        _validate_config(config, self.num_nodes)

        self.beta = _expand_beta(config.beta, self.num_nodes)
        self.belief_updater = BeliefUpdater(
            num_nodes=self.num_nodes,
            beta=self.beta,
            probe_miss_probability=config.probe_miss_probability,
            attacker_policy=belief_attacker_policy,
        )
        self.state_space = enumerate_binary_states(self.num_nodes)
        self.observation_space = spaces.MultiBinary(self.num_nodes)
        self.action_space = spaces.MultiBinary(self.num_nodes)
        self.render_mode = render_mode

        self._rng = np.random.default_rng()
        self._state = np.zeros(self.num_nodes, dtype=np.int8)
        self._belief = np.zeros(len(self.state_space), dtype=np.float64)
        self._step_count = 0
        self._last_defense = np.zeros(self.num_nodes, dtype=np.int8)
        self._last_attack = np.zeros(self.num_nodes, dtype=np.int8)
        self._last_observation = np.zeros(self.num_nodes, dtype=np.int8)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        del options
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._state = _sample_initial_state(
            self.num_nodes,
            self.config.initial_compromised_probability,
            self._rng,
        )
        self._belief = _belief_from_known_state(self.state_space, self._state)
        self._last_defense = np.zeros(self.num_nodes, dtype=np.int8)
        self._last_attack = np.zeros(self.num_nodes, dtype=np.int8)
        self._last_observation = np.zeros(self.num_nodes, dtype=np.int8)
        return self._observation(), self._info()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        attack = _sanitize_binary_budget(action, self.num_nodes, self.config.max_attack_nodes)
        defense = _sanitize_binary_budget(
            self.defender_policy.act(self._belief),
            self.num_nodes,
            self.config.max_defend_nodes,
        )
        next_state, detected = _transition(
            state=self._state,
            attack=attack,
            defense=defense,
            beta=self.beta,
            probe_miss_probability=self.config.probe_miss_probability,
            rng=self._rng,
        )

        self._belief = self.belief_updater.update(self._belief, detected, defense)
        self._state = next_state
        self._step_count += 1
        self._last_attack = attack
        self._last_defense = defense
        self._last_observation = detected

        defender_reward, attacker_reward = _rewards(
            next_state,
            attack,
            defense,
            self.config.attacker_cost,
            self.config.defender_cost,
        )
        truncated = self._step_count >= self.config.max_steps
        info = self._info()
        info["defender_reward"] = defender_reward
        return self._observation(), attacker_reward, False, truncated, info

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
        return self._state.copy()

    def _info(self) -> dict:
        return _info(
            nodes=self.nodes,
            state=self._state,
            defense=self._last_defense,
            attack=self._last_attack,
            detected=self._last_observation,
            belief=self._belief,
            step_count=self._step_count,
        )


def _validate_config(config: BasicCyberGraphDefenseConfig, num_nodes: int) -> None:
    if num_nodes < 1:
        raise ValueError("graph must contain at least one node.")
    if config.max_steps < 1:
        raise ValueError("max_steps must be at least 1.")
    if not 0.0 <= config.initial_compromised_probability <= 1.0:
        raise ValueError("initial_compromised_probability must be between 0 and 1.")


def _transition(
    state: np.ndarray,
    attack: np.ndarray,
    defense: np.ndarray,
    beta: np.ndarray,
    probe_miss_probability: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    q = node_compromise_probabilities(state, attack, defense, beta)
    next_state = (rng.random(len(state)) < q).astype(np.int8)
    detected = _sample_probe_observation(attack, probe_miss_probability, rng)
    return next_state, detected


def _sample_probe_observation(
    attack: np.ndarray,
    probe_miss_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    detection_probability = 1.0 - probe_miss_probability
    detected = np.zeros(len(attack), dtype=np.int8)
    attacked = np.flatnonzero(attack)
    rolls = rng.random(len(attacked))
    detected[attacked[rolls < detection_probability]] = 1
    return detected


def _rewards(
    next_state: np.ndarray,
    attack: np.ndarray,
    defense: np.ndarray,
    attacker_cost: float,
    defender_cost: float,
) -> tuple[float, float]:
    defender_reward = float(len(next_state) - int(next_state.sum()) - defender_cost * int(defense.sum()))
    attacker_reward = float(int(next_state.sum()) - attacker_cost * int(attack.sum()))
    return defender_reward, attacker_reward


def _sample_initial_state(
    num_nodes: int,
    initial_compromised_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    return (rng.random(num_nodes) < initial_compromised_probability).astype(np.int8)


def _belief_from_known_state(state_space: np.ndarray, state: np.ndarray) -> np.ndarray:
    matches = np.all(state_space == state, axis=1)
    belief = np.zeros(len(state_space), dtype=np.float64)
    belief[np.flatnonzero(matches)[0]] = 1.0
    return belief


def _info(
    nodes: list,
    state: np.ndarray,
    defense: np.ndarray,
    attack: np.ndarray,
    detected: np.ndarray,
    belief: np.ndarray,
    step_count: int,
) -> dict:
    return {
        "state": state.copy(),
        "state_nodes": {
            nodes[index]: int(value)
            for index, value in enumerate(state)
        },
        "defense": defense.copy(),
        "attack": attack.copy(),
        "detected_probes": detected.copy(),
        "belief": belief.copy(),
        "step": step_count,
    }


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
