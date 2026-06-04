"""Gymnasium environments for the binary graph cybersecurity game."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np

from .action_spaces import BudgetedSubsetActionSpace
from .belief import BeliefUpdater, FactoredBeliefUpdater, enumerate_binary_states, node_compromise_probabilities
from .gnn_belief import LearnedGNNBeliefUpdater
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
    full_defense_cost_multiplier: float = 1.0
    max_steps: int = 50
    max_attack_nodes: int | None = 1
    max_defend_nodes: int | None = 1
    allow_full_attack: bool = False
    allow_full_defense: bool = False
    initial_compromised_probability: float = 0.0
    belief_type: str = "exact"
    factored_attack_probability: float | None = None
    edge_compromise_weight: float = 0.0
    gnn_belief_model_path: str | None = None
    gnn_belief_device: str = "cpu"
    defender_reimage_compromised_bonus: float = 0.0
    defender_high_belief_reimage_bonus: float = 0.0
    defender_missed_high_belief_penalty: float = 0.0
    defender_high_belief_threshold: float = 0.8
    attacker_new_compromise_bonus: float = 0.0
    attacker_owned_attack_penalty: float = 0.0


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
        self.adjacency = _adjacency_matrix(self.graph, self.nodes)
        self.attacker_policy = attacker_policy or UniformAttackerPolicy(
            num_nodes=self.num_nodes,
            max_attack_nodes=config.max_attack_nodes,
            allow_full_attack=config.allow_full_attack,
        )
        self.state_space = _state_space(config.belief_type, self.num_nodes)
        self.belief_updater = _make_belief_updater(
            config=config,
            num_nodes=self.num_nodes,
            beta=self.beta,
            attacker_policy=self.attacker_policy,
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(_belief_size(config.belief_type, self.num_nodes),),
            dtype=np.float32,
        )
        self.defense_action_codec = BudgetedSubsetActionSpace(
            self.num_nodes,
            config.max_defend_nodes,
            include_all_action=config.allow_full_defense,
        )
        self.action_space = self.defense_action_codec.space
        self.render_mode = render_mode

        self._rng = np.random.default_rng()
        self._state = np.zeros(self.num_nodes, dtype=np.int8)
        self._belief = np.zeros(_belief_size(config.belief_type, self.num_nodes), dtype=np.float64)
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
        self._belief = _initial_belief(self.config.belief_type, self.state_space, self._state)
        self._last_attack = np.zeros(self.num_nodes, dtype=np.int8)
        self._last_observation = np.zeros(self.num_nodes, dtype=np.int8)
        return self._observation(), self._info(defense=np.zeros(self.num_nodes, dtype=np.int8))

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        defense = self.defense_action_codec.decode(action)
        attack = self.attacker_policy.sample(self._state, defense, self._rng)
        previous_state = self._state.copy()
        previous_belief = self._belief.copy()
        next_state, detected = _transition(
            state=self._state,
            attack=attack,
            defense=defense,
            beta=self.beta,
            probe_miss_probability=self.config.probe_miss_probability,
            rng=self._rng,
            adjacency=self.adjacency,
            edge_compromise_weight=self.config.edge_compromise_weight,
        )

        self._belief = self.belief_updater.update(self._belief, detected, defense)
        self._state = next_state
        self._step_count += 1
        self._last_attack = attack
        self._last_observation = detected

        defender_reward, attacker_reward = _rewards(
            previous_state=previous_state,
            next_state=next_state,
            attack=attack,
            defense=defense,
            defender_belief=previous_belief,
            state_space=self.state_space,
            config=self.config,
        )
        truncated = self._step_count >= self.config.max_steps
        info = self._info(defense=defense)
        info["attacker_reward"] = attacker_reward
        info["defender_reward"] = defender_reward
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
        self.adjacency = _adjacency_matrix(self.graph, self.nodes)
        self.state_space = _state_space(config.belief_type, self.num_nodes)
        self.belief_updater = _make_belief_updater(
            config=config,
            num_nodes=self.num_nodes,
            beta=self.beta,
            attacker_policy=belief_attacker_policy,
        )
        self.observation_space = spaces.MultiBinary(self.num_nodes)
        self.attack_action_codec = BudgetedSubsetActionSpace(
            self.num_nodes,
            config.max_attack_nodes,
            include_all_action=config.allow_full_attack,
        )
        self.action_space = self.attack_action_codec.space
        self.render_mode = render_mode

        self._rng = np.random.default_rng()
        self._state = np.zeros(self.num_nodes, dtype=np.int8)
        self._belief = np.zeros(_belief_size(config.belief_type, self.num_nodes), dtype=np.float64)
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
        self._belief = _initial_belief(self.config.belief_type, self.state_space, self._state)
        self._last_defense = np.zeros(self.num_nodes, dtype=np.int8)
        self._last_attack = np.zeros(self.num_nodes, dtype=np.int8)
        self._last_observation = np.zeros(self.num_nodes, dtype=np.int8)
        return self._observation(), self._info()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        attack = self.attack_action_codec.decode(action)
        defense = _sanitize_binary_budget(
            self.defender_policy.act(self._belief),
            self.num_nodes,
            self.config.max_defend_nodes,
            allow_full_action=self.config.allow_full_defense,
        )
        previous_state = self._state.copy()
        previous_belief = self._belief.copy()
        next_state, detected = _transition(
            state=self._state,
            attack=attack,
            defense=defense,
            beta=self.beta,
            probe_miss_probability=self.config.probe_miss_probability,
            rng=self._rng,
            adjacency=self.adjacency,
            edge_compromise_weight=self.config.edge_compromise_weight,
        )

        self._belief = self.belief_updater.update(self._belief, detected, defense)
        self._state = next_state
        self._step_count += 1
        self._last_attack = attack
        self._last_defense = defense
        self._last_observation = detected

        defender_reward, attacker_reward = _rewards(
            previous_state=previous_state,
            next_state=next_state,
            attack=attack,
            defense=defense,
            defender_belief=previous_belief,
            state_space=self.state_space,
            config=self.config,
        )
        truncated = self._step_count >= self.config.max_steps
        info = self._info()
        info["attacker_reward"] = attacker_reward
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
    if config.belief_type not in {"exact", "factored", "learned_gnn"}:
        raise ValueError("belief_type must be 'exact', 'factored', or 'learned_gnn'.")
    if config.belief_type == "learned_gnn" and not config.gnn_belief_model_path:
        raise ValueError("gnn_belief_model_path is required when belief_type='learned_gnn'.")
    if config.factored_attack_probability is not None and not (
        0.0 <= config.factored_attack_probability <= 1.0
    ):
        raise ValueError("factored_attack_probability must be between 0 and 1.")
    if config.edge_compromise_weight < 0.0:
        raise ValueError("edge_compromise_weight must be nonnegative.")
    if config.full_defense_cost_multiplier < 0.0:
        raise ValueError("full_defense_cost_multiplier must be nonnegative.")
    shaping_values = {
        "defender_reimage_compromised_bonus": config.defender_reimage_compromised_bonus,
        "defender_high_belief_reimage_bonus": config.defender_high_belief_reimage_bonus,
        "defender_missed_high_belief_penalty": config.defender_missed_high_belief_penalty,
        "attacker_new_compromise_bonus": config.attacker_new_compromise_bonus,
        "attacker_owned_attack_penalty": config.attacker_owned_attack_penalty,
    }
    for name, value in shaping_values.items():
        if value < 0.0:
            raise ValueError(f"{name} must be nonnegative.")
    if not 0.0 <= config.defender_high_belief_threshold <= 1.0:
        raise ValueError("defender_high_belief_threshold must be between 0 and 1.")


def _transition(
    state: np.ndarray,
    attack: np.ndarray,
    defense: np.ndarray,
    beta: np.ndarray,
    probe_miss_probability: float,
    rng: np.random.Generator,
    adjacency: np.ndarray | None = None,
    edge_compromise_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    q = node_compromise_probabilities(
        state,
        attack,
        defense,
        beta,
        adjacency=adjacency,
        edge_compromise_weight=edge_compromise_weight,
    )
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
    *,
    previous_state: np.ndarray,
    next_state: np.ndarray,
    attack: np.ndarray,
    defense: np.ndarray,
    defender_belief: np.ndarray,
    state_space: np.ndarray,
    config: BasicCyberGraphDefenseConfig,
) -> tuple[float, float]:
    defense_count = int(defense.sum())
    if defense_count == len(next_state):
        defense_cost_paid = (
            config.defender_cost
            * defense_count
            * config.full_defense_cost_multiplier
        )
    else:
        defense_cost_paid = config.defender_cost * defense_count

    defender_reward = float(len(next_state) - int(next_state.sum()) - defense_cost_paid)
    attacker_reward = float(int(next_state.sum()) - config.attacker_cost * int(attack.sum()))

    reimaged_compromised = int(np.sum((previous_state == 1) & (defense == 1) & (next_state == 0)))
    newly_compromised = int(np.sum((previous_state == 0) & (next_state == 1)))
    owned_attacks = int(np.sum((previous_state == 1) & (attack == 1)))
    node_belief = _node_belief(defender_belief, state_space, len(next_state))
    high_belief_reimaged = float(np.sum(node_belief * defense))
    missed_high_belief = int(np.sum((node_belief >= config.defender_high_belief_threshold) & (defense == 0)))

    defender_reward += (
        config.defender_reimage_compromised_bonus * reimaged_compromised
        + config.defender_high_belief_reimage_bonus * high_belief_reimaged
        - config.defender_missed_high_belief_penalty * missed_high_belief
    )
    attacker_reward += (
        config.attacker_new_compromise_bonus * newly_compromised
        - config.attacker_owned_attack_penalty * owned_attacks
    )
    return defender_reward, attacker_reward


def _node_belief(belief: np.ndarray, state_space: np.ndarray, num_nodes: int) -> np.ndarray:
    if len(belief) == num_nodes:
        return belief.astype(np.float64, copy=False)
    return np.asarray(belief, dtype=np.float64) @ state_space


def _sample_initial_state(
    num_nodes: int,
    initial_compromised_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    return (rng.random(num_nodes) < initial_compromised_probability).astype(np.int8)


def _initial_belief(belief_type: str, state_space: np.ndarray, state: np.ndarray) -> np.ndarray:
    if belief_type in {"factored", "learned_gnn"}:
        return state.astype(np.float64)
    return _exact_belief_from_known_state(state_space, state)


def _exact_belief_from_known_state(state_space: np.ndarray, state: np.ndarray) -> np.ndarray:
    matches = np.all(state_space == state, axis=1)
    belief = np.zeros(len(state_space), dtype=np.float64)
    belief[np.flatnonzero(matches)[0]] = 1.0
    return belief


def _make_belief_updater(
    config: BasicCyberGraphDefenseConfig,
    num_nodes: int,
    beta: np.ndarray,
    attacker_policy,
):
    if config.belief_type == "learned_gnn":
        return LearnedGNNBeliefUpdater(
            model_path=str(config.gnn_belief_model_path),
            adjacency=_adjacency_matrix(config.graph, list(config.graph.nodes)),
            device=config.gnn_belief_device,
        )
    if config.belief_type == "factored":
        attack_probability = config.factored_attack_probability
        if attack_probability is None:
            if config.max_attack_nodes is None:
                attack_probability = 0.5
            else:
                attack_probability = min(1.0, max(0.0, config.max_attack_nodes / num_nodes))
        return FactoredBeliefUpdater(
            num_nodes=num_nodes,
            beta=beta,
            probe_miss_probability=config.probe_miss_probability,
            attack_probability=float(attack_probability),
            adjacency=_adjacency_matrix(config.graph, list(config.graph.nodes)),
            edge_compromise_weight=config.edge_compromise_weight,
        )
    return BeliefUpdater(
        num_nodes=num_nodes,
        beta=beta,
        probe_miss_probability=config.probe_miss_probability,
        attacker_policy=attacker_policy,
        adjacency=_adjacency_matrix(config.graph, list(config.graph.nodes)),
        edge_compromise_weight=config.edge_compromise_weight,
    )


def _state_space(belief_type: str, num_nodes: int) -> np.ndarray:
    if belief_type in {"factored", "learned_gnn"}:
        return np.empty((0, num_nodes), dtype=np.int8)
    return enumerate_binary_states(num_nodes)


def _belief_size(belief_type: str, num_nodes: int) -> int:
    if belief_type in {"factored", "learned_gnn"}:
        return num_nodes
    return 2**num_nodes


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
    allow_full_action: bool = False,
) -> np.ndarray:
    sanitized = np.asarray(action, dtype=np.int8).reshape(num_nodes)
    sanitized = (sanitized > 0).astype(np.int8)
    if budget is None:
        return sanitized
    if allow_full_action and int(sanitized.sum()) == num_nodes:
        return sanitized
    budget = max(0, budget)
    active = np.flatnonzero(sanitized)
    if len(active) > budget:
        sanitized[:] = 0
        sanitized[active[:budget]] = 1
    return sanitized


def _adjacency_matrix(graph: nx.Graph, nodes: list) -> np.ndarray:
    return nx.to_numpy_array(graph, nodelist=nodes, dtype=np.float64)


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
