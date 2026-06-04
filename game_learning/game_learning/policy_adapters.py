"""Adapters that turn trained PPO models into fixed game policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .action_spaces import BudgetedSubsetActionSpace


@dataclass
class SB3DefenderPolicy:
    """Fixed defender policy backed by a Stable-Baselines3 model."""

    model: object
    num_nodes: int
    max_defend_nodes: int | None = None
    allow_full_defense: bool = False
    deterministic: bool = True

    def __post_init__(self) -> None:
        self.action_codec = BudgetedSubsetActionSpace(
            self.num_nodes,
            self.max_defend_nodes,
            include_all_action=self.allow_full_defense,
        )

    def act(self, belief: np.ndarray) -> np.ndarray:
        action_index, _ = self.model.predict(belief.astype(np.float32), deterministic=self.deterministic)
        return self.action_codec.decode(action_index)


@dataclass
class SB3AttackerPolicy:
    """Fixed attacker policy backed by a Stable-Baselines3 Discrete model.

    The attacker observes the true binary state only. It does not observe the
    defender action; the `defense` argument is accepted for the belief-update
    protocol and intentionally ignored.
    """

    model: object
    num_nodes: int
    max_attack_nodes: int | None = None
    allow_full_attack: bool = False
    deterministic: bool = True
    observation_type: str = "state"

    def __post_init__(self) -> None:
        self.action_codec = BudgetedSubsetActionSpace(
            self.num_nodes,
            self.max_attack_nodes,
            include_all_action=self.allow_full_attack,
        )
        self._last_state: np.ndarray | None = None
        self._previous_attack = np.zeros(self.num_nodes, dtype=np.int8)

    def reset(self) -> None:
        self._last_state = None
        self._previous_attack = np.zeros(self.num_nodes, dtype=np.int8)

    def sample(
        self,
        state: np.ndarray,
        defense: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del defense, rng
        observation = self._observation(state)
        action_index, _ = self.model.predict(observation.astype(np.float32), deterministic=self.deterministic)
        attack = self.action_codec.decode(action_index)
        current_state = self._extract_state(state)
        self._last_state = current_state.copy()
        self._previous_attack = attack.copy()
        return attack

    def probability(
        self,
        attack: np.ndarray,
        state: np.ndarray,
        defense: np.ndarray,
    ) -> float:
        del defense
        try:
            action_index = self.action_codec.encode(attack)
        except ValueError:
            return 0.0
        action_probs = self._action_probabilities(state)
        return float(action_probs[action_index])

    def _action_probabilities(self, state: np.ndarray) -> np.ndarray:
        import torch

        observation = self._observation(state).astype(np.float32)
        obs_tensor, _ = self.model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            categorical = distribution.distribution
            probs = categorical.probs.detach().cpu().numpy()
        return probs.reshape(-1)[: self.action_codec.space.n]


    def _extract_state(self, state_or_observation: np.ndarray) -> np.ndarray:
        values = np.asarray(state_or_observation, dtype=np.int8)
        if values.shape == (self.num_nodes,):
            return values
        if values.shape == (3 * self.num_nodes,):
            return values[: self.num_nodes]
        raise ValueError(
            f"Attacker observation must have shape ({self.num_nodes},) or "
            f"({3 * self.num_nodes},), got {values.shape}."
        )

    def _observation(self, state_or_observation: np.ndarray) -> np.ndarray:
        values = np.asarray(state_or_observation, dtype=np.int8)
        if values.shape == (3 * self.num_nodes,):
            return values.astype(np.float32)
        current_state = self._extract_state(values)
        if self.observation_type == "state":
            return current_state.astype(np.float32)
        if self.observation_type == "state_previous_attack_cleared":
            if self._last_state is None:
                cleared_owned = np.zeros(self.num_nodes, dtype=np.int8)
            else:
                cleared_owned = ((self._last_state == 1) & (current_state == 0)).astype(np.int8)
            return np.concatenate([current_state, self._previous_attack, cleared_owned]).astype(np.float32)
        raise ValueError(f"Unsupported attacker observation type: {self.observation_type!r}")
