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

    def __post_init__(self) -> None:
        self.action_codec = BudgetedSubsetActionSpace(
            self.num_nodes,
            self.max_attack_nodes,
            include_all_action=self.allow_full_attack,
        )

    def sample(
        self,
        state: np.ndarray,
        defense: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del defense, rng
        action_index, _ = self.model.predict(state.astype(np.float32), deterministic=self.deterministic)
        return self.action_codec.decode(action_index)

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

        observation = state.astype(np.float32)
        obs_tensor, _ = self.model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            categorical = distribution.distribution
            probs = categorical.probs.detach().cpu().numpy()
        return probs.reshape(-1)[: self.action_codec.space.n]
