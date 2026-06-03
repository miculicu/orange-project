"""Helpers for budgeted subset action spaces."""

from __future__ import annotations

from itertools import combinations

from gymnasium import spaces
import numpy as np


class BudgetedSubsetActionSpace:
    """Discrete action encoding for budgeted subsets plus optional full set.

    If `include_all_action` is true, the all-ones subset is included as one
    extra legal action when it is not already covered by the budget. This gives
    action sizes `{0, ..., budget, n}` rather than every intermediate size.
    """

    def __init__(
        self,
        num_nodes: int,
        budget: int | None,
        include_all_action: bool = False,
    ) -> None:
        if num_nodes < 1:
            raise ValueError("num_nodes must be at least 1.")
        self.num_nodes = num_nodes
        self.budget = num_nodes if budget is None else max(0, min(int(budget), num_nodes))
        self.include_all_action = include_all_action
        self.actions = self._build_actions()
        self.space = spaces.Discrete(len(self.actions))

    def decode(self, action) -> np.ndarray:
        index = int(np.asarray(action).item())
        if not 0 <= index < len(self.actions):
            raise ValueError(f"action index {index} is outside [0, {len(self.actions)}).")
        return self.actions[index].copy()

    def encode(self, action: np.ndarray) -> int:
        vector = np.asarray(action, dtype=np.int8).reshape(self.num_nodes)
        vector = (vector > 0).astype(np.int8)
        for index, candidate in enumerate(self.actions):
            if np.array_equal(vector, candidate):
                return index
        raise ValueError(f"Action is not in the configured action space: {vector.tolist()}")

    def _build_actions(self) -> list[np.ndarray]:
        actions: list[np.ndarray] = []
        for size in range(self.budget + 1):
            for nodes in combinations(range(self.num_nodes), size):
                action = np.zeros(self.num_nodes, dtype=np.int8)
                action[list(nodes)] = 1
                actions.append(action)

        if self.include_all_action and self.budget < self.num_nodes:
            actions.append(np.ones(self.num_nodes, dtype=np.int8))
        return actions
