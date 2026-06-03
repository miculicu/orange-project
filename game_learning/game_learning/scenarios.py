"""Small scenario helpers kept for examples and backwards compatibility."""

from __future__ import annotations

import networkx as nx

from .env import GameConfig
from .experiment_config import build_game_config, load_experiment_config


DEFAULT_CONFIG_PATH = "configs/path_graph_7.toml"


def path_graph_training_scenario(max_steps: int = 50) -> tuple[nx.Graph, GameConfig]:
    """Return the default path-graph scenario used by PPO examples."""
    experiment = load_experiment_config(DEFAULT_CONFIG_PATH)
    return build_game_config(experiment, max_steps=max_steps)
