"""Load experiment configs and construct environments/output paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

import networkx as nx

from .env import GameConfig


@dataclass(frozen=True)
class EnvSpec:
    name: str
    graph_type: str
    num_nodes: int
    alpha: float | list[float]
    probe_miss_probability: float
    attacker_cost: float
    defender_cost: float
    control_reward: float
    max_steps: int
    num_attackers: int
    attacker_idle_probability: float
    max_defend_nodes: int | None
    initial_compromised_probability: float


@dataclass(frozen=True)
class TrainingSpec:
    algorithm: str
    total_timesteps: int
    seed: int | None
    device: str
    verbose: int
    model_name: str
    output_root: Path
    learning_rate: float
    n_steps: int
    batch_size: int
    gamma: float


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    env: EnvSpec
    training: TrainingSpec

    @property
    def output_dir(self) -> Path:
        return self.training.output_root / self.env.name

    @property
    def model_path(self) -> Path:
        return self.output_dir / self.training.model_name

    @property
    def model_zip_path(self) -> Path:
        return self.output_dir / f"{self.training.model_name}.zip"

    @property
    def ppo_frame_dir(self) -> Path:
        return self.output_dir / "ppo_rollout_frames"


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    env_data = data.get("env")
    training_data = data.get("training")
    if not isinstance(env_data, dict):
        raise ValueError("Config must contain an [env] block.")
    if not isinstance(training_data, dict):
        raise ValueError("Config must contain a [training] block.")

    env = EnvSpec(
        name=str(_required(env_data, "name")),
        graph_type=str(_required(env_data, "graph_type")),
        num_nodes=int(_required(env_data, "num_nodes")),
        alpha=_required(env_data, "alpha"),
        probe_miss_probability=float(_required(env_data, "probe_miss_probability")),
        attacker_cost=float(env_data.get("attacker_cost", 0.05)),
        defender_cost=float(_required(env_data, "defender_cost")),
        control_reward=float(env_data.get("control_reward", 1.0)),
        max_steps=int(_required(env_data, "max_steps")),
        num_attackers=int(_required(env_data, "num_attackers")),
        attacker_idle_probability=float(env_data.get("attacker_idle_probability", 0.0)),
        max_defend_nodes=_optional_int(env_data.get("max_defend_nodes")),
        initial_compromised_probability=float(
            env_data.get("initial_compromised_probability", 0.0)
        ),
    )
    training = TrainingSpec(
        algorithm=str(training_data.get("algorithm", "ppo")),
        total_timesteps=int(_required(training_data, "total_timesteps")),
        seed=_optional_int(training_data.get("seed")),
        device=str(training_data.get("device", "auto")),
        verbose=int(training_data.get("verbose", 1)),
        model_name=str(training_data.get("model_name", "ppo_defender")),
        output_root=Path(str(training_data.get("output_root", "outputs"))),
        learning_rate=float(training_data.get("learning_rate", 0.0003)),
        n_steps=int(training_data.get("n_steps", 2048)),
        batch_size=int(training_data.get("batch_size", 64)),
        gamma=float(training_data.get("gamma", 0.99)),
    )
    if training.algorithm != "ppo":
        raise ValueError(f"Unsupported training.algorithm: {training.algorithm!r}")
    return ExperimentConfig(path=config_path, env=env, training=training)


def build_graph(env: EnvSpec) -> nx.Graph:
    # The game is edge-free; the graph only labels/positions nodes for plots.
    if env.graph_type in {"path", "empty"}:
        return nx.empty_graph(env.num_nodes)
    if env.graph_type == "cycle":
        return nx.cycle_graph(env.num_nodes)
    if env.graph_type == "complete":
        return nx.complete_graph(env.num_nodes)
    raise ValueError(f"Unsupported env.graph_type: {env.graph_type!r}")


def build_game_config(config: ExperimentConfig, *, max_steps: int | None = None) -> tuple[nx.Graph, GameConfig]:
    graph = build_graph(config.env)
    return graph, GameConfig(
        graph=graph,
        alpha=config.env.alpha,
        probe_miss_probability=config.env.probe_miss_probability,
        num_attackers=config.env.num_attackers,
        attacker_cost=config.env.attacker_cost,
        defender_cost=config.env.defender_cost,
        control_reward=config.env.control_reward,
        max_steps=config.env.max_steps if max_steps is None else max_steps,
        attacker_idle_probability=config.env.attacker_idle_probability,
        max_defend_nodes=config.env.max_defend_nodes,
        initial_compromised_probability=config.env.initial_compromised_probability,
    )


def _required(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing required config key: {key}")
    return data[key]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
