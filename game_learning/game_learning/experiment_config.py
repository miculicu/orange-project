"""Load experiment configs and construct configured environments/output paths."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any
import tomllib

import gymnasium as gym
import networkx as nx

from .env import BasicCyberGraphDefenseConfig, BasicCyberGraphDefenseEnv


BASIC_CYBER_GRAPH_DEFENSE_ENV_ID = "basic_cyber_graph_defense"


@dataclass(frozen=True)
class EnvSpec:
    """Raw environment block from a TOML config."""

    name: str
    env_id: str
    params: dict[str, Any]


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
    ent_coef: float


@dataclass(frozen=True)
class IterativeSpec:
    iterations: int
    defender_timesteps: int
    attacker_timesteps: int
    eval_every_iterations: int
    defender_ent_coef: float | None
    attacker_ent_coef: float | None


@dataclass(frozen=True)
class EvaluationSpec:
    episodes: int
    steps: int
    seed: int
    frame_every: int
    max_frames: int | None
    video: bool
    video_every: int
    max_video_frames: int | None
    video_fps: int


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    env: EnvSpec
    training: TrainingSpec
    iterative: IterativeSpec
    evaluation: EvaluationSpec

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

    @property
    def iterative_dir(self) -> Path:
        return self.output_dir / "iterative"

    def iteration_dir(self, iteration: int) -> Path:
        return self.iterative_dir / f"iteration_{iteration:03d}"

    def defender_model_path(self, iteration: int) -> Path:
        return self.iteration_dir(iteration) / "defender"

    def defender_model_zip_path(self, iteration: int) -> Path:
        return self.iteration_dir(iteration) / "defender.zip"

    def attacker_model_path(self, iteration: int) -> Path:
        return self.iteration_dir(iteration) / "attacker"

    def attacker_model_zip_path(self, iteration: int) -> Path:
        return self.iteration_dir(iteration) / "attacker.zip"


@dataclass(frozen=True)
class BuiltEnvironment:
    """Constructed environment plus optional graph for graph visualizations."""

    env: gym.Env
    graph: nx.Graph | None = None


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    env_data = data.get("env")
    training_data = data.get("training")
    iterative_data = data.get("iterative", {})
    evaluation_data = data.get("evaluation", {})
    if not isinstance(env_data, dict):
        raise ValueError("Config must contain an [env] block.")
    if not isinstance(training_data, dict):
        raise ValueError("Config must contain a [training] block.")
    if not isinstance(iterative_data, dict):
        raise ValueError("Config [iterative] block must be a table when present.")
    if not isinstance(evaluation_data, dict):
        raise ValueError("Config [evaluation] block must be a table when present.")

    env = _load_env_spec(env_data)
    training = _load_training_spec(training_data)
    iterative = _load_iterative_spec(iterative_data, training)
    evaluation = _load_evaluation_spec(evaluation_data, training)
    if training.algorithm != "ppo":
        raise ValueError(f"Unsupported training.algorithm: {training.algorithm!r}")
    return ExperimentConfig(
        path=config_path,
        env=env,
        training=training,
        iterative=iterative,
        evaluation=evaluation,
    )


def build_environment(
    config: ExperimentConfig,
    *,
    max_steps: int | None = None,
) -> BuiltEnvironment:
    """Build the Gym environment selected by config.env.env_id."""
    if config.env.env_id == BASIC_CYBER_GRAPH_DEFENSE_ENV_ID:
        graph, game_config = build_basic_cyber_graph_defense_config(config, max_steps=max_steps)
        return BuiltEnvironment(env=BasicCyberGraphDefenseEnv(game_config), graph=graph)
    raise ValueError(f"Unsupported env.env_id: {config.env.env_id!r}")


def build_basic_cyber_graph_defense_config(
    config: ExperimentConfig,
    *,
    max_steps: int | None = None,
) -> tuple[nx.Graph, BasicCyberGraphDefenseConfig]:
    """Build the current binary graph defender environment config."""
    _require_env(config, BASIC_CYBER_GRAPH_DEFENSE_ENV_ID)
    params = config.env.params
    graph = build_graph(params)
    return graph, BasicCyberGraphDefenseConfig(
        graph=graph,
        beta=_load_beta(params),
        probe_miss_probability=float(_required(params, "probe_miss_probability")),
        attacker_cost=float(params.get("attacker_cost", 0.05)),
        defender_cost=float(_required(params, "defender_cost")),
        full_defense_cost_multiplier=float(params.get("full_defense_cost_multiplier", 1.0)),
        max_steps=int(_required(params, "max_steps")) if max_steps is None else max_steps,
        max_attack_nodes=int(_required(params, "max_attack_nodes")),
        max_defend_nodes=_optional_int(params.get("max_defend_nodes")),
        allow_full_attack=bool(params.get("allow_full_attack", False)),
        allow_full_defense=bool(params.get("allow_full_defense", False)),
        initial_compromised_probability=float(
            params.get("initial_compromised_probability", 0.0)
        ),
        belief_type=str(params.get("belief_type", "exact")),
        factored_attack_probability=_optional_float(
            params.get("factored_attack_probability")
        ),
        edge_compromise_weight=float(params.get("edge_compromise_weight", 0.0)),
        gnn_belief_model_path=_optional_str(params.get("gnn_belief_model_path")),
        gnn_belief_device=str(params.get("gnn_belief_device", "cpu")),
        defender_reimage_compromised_bonus=float(params.get("defender_reimage_compromised_bonus", 0.0)),
        defender_high_belief_reimage_bonus=float(params.get("defender_high_belief_reimage_bonus", 0.0)),
        defender_missed_high_belief_penalty=float(params.get("defender_missed_high_belief_penalty", 0.0)),
        defender_high_belief_threshold=float(params.get("defender_high_belief_threshold", 0.8)),
        attacker_new_compromise_bonus=float(params.get("attacker_new_compromise_bonus", 0.0)),
        attacker_owned_attack_penalty=float(params.get("attacker_owned_attack_penalty", 0.0)),
        attacker_frontier_attack_bonus=float(params.get("attacker_frontier_attack_bonus", 0.0)),
        attacker_discovery_attack_bonus=float(params.get("attacker_discovery_attack_bonus", 0.0)),
        attacker_repeat_attack_penalty=float(params.get("attacker_repeat_attack_penalty", 0.0)),
        attacker_observation_type=str(params.get("attacker_observation_type", "state")),
    )


def build_game_config(
    config: ExperimentConfig,
    *,
    max_steps: int | None = None,
) -> tuple[nx.Graph, BasicCyberGraphDefenseConfig]:
    """Backward-compatible alias for the current graph-defense env builder."""
    return build_basic_cyber_graph_defense_config(config, max_steps=max_steps)


def build_graph(params: dict[str, Any]) -> nx.Graph:
    graph_type = str(_required(params, "graph_type"))
    num_nodes = int(_required(params, "num_nodes"))
    if graph_type == "path":
        return nx.path_graph(num_nodes)
    if graph_type == "cycle":
        return nx.cycle_graph(num_nodes)
    if graph_type == "complete":
        return nx.complete_graph(num_nodes)
    raise ValueError(f"Unsupported env.graph_type: {graph_type!r}")


def _load_env_spec(env_data: dict[str, Any]) -> EnvSpec:
    name = str(_required(env_data, "name"))
    env_id = str(env_data.get("env_id", BASIC_CYBER_GRAPH_DEFENSE_ENV_ID))
    params = dict(env_data)
    params.pop("name", None)
    params.pop("env_id", None)
    return EnvSpec(name=name, env_id=env_id, params=params)


def _load_training_spec(training_data: dict[str, Any]) -> TrainingSpec:
    return TrainingSpec(
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
        ent_coef=float(training_data.get("ent_coef", 0.0)),
    )


def _load_iterative_spec(
    iterative_data: dict[str, Any],
    training: TrainingSpec,
) -> IterativeSpec:
    return IterativeSpec(
        iterations=int(iterative_data.get("iterations", 3)),
        defender_timesteps=int(iterative_data.get("defender_timesteps", training.total_timesteps)),
        attacker_timesteps=int(iterative_data.get("attacker_timesteps", training.total_timesteps)),
        eval_every_iterations=int(iterative_data.get("eval_every_iterations", 0)),
        defender_ent_coef=_optional_float(iterative_data.get("defender_ent_coef")),
        attacker_ent_coef=_optional_float(iterative_data.get("attacker_ent_coef")),
    )


def _load_evaluation_spec(
    evaluation_data: dict[str, Any],
    training: TrainingSpec,
) -> EvaluationSpec:
    return EvaluationSpec(
        episodes=int(evaluation_data.get("episodes", 3)),
        steps=int(evaluation_data.get("steps", 100)),
        seed=int(evaluation_data.get("seed", training.seed if training.seed is not None else 7)),
        frame_every=int(evaluation_data.get("frame_every", 1)),
        max_frames=_optional_int(evaluation_data.get("max_frames")),
        video=bool(evaluation_data.get("video", False)),
        video_every=int(evaluation_data.get("video_every", 1)),
        max_video_frames=_optional_int(evaluation_data.get("max_video_frames")),
        video_fps=int(evaluation_data.get("video_fps", 8)),
    )


def _load_beta(params: dict[str, Any]) -> Any:
    has_beta = "beta" in params
    has_alpha = "alpha" in params
    if has_beta and has_alpha:
        raise ValueError("Specify only one of env.beta or env.alpha, not both.")
    if has_beta:
        return params["beta"]
    if has_alpha:
        return _alpha_to_beta(params["alpha"])
    raise ValueError("Missing required config key: beta or alpha")


def _alpha_to_beta(alpha: Any) -> Any:
    if isinstance(alpha, list):
        return [_alpha_to_beta(value) for value in alpha]
    alpha_value = float(alpha)
    if alpha_value < 0.0:
        raise ValueError("alpha entries must be nonnegative when beta = 1 - exp(-alpha).")
    return 1.0 - math.exp(-alpha_value)


def _require_env(config: ExperimentConfig, env_id: str) -> None:
    if config.env.env_id != env_id:
        raise ValueError(
            f"Config env_id {config.env.env_id!r} cannot be built as {env_id!r}."
        )


def _required(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing required config key: {key}")
    return data[key]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
