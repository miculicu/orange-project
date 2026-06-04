"""Alternating PPO training for defender and attacker policies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from game_learning.env import BasicCyberGraphAttackEnv, BasicCyberGraphDefenseEnv
from game_learning.experiment_config import (
    build_basic_cyber_graph_defense_config,
    load_experiment_config,
)
from game_learning.policies import UniformAttackerPolicy
from game_learning.policy_adapters import SB3AttackerPolicy, SB3DefenderPolicy


DEFAULT_CONFIG = Path("configs/path_graph_7.toml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to a TOML experiment config.",
    )
    args = parser.parse_args()

    experiment = load_experiment_config(args.config)
    _, game_config = build_basic_cyber_graph_defense_config(experiment)
    experiment.iterative_dir.mkdir(parents=True, exist_ok=True)

    attacker_policy = UniformAttackerPolicy(
        num_nodes=game_config.graph.number_of_nodes(),
        max_attack_nodes=game_config.max_attack_nodes,
        allow_full_attack=game_config.allow_full_attack,
    )
    summary_rows: list[dict[str, Any]] = []

    for iteration in range(experiment.iterative.iterations):
        iteration_dir = experiment.iteration_dir(iteration)
        iteration_dir.mkdir(parents=True, exist_ok=True)
        seed = None if experiment.training.seed is None else experiment.training.seed + iteration

        print(f"\n=== Iteration {iteration:03d}: train defender against fixed attacker ===")
        defender_env = BasicCyberGraphDefenseEnv(
            game_config,
            attacker_policy=attacker_policy,
        )
        defender_model = _make_ppo(experiment, defender_env, seed=seed)
        defender_log_dir = iteration_dir / "logs" / "defender"
        defender_model.set_logger(configure(str(defender_log_dir), ["stdout", "csv", "json"]))
        defender_model.learn(total_timesteps=experiment.iterative.defender_timesteps)
        defender_model.save(experiment.defender_model_path(iteration))
        print(f"Saved defender to {experiment.defender_model_zip_path(iteration)}")
        defender_progress_csv = defender_log_dir / "progress.csv"
        summary_rows.append(_summarize_progress(
            defender_progress_csv,
            role="defender",
            iteration=iteration,
            model_path=experiment.defender_model_zip_path(iteration),
        ))
        _plot_iteration_progress_curves(
            experiment.iterative_dir / "training_metrics" / "plots" / "per_iteration",
            defender_progress_csv,
            role="defender",
            iteration=iteration,
        )
        _write_training_artifacts(experiment.iterative_dir, summary_rows)

        fixed_defender = SB3DefenderPolicy(
            model=defender_model,
            num_nodes=game_config.graph.number_of_nodes(),
            max_defend_nodes=game_config.max_defend_nodes,
            allow_full_defense=game_config.allow_full_defense,
        )

        print(f"\n=== Iteration {iteration:03d}: train attacker against fixed defender ===")
        attacker_env = BasicCyberGraphAttackEnv(
            game_config,
            defender_policy=fixed_defender,
            belief_attacker_policy=attacker_policy,
        )
        attacker_model = _make_ppo(experiment, attacker_env, seed=seed)
        attacker_log_dir = iteration_dir / "logs" / "attacker"
        attacker_model.set_logger(configure(str(attacker_log_dir), ["stdout", "csv", "json"]))
        attacker_model.learn(total_timesteps=experiment.iterative.attacker_timesteps)
        attacker_model.save(experiment.attacker_model_path(iteration))
        print(f"Saved attacker to {experiment.attacker_model_zip_path(iteration)}")
        attacker_progress_csv = attacker_log_dir / "progress.csv"
        summary_rows.append(_summarize_progress(
            attacker_progress_csv,
            role="attacker",
            iteration=iteration,
            model_path=experiment.attacker_model_zip_path(iteration),
        ))
        _plot_iteration_progress_curves(
            experiment.iterative_dir / "training_metrics" / "plots" / "per_iteration",
            attacker_progress_csv,
            role="attacker",
            iteration=iteration,
        )
        _write_training_artifacts(experiment.iterative_dir, summary_rows)

        attacker_policy = SB3AttackerPolicy(
            model=attacker_model,
            num_nodes=game_config.graph.number_of_nodes(),
            max_attack_nodes=game_config.max_attack_nodes,
            allow_full_attack=game_config.allow_full_attack,
        )
        _maybe_evaluate_iteration(experiment, args.config, iteration)


def _maybe_evaluate_iteration(experiment, config_path: Path, iteration: int) -> None:
    eval_every = experiment.iterative.eval_every_iterations
    if eval_every <= 0 or (iteration + 1) % eval_every != 0:
        return
    name = f"auto_iteration_{iteration:03d}"
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate_matchup.py")),
        "--config",
        str(config_path),
        "--iteration",
        str(iteration),
        "--name",
        name,
    ]
    print(f"\n=== Evaluating iteration {iteration:03d} -> {name} ===")
    subprocess.run(command, check=True)


def _make_ppo(experiment, env, seed: int | None) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        verbose=experiment.training.verbose,
        seed=seed,
        device=experiment.training.device,
        learning_rate=experiment.training.learning_rate,
        n_steps=experiment.training.n_steps,
        batch_size=experiment.training.batch_size,
        gamma=experiment.training.gamma,
    )


SUMMARY_KEYS = [
    "time/total_timesteps",
    "time/fps",
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/loss",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/entropy_loss",
    "train/explained_variance",
    "train/approx_kl",
    "train/clip_fraction",
]


def _summarize_progress(
    progress_csv: Path,
    *,
    role: str,
    iteration: int,
    model_path: Path,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "iteration": iteration,
        "role": role,
        "model_path": str(model_path),
        "progress_csv": str(progress_csv),
        "progress_json": str(progress_csv.with_suffix(".json")),
    }
    last_values = _last_nonempty_values(progress_csv)
    for key in SUMMARY_KEYS:
        row[f"final_{_clean_metric_name(key)}"] = last_values.get(key, "")
    return row


def _last_nonempty_values(progress_csv: Path) -> dict[str, str]:
    if not progress_csv.exists():
        return {}
    values: dict[str, str] = {}
    with progress_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key, value in row.items():
                if value not in (None, ""):
                    values[key] = value
    return values


def _write_training_artifacts(iterative_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary_dir = iterative_dir / "training_metrics"
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(summary_dir / "summary.csv", rows)
    (summary_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _plot_training_summary(summary_dir / "plots", rows)


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_training_summary(plot_dir: Path, rows: list[dict[str, Any]]) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    _plot_metric(plot_dir / "episode_reward_mean.png", rows, "final_rollout__ep_rew_mean", "episode reward mean")
    _plot_metric(plot_dir / "value_loss.png", rows, "final_train__value_loss", "value loss")
    _plot_metric(plot_dir / "policy_gradient_loss.png", rows, "final_train__policy_gradient_loss", "policy gradient loss")
    _plot_metric(plot_dir / "explained_variance.png", rows, "final_train__explained_variance", "explained variance")
    _plot_metric(plot_dir / "fps.png", rows, "final_time__fps", "fps")
    _plot_combined_progress_curves(plot_dir / "within_iteration_episode_reward_mean.png", rows, "rollout/ep_rew_mean", "episode reward mean during each PPO fit")
    _plot_combined_progress_curves(plot_dir / "within_iteration_value_loss.png", rows, "train/value_loss", "value loss during each PPO fit")



def _plot_iteration_progress_curves(
    plot_dir: Path,
    progress_csv: Path,
    *,
    role: str,
    iteration: int,
) -> None:
    rows = _read_progress_rows(progress_csv)
    if not rows:
        return
    role_dir = plot_dir / role
    role_dir.mkdir(parents=True, exist_ok=True)
    stem = f"iteration_{iteration:03d}_{role}"
    _plot_progress_metric(
        role_dir / f"{stem}_episode_reward_mean.png",
        rows,
        "rollout/ep_rew_mean",
        f"iteration {iteration:03d} {role} reward mean",
    )
    _plot_progress_metric(
        role_dir / f"{stem}_value_loss.png",
        rows,
        "train/value_loss",
        f"iteration {iteration:03d} {role} value loss",
    )


def _plot_combined_progress_curves(
    path: Path,
    summary_rows: list[dict[str, Any]],
    metric: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for row in summary_rows:
        progress_csv = Path(str(row.get("progress_csv", "")))
        progress_rows = _read_progress_rows(progress_csv)
        xs, ys = _progress_xy(progress_rows, metric)
        if not xs:
            continue
        label = f"{row['role']} {int(row['iteration']):03d}"
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, label=label)
        plotted = True
    ax.set_xlabel("timesteps inside PPO fit")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(loc="best", fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_progress_metric(path: Path, rows: list[dict[str, str]], metric: str, title: str) -> None:
    xs, ys = _progress_xy(rows, metric)
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("timesteps inside PPO fit")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _progress_xy(rows: list[dict[str, str]], metric: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for index, row in enumerate(rows):
        y = _to_float(row.get(metric))
        if y is None:
            continue
        x = _to_float(row.get("time/total_timesteps"))
        xs.append(float(index) if x is None else x)
        ys.append(y)
    return xs, ys


def _read_progress_rows(progress_csv: Path) -> list[dict[str, str]]:
    if not progress_csv.exists():
        return []
    with progress_csv.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def _plot_metric(path: Path, rows: list[dict[str, Any]], metric: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    plotted = False
    for role in sorted({str(row["role"]) for row in rows}):
        role_rows = [row for row in rows if row["role"] == role and _to_float(row.get(metric)) is not None]
        if not role_rows:
            continue
        x = [int(row["iteration"]) for row in role_rows]
        y = [_to_float(row.get(metric)) for row in role_rows]
        ax.plot(x, y, marker="o", label=role)
        plotted = True
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_metric_name(name: str) -> str:
    return name.replace("/", "__")


if __name__ == "__main__":
    main()
