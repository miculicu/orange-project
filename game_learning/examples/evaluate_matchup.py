"""Evaluate and visualize a trained attacker/defender matchup."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from game_learning.action_spaces import BudgetedSubsetActionSpace
from game_learning.belief import enumerate_binary_states
from game_learning.env import BasicCyberGraphAttackEnv
from game_learning.experiment_config import (
    build_basic_cyber_graph_defense_config,
    load_experiment_config,
)
from game_learning.policy_adapters import SB3AttackerPolicy, SB3DefenderPolicy
from game_learning.visualization import draw_game_state


DEFAULT_CONFIG = Path("configs/path_graph_7_iterative.toml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--iteration", type=int, help="Use attacker/defender from this iterative-training iteration.")
    parser.add_argument("--defender", type=Path, help="Path to defender .zip model.")
    parser.add_argument("--attacker", type=Path, help="Path to attacker .zip model.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--name", type=str, default=None, help="Evaluation folder name.")
    parser.add_argument("--frame-every", type=int, default=1, help="Consider graph-frame saving every N environment steps.")
    parser.add_argument("--max-frames", type=int, default=None, help="Save the first N graph frames densely, then fall back to --frame-every.")
    parser.add_argument("--no-frames", action="store_true", help="Skip graph frame PNG generation.")
    args = parser.parse_args()

    experiment = load_experiment_config(args.config)
    graph, game_config = build_basic_cyber_graph_defense_config(experiment, max_steps=args.steps)
    defender_path, attacker_path = _resolve_model_paths(experiment, args)
    output_dir = _evaluation_dir(experiment, args.name, defender_path, attacker_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    defender_model = PPO.load(defender_path, device=experiment.training.device)
    attacker_model = PPO.load(attacker_path, device=experiment.training.device)
    num_nodes = graph.number_of_nodes()
    _validate_model_spaces(
        defender_model,
        attacker_model,
        num_nodes,
        game_config.max_defend_nodes,
        game_config.max_attack_nodes,
        game_config.allow_full_defense,
        game_config.allow_full_attack,
    )
    defender_policy = SB3DefenderPolicy(
        model=defender_model,
        num_nodes=num_nodes,
        max_defend_nodes=game_config.max_defend_nodes,
        allow_full_defense=game_config.allow_full_defense,
    )
    attacker_policy = SB3AttackerPolicy(
        model=attacker_model,
        num_nodes=num_nodes,
        max_attack_nodes=game_config.max_attack_nodes,
        allow_full_attack=game_config.allow_full_attack,
    )

    rows = []
    frame_dir = output_dir / "graph_frames"
    pos = None
    frames_saved = 0
    for episode in range(args.episodes):
        env = BasicCyberGraphAttackEnv(
            game_config,
            defender_policy=defender_policy,
            belief_attacker_policy=attacker_policy,
        )
        observation, info = env.reset(seed=args.seed + episode)
        if not args.no_frames:
            pos, frames_saved = _maybe_save_frame(
                graph,
                info,
                frame_dir,
                pos,
                episode,
                step=0,
                title=f"episode {episode} step 0",
                frame_every=args.frame_every,
                max_frames=args.max_frames,
                frames_saved=frames_saved,
            )

        done = False
        while not done:
            attack = attacker_policy.sample(
                state=observation,
                defense=np.zeros(num_nodes, dtype=np.int8),
                rng=np.random.default_rng(args.seed + episode),
            )
            observation, attacker_reward, terminated, truncated, info = env.step(attacker_policy.action_codec.encode(attack))
            row = _row_from_info(
                episode=episode,
                info=info,
                attacker_reward=float(attacker_reward),
                defender_reward=float(info["defender_reward"]),
                num_nodes=num_nodes,
            )
            rows.append(row)
            if not args.no_frames:
                pos, frames_saved = _maybe_save_frame(
                    graph,
                    info,
                    frame_dir,
                    pos,
                    episode,
                    step=int(info["step"]),
                    title=(
                        f"episode {episode} step {info['step']} "
                        f"A={attacker_reward:.2f} D={info['defender_reward']:.2f}"
                    ),
                    frame_every=args.frame_every,
                    max_frames=args.max_frames,
                    frames_saved=frames_saved,
                )
            done = terminated or truncated

    _write_csv(output_dir / "rollout.csv", rows)
    summary = _summary(rows, defender_path, attacker_path, args)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_all(output_dir / "plots", rows, num_nodes)
    print(f"Wrote evaluation to {output_dir}")


def _validate_model_spaces(
    defender_model,
    attacker_model,
    num_nodes: int,
    max_defend_nodes: int | None,
    max_attack_nodes: int | None,
    allow_full_defense: bool,
    allow_full_attack: bool,
) -> None:
    expected_defender_observation = (2**num_nodes,)
    expected_attacker_observation = (num_nodes,)
    expected_defender_actions = BudgetedSubsetActionSpace(
        num_nodes,
        max_defend_nodes,
        include_all_action=allow_full_defense,
    ).space.n
    expected_attacker_actions = BudgetedSubsetActionSpace(
        num_nodes,
        max_attack_nodes,
        include_all_action=allow_full_attack,
    ).space.n
    if defender_model.observation_space.shape != expected_defender_observation:
        raise SystemExit(
            "Defender model observation space "
            f"{defender_model.observation_space} does not match config with "
            f"{num_nodes} nodes: expected Box shape {expected_defender_observation}."
        )
    if getattr(defender_model.action_space, "n", None) != expected_defender_actions:
        raise SystemExit(
            "Defender model action space "
            f"{defender_model.action_space} does not match config: expected "
            f"Discrete({expected_defender_actions}). Retrain after the action-space fix."
        )
    if attacker_model.observation_space.shape != expected_attacker_observation:
        raise SystemExit(
            "Attacker model observation space "
            f"{attacker_model.observation_space} does not match config with {num_nodes} nodes."
        )
    if getattr(attacker_model.action_space, "n", None) != expected_attacker_actions:
        raise SystemExit(
            "Attacker model action space "
            f"{attacker_model.action_space} does not match config: expected "
            f"Discrete({expected_attacker_actions}). Retrain after the action-space fix."
        )


def _resolve_model_paths(experiment, args) -> tuple[Path, Path]:
    if args.iteration is not None:
        defender_path = experiment.defender_model_zip_path(args.iteration)
        attacker_path = experiment.attacker_model_zip_path(args.iteration)
    else:
        if args.defender is None or args.attacker is None:
            raise SystemExit("Provide --iteration or both --defender and --attacker.")
        defender_path = args.defender
        attacker_path = args.attacker
    if not defender_path.exists():
        raise SystemExit(f"Missing defender model: {defender_path}")
    if not attacker_path.exists():
        raise SystemExit(f"Missing attacker model: {attacker_path}")
    return defender_path, attacker_path


def _evaluation_dir(experiment, name: str | None, defender_path: Path, attacker_path: Path) -> Path:
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"matchup_{defender_path.stem}_vs_{attacker_path.stem}_{timestamp}"
    return experiment.output_dir / "evaluations" / name


def _row_from_info(
    episode: int,
    info: dict,
    attacker_reward: float,
    defender_reward: float,
    num_nodes: int,
) -> dict:
    state = np.asarray(info["state"], dtype=np.int8)
    attack = np.asarray(info["attack"], dtype=np.int8)
    defense = np.asarray(info["defense"], dtype=np.int8)
    detected = np.asarray(info["detected_probes"], dtype=np.int8)
    belief_marginals = _belief_marginals(np.asarray(info["belief"], dtype=np.float64), num_nodes)
    row = {
        "episode": episode,
        "step": int(info["step"]),
        "attacker_reward": attacker_reward,
        "defender_reward": defender_reward,
        "num_compromised": int(state.sum()),
        "num_attacked": int(attack.sum()),
        "num_defended": int(defense.sum()),
        "num_detected": int(detected.sum()),
        "state": _bits(state),
        "attack": _bits(attack),
        "defense": _bits(defense),
        "detected": _bits(detected),
    }
    for index, value in enumerate(belief_marginals):
        row[f"belief_p_node_{index}"] = float(value)
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict], defender_path: Path, attacker_path: Path, args) -> dict:
    if not rows:
        return {}
    return {
        "defender_model": str(defender_path),
        "attacker_model": str(attacker_path),
        "episodes": args.episodes,
        "steps": args.steps,
        "mean_attacker_reward": float(np.mean([r["attacker_reward"] for r in rows])),
        "mean_defender_reward": float(np.mean([r["defender_reward"] for r in rows])),
        "mean_num_compromised": float(np.mean([r["num_compromised"] for r in rows])),
        "mean_num_attacked": float(np.mean([r["num_attacked"] for r in rows])),
        "mean_num_defended": float(np.mean([r["num_defended"] for r in rows])),
        "mean_num_detected": float(np.mean([r["num_detected"] for r in rows])),
    }


def _plot_all(plot_dir: Path, rows: list[dict], num_nodes: int) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    x = np.arange(len(rows))
    _plot_lines(
        plot_dir / "rewards.png",
        x,
        {
            "attacker": [r["attacker_reward"] for r in rows],
            "defender": [r["defender_reward"] for r in rows],
        },
        ylabel="reward",
        title="Rewards over rollout",
    )
    _plot_lines(
        plot_dir / "counts.png",
        x,
        {
            "compromised": [r["num_compromised"] for r in rows],
            "attacked": [r["num_attacked"] for r in rows],
            "defended": [r["num_defended"] for r in rows],
            "detected": [r["num_detected"] for r in rows],
        },
        ylabel="count",
        title="Game counts over rollout",
    )
    _plot_lines(
        plot_dir / "belief_marginals.png",
        x,
        {
            f"node {index}": [r[f"belief_p_node_{index}"] for r in rows]
            for index in range(num_nodes)
        },
        ylabel="P(compromised)",
        title="Defender belief marginals",
    )
    _plot_action_frequencies(plot_dir / "action_frequencies.png", rows, num_nodes)


def _plot_lines(path: Path, x: np.ndarray, series: dict[str, list], ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, values in series.items():
        ax.plot(x, values, label=label, linewidth=1.5)
    ax.set_xlabel("rollout step index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_action_frequencies(path: Path, rows: list[dict], num_nodes: int) -> None:
    attacks = np.zeros(num_nodes, dtype=np.float64)
    defenses = np.zeros(num_nodes, dtype=np.float64)
    for row in rows:
        attacks += np.array([int(bit) for bit in row["attack"]], dtype=np.float64)
        defenses += np.array([int(bit) for bit in row["defense"]], dtype=np.float64)
    attacks /= len(rows)
    defenses /= len(rows)

    x = np.arange(num_nodes)
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, attacks, width=width, label="attacked")
    ax.bar(x + width / 2, defenses, width=width, label="defended")
    ax.set_xlabel("node")
    ax.set_ylabel("fraction of steps")
    ax.set_title("Action frequencies by node")
    ax.set_xticks(x)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _maybe_save_frame(
    graph,
    info: dict,
    frame_dir: Path,
    pos: dict | None,
    episode: int,
    step: int,
    title: str,
    frame_every: int,
    max_frames: int | None,
    frames_saved: int,
) -> tuple[dict | None, int]:
    in_dense_prefix = max_frames is not None and frames_saved < max_frames
    in_sparse_tail = frame_every > 0 and step % frame_every == 0
    if not in_dense_prefix and not in_sparse_tail:
        return pos, frames_saved
    save_path = frame_dir / f"episode_{episode:03d}_step_{step:04d}.png"
    pos = draw_game_state(graph, info, pos=pos, title=title, save_path=str(save_path), show=False)
    return pos, frames_saved + 1


def _belief_marginals(belief: np.ndarray, num_nodes: int) -> np.ndarray:
    states = enumerate_binary_states(num_nodes).astype(np.float64)
    return belief @ states


def _bits(values: np.ndarray) -> str:
    return "".join(str(int(value)) for value in values)


if __name__ == "__main__":
    main()
