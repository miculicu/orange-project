"""Visualize a rollout from a trained PPO defender policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from game_learning.experiment_config import build_environment, load_experiment_config
from game_learning.visualization import LearningGraphLiveView, draw_game_state


DEFAULT_CONFIG = Path("configs/path_graph_7.toml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the TOML config used for training.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of rollout steps to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Environment reset seed for visualization.",
    )
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "Install stable-baselines3 first: pip install stable-baselines3"
        ) from exc

    experiment = load_experiment_config(args.config)
    if not experiment.model_zip_path.exists():
        raise SystemExit(
            f"Missing {experiment.model_zip_path}. Train first with: "
            f".venv/bin/python examples/train_ppo.py --config {args.config}"
        )

    built = build_environment(experiment, max_steps=args.steps)
    if built.graph is None:
        raise SystemExit(f"Environment {experiment.env.env_id!r} does not provide graph visualization.")
    graph = built.graph
    model = PPO.load(experiment.model_zip_path, env=built.env, device=experiment.training.device)

    observation, info = built.env.reset(seed=args.seed)
    experiment.ppo_frame_dir.mkdir(parents=True, exist_ok=True)
    pos = draw_game_state(
        graph,
        info,
        title=f"{experiment.env.name} PPO rollout - initial",
        save_path=str(experiment.ppo_frame_dir / "step_00.png"),
        show=False,
    )

    view = LearningGraphLiveView(graph, pos=pos)
    view.update(info)
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = built.env.step(action)
        view.update(info, reward=reward)
        draw_game_state(
            graph,
            info,
            pos=pos,
            title=f"{experiment.env.name} PPO rollout - step {info['step']}",
            save_path=str(experiment.ppo_frame_dir / f"step_{info['step']:02d}.png"),
            show=False,
        )
        done = terminated or truncated
    view.show_until_closed()


if __name__ == "__main__":
    main()
