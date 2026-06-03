"""Train a PPO defender policy from an experiment config."""

from __future__ import annotations

import argparse
from pathlib import Path

from game_learning.experiment_config import build_environment, load_experiment_config
from stable_baselines3 import PPO


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
    built = build_environment(experiment)
    experiment.output_dir.mkdir(parents=True, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        built.env,
        verbose=experiment.training.verbose,
        seed=experiment.training.seed,
        device=experiment.training.device,
        learning_rate=experiment.training.learning_rate,
        n_steps=experiment.training.n_steps,
        batch_size=experiment.training.batch_size,
        gamma=experiment.training.gamma,
    )
    model.learn(total_timesteps=experiment.training.total_timesteps)
    model.save(experiment.model_path)
    print(f"Saved model to {experiment.model_zip_path}")


if __name__ == "__main__":
    main()
