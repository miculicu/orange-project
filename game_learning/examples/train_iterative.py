"""Alternating PPO training for defender and attacker policies."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

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
    )

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
        defender_model.learn(total_timesteps=experiment.iterative.defender_timesteps)
        defender_model.save(experiment.defender_model_path(iteration))
        print(f"Saved defender to {experiment.defender_model_zip_path(iteration)}")

        fixed_defender = SB3DefenderPolicy(
            model=defender_model,
            num_nodes=game_config.graph.number_of_nodes(),
            max_defend_nodes=game_config.max_defend_nodes,
        )

        print(f"\n=== Iteration {iteration:03d}: train attacker against fixed defender ===")
        attacker_env = BasicCyberGraphAttackEnv(
            game_config,
            defender_policy=fixed_defender,
            belief_attacker_policy=attacker_policy,
        )
        attacker_model = _make_ppo(experiment, attacker_env, seed=seed)
        attacker_model.learn(total_timesteps=experiment.iterative.attacker_timesteps)
        attacker_model.save(experiment.attacker_model_path(iteration))
        print(f"Saved attacker to {experiment.attacker_model_zip_path(iteration)}")

        attacker_policy = SB3AttackerPolicy(
            model=attacker_model,
            num_nodes=game_config.graph.number_of_nodes(),
            max_attack_nodes=game_config.max_attack_nodes,
        )


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


if __name__ == "__main__":
    main()
