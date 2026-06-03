"""Optional Stable-Baselines3 training example for the defender policy."""

from __future__ import annotations

import networkx as nx

from game_learning import CyberGraphDefenseEnv, GameConfig


def main() -> None:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "Install stable-baselines3 first: pip install stable-baselines3"
        ) from exc

    graph = nx.path_graph(4)
    env = CyberGraphDefenseEnv(
        GameConfig(
            graph=graph,
            beta=0.5,
            probe_miss_probability=0.2,
            defender_cost=0.1,
            max_steps=50,
            max_attack_nodes=1,
            max_defend_nodes=1,
        )
    )
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)
    model.save("ppo_cybergraph_defender")


if __name__ == "__main__":
    main()

