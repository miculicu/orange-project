"""Optional Stable-Baselines3 PPO training for the belief-based defender.

The defender learns over the exact belief observation while a fixed roster of
attackers probes the nodes. (For a joint defender/attacker equilibrium that
also trains the attackers, see ``fictitious_play_demo.py``.)
"""

from __future__ import annotations

from game_learning import CyberGraphDefenseEnv, GameConfig


def main() -> None:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "Install stable-baselines3 first: pip install stable-baselines3"
        ) from exc

    env = CyberGraphDefenseEnv(
        GameConfig(
            num_nodes=4,
            alpha=0.5,
            probe_miss_probability=0.2,
            num_attackers=2,
            defender_cost=0.1,
            max_steps=50,
            max_defend_nodes=1,
        )
    )
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)
    model.save("ppo_cybergraph_defender")


if __name__ == "__main__":
    main()
