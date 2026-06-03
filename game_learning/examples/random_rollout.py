"""Run a short random-defender rollout against multiple attackers."""

from __future__ import annotations

import numpy as np

from game_learning import CyberGraphDefenseEnv, GameConfig
from game_learning.policies import RandomDefenderPolicy


def main() -> None:
    num_nodes = 4
    env = CyberGraphDefenseEnv(
        GameConfig(
            num_nodes=num_nodes,
            alpha=0.5,
            probe_miss_probability=0.2,
            num_attackers=2,
            attacker_cost=0.05,
            defender_cost=0.1,
            max_steps=10,
            max_defend_nodes=1,
        ),
        render_mode="ansi",
    )
    observation, info = env.reset(seed=7)
    defender = RandomDefenderPolicy(num_nodes=num_nodes, max_defend_nodes=1)
    rng = np.random.default_rng(7)

    print("initial belief argmax:", int(observation.argmax()), info["state"].tolist())
    done = False
    while not done:
        action = defender.sample(rng)
        observation, reward, terminated, truncated, info = env.step(action)
        print(
            env.render(),
            "reward=", round(reward, 3),
            "attacker_reward=", round(info["attacker_reward"], 3),
            "belief_argmax=", int(observation.argmax()),
        )
        done = terminated or truncated


if __name__ == "__main__":
    main()
