"""Visualize a short random-policy rollout."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np

from game_learning import BasicCyberGraphDefenseEnv, BasicCyberGraphDefenseConfig
from game_learning.policies import RandomDefenderPolicy
from game_learning.visualization import LearningGraphLiveView, draw_game_state


def main() -> None:
    graph = nx.path_graph(4)
    env = BasicCyberGraphDefenseEnv(
        BasicCyberGraphDefenseConfig(
            graph=graph,
            beta=0.5,
            probe_miss_probability=0.2,
            defender_cost=0.1,
            max_steps=10,
            max_attack_nodes=1,
            max_defend_nodes=1,
        )
    )
    _, info = env.reset(seed=7)
    defender = RandomDefenderPolicy(num_nodes=graph.number_of_nodes(), max_defend_nodes=1)
    rng = np.random.default_rng(7)

    output_dir = Path("rollout_frames")
    output_dir.mkdir(exist_ok=True)
    pos = draw_game_state(
        graph,
        info,
        title="Game learning rollout - initial",
        save_path=str(output_dir / "step_00.png"),
        show=False,
    )

    view = LearningGraphLiveView(graph, pos=pos)
    view.update(info)
    done = False
    while not done:
        action = defender.sample(rng)
        _, reward, terminated, truncated, info = env.step(action)
        view.update(info, reward=reward)
        draw_game_state(
            graph,
            info,
            pos=pos,
            title=f"Game learning rollout - step {info['step']}",
            save_path=str(output_dir / f"step_{info['step']:02d}.png"),
            show=False,
        )
        done = terminated or truncated
    view.show_until_closed()


if __name__ == "__main__":
    main()
