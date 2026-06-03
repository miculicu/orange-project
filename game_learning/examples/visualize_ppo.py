"""Visualize a rollout from a trained PPO defender policy."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from game_learning import CyberGraphDefenseEnv, GameConfig
from game_learning.visualization import LearningGraphLiveView, draw_game_state


MODEL_PATH = Path("ppo_cybergraph_defender.zip")


def main() -> None:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "Install stable-baselines3 first: pip install stable-baselines3"
        ) from exc

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Missing {MODEL_PATH}. Train a model first with: "
            ".venv/bin/python examples/train_ppo.py"
        )

    graph = nx.empty_graph(4)  # edge-free: nodes are independent
    env = CyberGraphDefenseEnv(
        GameConfig(
            graph=graph,
            alpha=0.5,
            probe_miss_probability=0.2,
            num_attackers=2,
            defender_cost=0.1,
            max_steps=10,
            max_defend_nodes=1,
        )
    )
    model = PPO.load(MODEL_PATH, env=env, device="cpu")

    observation, info = env.reset(seed=7)
    output_dir = Path("ppo_rollout_frames")
    output_dir.mkdir(exist_ok=True)
    pos = draw_game_state(
        graph,
        info,
        title="PPO defender rollout - initial",
        save_path=str(output_dir / "step_00.png"),
        show=False,
    )

    view = LearningGraphLiveView(graph, pos=pos)
    view.update(info)
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        view.update(info, reward=reward)
        draw_game_state(
            graph,
            info,
            pos=pos,
            title=f"PPO defender rollout - step {info['step']}",
            save_path=str(output_dir / f"step_{info['step']:02d}.png"),
            show=False,
        )
        done = terminated or truncated
    view.show_until_closed()


if __name__ == "__main__":
    main()
