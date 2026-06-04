"""Train a graph neural belief updater from simulated transition traces."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch import nn

from game_learning.env import _sample_initial_state, _transition
from game_learning.experiment_config import (
    build_basic_cyber_graph_defense_config,
    load_experiment_config,
)
from game_learning.gnn_belief import GraphBeliefBatch, GraphBeliefGRU
from game_learning.policies import UniformAttackerPolicy


DEFAULT_CONFIG = Path("configs/edge_factored_cycle_25.toml")
NODE_FEATURE_DIM = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--val-episodes", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--steps", type=int, default=None, help="Optional steps per generated episode; defaults to env.max_steps.")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--message-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.torch_threads))

    experiment = load_experiment_config(args.config)
    graph, config = build_basic_cyber_graph_defense_config(experiment)
    seed = args.seed if args.seed is not None else (experiment.training.seed or 0)
    output_dir = experiment.output_dir / "belief_models" / "gnn"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = _generate_dataset(config, graph, args.episodes, seed, max_steps=args.steps)
    val_data = _generate_dataset(config, graph, args.val_episodes, seed + 10_000, max_steps=args.steps)

    device = torch.device(args.device)
    adjacency = torch.as_tensor(
        nx.to_numpy_array(graph, dtype=np.float32), dtype=torch.float32, device=device
    )
    degree_feature = _degree_feature(adjacency)
    model = GraphBeliefGRU(
        node_feature_dim=NODE_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        message_dim=args.message_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCELoss()

    metrics = []
    for epoch in range(args.epochs):
        train_loss, train_acc = _run_epoch(
            model,
            train_data,
            adjacency,
            degree_feature,
            loss_fn,
            device,
            batch_size=args.batch_size,
            optimizer=optimizer,
        )
        val_loss, val_acc = _run_epoch(
            model,
            val_data,
            adjacency,
            degree_feature,
            loss_fn,
            device,
            batch_size=args.batch_size,
            optimizer=None,
        )
        row = {
            "epoch": epoch,
            "train_bce": train_loss,
            "train_accuracy": train_acc,
            "val_bce": val_loss,
            "val_accuracy": val_acc,
        }
        metrics.append(row)
        print(
            f"epoch={epoch:03d} train_bce={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_bce={val_loss:.4f} val_acc={val_acc:.3f}"
        )

    checkpoint_path = output_dir / "gnn_belief.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "node_feature_dim": NODE_FEATURE_DIM,
            "hidden_dim": args.hidden_dim,
            "message_dim": args.message_dim,
            "config": str(args.config),
            "env_name": experiment.env.name,
            "num_nodes": graph.number_of_nodes(),
        },
        checkpoint_path,
    )
    _write_metrics(output_dir / "metrics.csv", metrics)
    _plot_metrics(output_dir / "plots", metrics)
    print(f"Saved GNN belief model to {checkpoint_path}")


def _generate_dataset(config, graph, episodes: int, seed: int, max_steps: int | None = None) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    num_nodes = graph.number_of_nodes()
    steps = config.max_steps if max_steps is None else min(max_steps, config.max_steps)
    beta = np.full(num_nodes, float(config.beta), dtype=np.float64) if np.isscalar(config.beta) else np.asarray(config.beta, dtype=np.float64)
    adjacency = nx.to_numpy_array(graph, dtype=np.float64)
    attacker = UniformAttackerPolicy(
        num_nodes=num_nodes,
        max_attack_nodes=config.max_attack_nodes,
        allow_full_attack=config.allow_full_attack,
    )

    initial_states = np.zeros((episodes, num_nodes), dtype=np.float32)
    defenses = np.zeros((episodes, steps, num_nodes), dtype=np.float32)
    detections = np.zeros((episodes, steps, num_nodes), dtype=np.float32)
    targets = np.zeros((episodes, steps, num_nodes), dtype=np.float32)

    for episode in range(episodes):
        state = _sample_initial_state(num_nodes, config.initial_compromised_probability, rng)
        initial_states[episode] = state
        for step in range(steps):
            defense = _sample_defense(num_nodes, config.max_defend_nodes, config.allow_full_defense, rng)
            attack = attacker.sample(state, defense, rng)
            next_state, detected = _transition(
                state=state,
                attack=attack,
                defense=defense,
                beta=beta,
                probe_miss_probability=config.probe_miss_probability,
                rng=rng,
                adjacency=adjacency,
                edge_compromise_weight=config.edge_compromise_weight,
            )
            defenses[episode, step] = defense
            detections[episode, step] = detected
            targets[episode, step] = next_state
            state = next_state

    return {
        "initial_states": initial_states,
        "defenses": defenses,
        "detections": detections,
        "targets": targets,
    }


def _sample_defense(num_nodes: int, budget: int | None, allow_full: bool, rng: np.random.Generator) -> np.ndarray:
    action = np.zeros(num_nodes, dtype=np.int8)
    if allow_full and rng.random() < 0.05:
        action[:] = 1
        return action
    max_size = num_nodes if budget is None else min(budget, num_nodes)
    size = int(rng.integers(0, max_size + 1))
    if size:
        nodes = rng.choice(num_nodes, size=size, replace=False)
        action[nodes] = 1
    return action


def _run_epoch(
    model: GraphBeliefGRU,
    data: dict[str, np.ndarray],
    adjacency: torch.Tensor,
    degree_feature: torch.Tensor,
    loss_fn,
    device: torch.device,
    batch_size: int,
    optimizer=None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    episode_count = data["initial_states"].shape[0]
    indices = np.arange(episode_count)
    if training:
        np.random.shuffle(indices)

    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    total_batches = 0
    for start in range(0, episode_count, batch_size):
        batch_indices = indices[start:start + batch_size]
        initial = _tensor(data["initial_states"][batch_indices], device)
        defenses = _tensor(data["defenses"][batch_indices], device)
        detections = _tensor(data["detections"][batch_indices], device)
        targets = _tensor(data["targets"][batch_indices], device)

        belief = initial
        hidden = model.initial_hidden(belief)
        loss = torch.zeros((), dtype=torch.float32, device=device)
        predictions = []
        for step in range(targets.shape[1]):
            degree = degree_feature.unsqueeze(0).expand(belief.shape[0], -1)
            node_features = torch.stack([belief, defenses[:, step], detections[:, step], degree], dim=-1)
            belief, hidden = model(GraphBeliefBatch(node_features, adjacency, hidden))
            loss = loss + loss_fn(belief, targets[:, step])
            predictions.append(belief)
        loss = loss / targets.shape[1]

        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        with torch.no_grad():
            prediction_tensor = torch.stack(predictions, dim=1)
            correct = ((prediction_tensor >= 0.5) == (targets >= 0.5)).float().sum().item()
            total_correct += correct
            total_count += int(targets.numel())
            total_loss += float(loss.item())
            total_batches += 1
    return total_loss / max(total_batches, 1), total_correct / max(total_count, 1)


def _degree_feature(adjacency: torch.Tensor) -> torch.Tensor:
    degrees = adjacency.sum(dim=1)
    return degrees / torch.clamp(degrees.max(), min=1.0)


def _tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _write_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_metrics(plot_dir: Path, rows: list[dict]) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, [row["train_bce"] for row in rows], label="train")
    ax.plot(epochs, [row["val_bce"] for row in rows], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE")
    ax.set_title("GNN belief prediction loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "bce.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, [row["train_accuracy"] for row in rows], label="train")
    ax.plot(epochs, [row["val_accuracy"] for row in rows], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("node accuracy")
    ax.set_title("GNN belief prediction accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "accuracy.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
