"""Visualization helpers for the binary graph learning environment."""

from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
import numpy as np

NODE_SIZE = 800


def draw_game_state(
    graph: nx.Graph,
    info: dict,
    pos: dict | None = None,
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
    ax=None,
) -> dict:
    """Draw one environment state and return the graph layout positions."""
    pos = _complete_layout(graph, pos)
    created_axes = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    _draw_on_ax(graph, info, pos, ax, title)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    elif created_axes:
        plt.close(fig)
    return pos


class LearningGraphLiveView:
    """Small Matplotlib live view for stepping through learning-game rollouts."""

    def __init__(self, graph: nx.Graph, pos: dict | None = None) -> None:
        self.graph = graph
        self.pos = _complete_layout(graph, pos)
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        plt.ion()

    def update(self, info: dict, reward: float | None = None) -> None:
        title = f"Game learning rollout - step {info.get('step', 0)}"
        if reward is not None:
            title += f" - reward {reward:.2f}"
        _draw_on_ax(self.graph, info, self.pos, self.ax, title)
        self.fig.canvas.draw_idle()
        plt.pause(0.25)

    def show_until_closed(self) -> None:
        plt.ioff()
        plt.show()


def _draw_on_ax(
    graph: nx.Graph,
    info: dict,
    pos: dict,
    ax,
    title: str | None,
) -> None:
    ax.clear()
    node_count = graph.number_of_nodes()
    state = np.asarray(info.get("state", np.zeros(node_count)), dtype=np.int8)
    probe_counts = info.get("probe_counts", info.get("attack", np.zeros(node_count)))
    attack = (np.asarray(probe_counts, dtype=np.int64) > 0).astype(np.int8)
    defense = np.asarray(info.get("defense", np.zeros(node_count)), dtype=np.int8)
    detected = (
        np.asarray(info.get("detected_probes", np.zeros(node_count)), dtype=np.int64) > 0
    ).astype(np.int8)
    belief = np.asarray(info.get("belief", []), dtype=np.float64)
    node_list = list(graph.nodes)

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#9aa0a6", width=1.5)
    node_colors = ["#d62728" if state[index] else "#2ca02c" for index in range(node_count)]
    node_edges = [_node_edge_color(index, attack, defense, detected) for index in range(node_count)]
    node_widths = [_node_edge_width(index, attack, defense, detected) for index in range(node_count)]

    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=node_list,
        node_color=node_colors,
        edgecolors=node_edges,
        linewidths=node_widths,
        node_size=NODE_SIZE,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_color="white", font_weight="bold")

    ax.text(
        0.02,
        0.02,
        _status_text(node_list, state, attack, defense, detected, belief),
        transform=ax.transAxes,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#c5c9d1"},
        fontsize=9,
        family="monospace",
        verticalalignment="bottom",
    )
    ax.text(
        0.98,
        0.02,
        "green=clean red=compromised\nblue ring=defended orange=attacked purple=detected",
        transform=ax.transAxes,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#c5c9d1"},
        fontsize=8,
        horizontalalignment="right",
        verticalalignment="bottom",
    )
    if title:
        ax.set_title(title)
    ax.axis("off")
    ax.figure.tight_layout()


def _complete_layout(graph: nx.Graph, pos: dict | None) -> dict:
    if pos is not None and set(pos) == set(graph.nodes):
        return pos
    return nx.spring_layout(graph, seed=0)


def _node_edge_color(index: int, attack: np.ndarray, defense: np.ndarray, detected: np.ndarray) -> str:
    if detected[index]:
        return "#7b2cbf"
    if attack[index]:
        return "#ff7f0e"
    if defense[index]:
        return "#1f77b4"
    return "#333333"


def _node_edge_width(index: int, attack: np.ndarray, defense: np.ndarray, detected: np.ndarray) -> float:
    if detected[index] or attack[index] or defense[index]:
        return 4.0
    return 1.5


def _status_text(
    node_list: list,
    state: np.ndarray,
    attack: np.ndarray,
    defense: np.ndarray,
    detected: np.ndarray,
    belief: np.ndarray,
) -> str:
    compromised = _selected_nodes(node_list, state)
    attacked = _selected_nodes(node_list, attack)
    defended = _selected_nodes(node_list, defense)
    seen = _selected_nodes(node_list, detected)
    belief_text = ""
    if belief.size:
        compromise_prob = _marginal_compromise_probabilities(belief, len(node_list))
        belief_text = "\nbelief P(comp): " + ", ".join(
            f"{node}:{prob:.2f}" for node, prob in zip(node_list, compromise_prob)
        )
    return (
        f"compromised: {compromised}\n"
        f"attacked:     {attacked}\n"
        f"defended:     {defended}\n"
        f"detected:     {seen}"
        f"{belief_text}"
    )


def _selected_nodes(node_list: list, mask: np.ndarray) -> list:
    return [node for index, node in enumerate(node_list) if index < len(mask) and mask[index]]


def _marginal_compromise_probabilities(belief: np.ndarray, num_nodes: int) -> np.ndarray:
    states = np.array(
        [list(map(int, f"{index:0{num_nodes}b}")) for index in range(2**num_nodes)],
        dtype=np.float64,
    )
    return belief @ states