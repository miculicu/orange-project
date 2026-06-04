"""Visualization helpers for the binary graph learning environment."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    attack = np.asarray(info.get("attack", np.zeros(node_count)), dtype=np.int8)
    defense = np.asarray(info.get("defense", np.zeros(node_count)), dtype=np.int8)
    detected = np.asarray(info.get("detected_probes", np.zeros(node_count)), dtype=np.int8)
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

    _add_status_panel(ax, _status_text(node_list, state, attack, defense, detected, belief))
    _add_visual_legend(ax)
    _pad_axes_for_annotations(pos, ax)
    display_title = _title_with_rewards(title, info)
    if display_title:
        ax.set_title(display_title)
    ax.axis("off")
    ax.figure.subplots_adjust(left=0.04, right=0.88, bottom=0.24, top=0.90)


def _complete_layout(graph: nx.Graph, pos: dict | None) -> dict:
    if pos is not None and set(pos) == set(graph.nodes):
        return pos
    if _is_cycle_graph(graph):
        return nx.circular_layout(graph)
    return nx.spring_layout(graph, seed=0)


def _is_cycle_graph(graph: nx.Graph) -> bool:
    if graph.number_of_nodes() < 3:
        return False
    return graph.number_of_edges() == graph.number_of_nodes() and all(
        degree == 2 for _, degree in graph.degree()
    )


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
        belief_text = _belief_summary_text(node_list, compromise_prob)
    return (
        f"compromised: {compromised}\n"
        f"attacked:     {attacked}\n"
        f"defended:     {defended}\n"
        f"detected:     {seen}"
        f"{belief_text}"
    )


def _title_with_rewards(title: str | None, info: dict) -> str | None:
    attacker_reward = info.get("attacker_reward")
    defender_reward = info.get("defender_reward")
    if attacker_reward is None and defender_reward is None:
        return title
    parts = []
    if attacker_reward is not None:
        parts.append(f"A={float(attacker_reward):.2f}")
    if defender_reward is not None:
        parts.append(f"D={float(defender_reward):.2f}")
    reward_text = " ".join(parts)
    if title and "A=" not in title and "D=" not in title:
        return f"{title} {reward_text}"
    return title or reward_text


def _pad_axes_for_annotations(pos: dict, ax) -> None:
    coordinates = np.asarray(list(pos.values()), dtype=np.float64)
    if coordinates.size == 0:
        return
    x_min, y_min = coordinates.min(axis=0)
    x_max, y_max = coordinates.max(axis=0)
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    ax.set_xlim(x_min - 0.08 * x_span, x_max + 0.12 * x_span)
    ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.10 * y_span)


def _add_status_panel(ax, text: str) -> None:
    ax.figure.text(
        0.04,
        0.035,
        text,
        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "#c5c9d1"},
        fontsize=9,
        family="monospace",
        verticalalignment="bottom",
        horizontalalignment="left",
    )


def _add_visual_legend(ax) -> None:
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=9,
               markerfacecolor="#2ca02c", markeredgecolor="#333333", label="clean"),
        Line2D([0], [0], marker="o", linestyle="", markersize=9,
               markerfacecolor="#d62728", markeredgecolor="#333333", label="compromised"),
        Line2D([0], [0], marker="o", linestyle="", markersize=9,
               markerfacecolor="white", markeredgecolor="#1f77b4", markeredgewidth=2.5, label="reimaged"),
        Line2D([0], [0], marker="o", linestyle="", markersize=9,
               markerfacecolor="white", markeredgecolor="#ff7f0e", markeredgewidth=2.5, label="attacked"),
        Line2D([0], [0], marker="o", linestyle="", markersize=9,
               markerfacecolor="white", markeredgecolor="#7b2cbf", markeredgewidth=2.5, label="detected"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        borderaxespad=0.0,
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        edgecolor="#c5c9d1",
        fontsize=8,
        borderpad=0.5,
        labelspacing=0.4,
        handlelength=1.0,
    )


def _belief_summary_text(node_list: list, compromise_prob: np.ndarray, limit: int = 5) -> str:
    if compromise_prob.size == 0:
        return ""
    count = min(limit, len(node_list), compromise_prob.size)
    top_indices = np.argsort(compromise_prob)[-count:][::-1]
    top = ", ".join(
        f"{node_list[index]}:{compromise_prob[index]:.2f}"
        for index in top_indices
    )
    mean_belief = float(np.mean(compromise_prob))
    return f"\nbelief top {count}: {top}\nbelief mean: {mean_belief:.2f}"


def _selected_nodes(node_list: list, mask: np.ndarray) -> list:
    return [node for index, node in enumerate(node_list) if index < len(mask) and mask[index]]


def _marginal_compromise_probabilities(belief: np.ndarray, num_nodes: int) -> np.ndarray:
    if belief.shape == (num_nodes,):
        return belief.astype(np.float64)
    states = np.array(
        [list(map(int, f"{index:0{num_nodes}b}")) for index in range(2**num_nodes)],
        dtype=np.float64,
    )
    return belief @ states
