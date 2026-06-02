"""Visualization helpers for cybergraph-game graphs."""

import matplotlib.pyplot as plt
import networkx as nx

NODE_SIZE = 650
SECURITY_RING_STEP = 130


def draw_graph(
    graph: nx.Graph,
    pos: dict | None = None,
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
    ax=None,
    status_text: str | None = None,
) -> dict:
    """Draw a cybergraph and return the layout positions used."""
    pos = _complete_layout(graph, pos)

    created_axes = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    _draw_graph_on_ax(graph, pos, ax, title=title, status_text=status_text)

    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    elif created_axes:
        plt.close(fig)

    return pos


class GraphLiveView:
    """A small Matplotlib live view that redraws the graph in one window."""

    def __init__(
        self,
        graph: nx.Graph,
        pos: dict | None = None,
    ) -> None:
        self.pos = _complete_layout(graph, pos)
        self.fig, self.ax = plt.subplots()
        self._advance_requested = False
        self._quit_requested = False
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        plt.ion()

    def update(
        self,
        graph: nx.Graph,
        time_step: int,
        status_text: str | None = None,
    ) -> None:
        self.pos = _complete_layout(graph, self.pos)
        title = f"Cybergraph simulation - time step {time_step}"
        _draw_graph_on_ax(
            graph,
            self.pos,
            self.ax,
            title=title,
            status_text=status_text,
        )
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def wait_for_next_step_or_quit(self) -> bool:
        """Wait for Enter to advance; return False when the user quits."""
        self._advance_requested = False
        while (
            not self._advance_requested
            and not self._quit_requested
            and plt.fignum_exists(self.fig.number)
        ):
            plt.pause(0.05)
        return not self._quit_requested and plt.fignum_exists(self.fig.number)

    def show_until_closed(self) -> None:
        plt.ioff()
        plt.show()

    def _on_key_press(self, event) -> None:
        if event.key in {"enter", "return"}:
            self._advance_requested = True
        elif event.key == "q":
            self._quit_requested = True
            plt.close(self.fig)


def _complete_layout(graph: nx.Graph, pos: dict | None = None) -> dict:
    if pos is not None and set(pos) == set(graph.nodes):
        return pos
    if pos:
        current_pos = {node: pos[node] for node in graph if node in pos}
        fixed_nodes = list(current_pos)
        return nx.spring_layout(graph, pos=current_pos, fixed=fixed_nodes, seed=0)
    return nx.spring_layout(graph, seed=0)


def _draw_graph_on_ax(
    graph: nx.Graph,
    pos: dict,
    ax,
    title: str | None = None,
    status_text: str | None = None,
) -> None:
    ax.clear()

    entry_nodes = [
        node for node, data in graph.nodes(data=True) if data.get("is_entry_point")
    ]
    regular_nodes = [
        node for node, data in graph.nodes(data=True) if not data.get("is_entry_point")
    ]

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#8a8f98")

    _draw_nodes_with_security_rings(graph, pos, regular_nodes, node_shape="o", ax=ax)
    _draw_nodes_with_security_rings(graph, pos, entry_nodes, node_shape="s", ax=ax)

    nx.draw_networkx_labels(graph, pos, ax=ax)

    if title:
        ax.set_title(title)
    if status_text:
        ax.text(
            0.02,
            0.02,
            status_text,
            transform=ax.transAxes,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#c5c9d1"},
            fontsize=9,
            verticalalignment="bottom",
        )
    ax.axis("off")
    ax.figure.tight_layout()


def _node_color(control_state: object) -> str:
    if control_state == "defended":
        return "#2ca02c"
    if control_state == "captured":
        return "#d62728"
    return "#9aa0a6"


def _draw_nodes_with_security_rings(
    graph: nx.Graph,
    pos: dict,
    nodes: list,
    node_shape: str,
    ax,
) -> None:
    if not nodes:
        return

    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=nodes,
        node_color=[
            _node_color(graph.nodes[node].get("control_state")) for node in nodes
        ],
        node_shape=node_shape,
        node_size=NODE_SIZE,
        linewidths=0,
        ax=ax,
    )

    for node in nodes:
        security_level = _security_level_for_display(
            graph.nodes[node].get("security_level")
        )
        for layer in range(security_level, 0, -1):
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[node],
                node_color="none",
                edgecolors="#20242a",
                node_shape=node_shape,
                node_size=NODE_SIZE + layer * SECURITY_RING_STEP,
                linewidths=1.2,
                ax=ax,
            )


def _security_level_for_display(security_level: object) -> int:
    if isinstance(security_level, int) and security_level > 0:
        return security_level
    return 1
