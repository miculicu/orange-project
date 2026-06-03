"""Visualization helpers for cybergraph-game graphs."""

import textwrap

import matplotlib.pyplot as plt
import networkx as nx

NODE_SIZE = 650
SECURITY_RING_STEP = 130
LOG_FONT_SIZE = 9


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
        self.fig, (self.ax, self.log_ax) = plt.subplots(
            2,
            1,
            figsize=(9, 8),
            gridspec_kw={"height_ratios": [4.5, 1.7]},
        )
        self._log_lines: list[str] = []
        self._log_scroll_from_bottom = 0
        self._advance_requested = False
        self._quit_requested = False
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        plt.ion()

    def update(
        self,
        graph: nx.Graph,
        time_step: int,
        status_text: str | None = None,
    ) -> None:
        self.pos = _complete_layout(graph, self.pos)
        if status_text:
            self._append_log(status_text)
        title = f"Cybergraph simulation - time step {time_step}"
        _draw_graph_on_ax(
            graph,
            self.pos,
            self.ax,
            title=title,
        )
        self._draw_log_panel()
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
        elif event.key in {"up", "pageup"}:
            amount = 5 if event.key == "pageup" else 1
            self._scroll_log(amount)
        elif event.key in {"down", "pagedown"}:
            amount = 5 if event.key == "pagedown" else 1
            self._scroll_log(-amount)
        elif event.key == "home":
            self._log_scroll_from_bottom = max(
                0,
                len(self._log_lines) - self._visible_log_line_count(),
            )
            self._draw_log_panel()
            self.fig.canvas.draw_idle()
        elif event.key == "end":
            self._log_scroll_from_bottom = 0
            self._draw_log_panel()
            self.fig.canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.log_ax:
            return
        self._scroll_log(3 if event.button == "up" else -3)

    def _append_log(self, status_text: str) -> None:
        if self._log_lines:
            self._log_lines.append("")
        self._log_lines.extend(status_text.splitlines())

    def _scroll_log(self, amount: int) -> None:
        max_scroll = max(0, len(self._log_lines) - self._visible_log_line_count())
        self._log_scroll_from_bottom = min(
            max(self._log_scroll_from_bottom + amount, 0),
            max_scroll,
        )
        self._draw_log_panel()
        self.fig.canvas.draw_idle()

    def _draw_log_panel(self) -> None:
        self.log_ax.clear()
        self.log_ax.set_facecolor("#f7f8fa")
        for spine in self.log_ax.spines.values():
            spine.set_edgecolor("#c5c9d1")
        self.log_ax.set_xticks([])
        self.log_ax.set_yticks([])

        visible_count = self._visible_log_line_count()
        max_scroll = max(0, len(self._log_lines) - visible_count)
        self._log_scroll_from_bottom = min(self._log_scroll_from_bottom, max_scroll)

        bottom = len(self._log_lines) - self._log_scroll_from_bottom
        top = max(0, bottom - visible_count)
        visible_lines = self._wrap_log_lines(self._log_lines[top:bottom])
        visible_lines = visible_lines[-visible_count:]
        log_text = "\n".join(visible_lines)

        self.log_ax.text(
            0.015,
            0.96,
            log_text,
            transform=self.log_ax.transAxes,
            fontsize=LOG_FONT_SIZE,
            family="monospace",
            verticalalignment="top",
        )
        self.fig.tight_layout()

    def _visible_log_line_count(self) -> int:
        height_pixels = self.log_ax.get_window_extent().height
        line_height = LOG_FONT_SIZE * self.fig.dpi / 72 * 1.35
        return max(3, int(height_pixels / line_height) - 1)

    def _wrap_log_lines(self, lines: list[str]) -> list[str]:
        width_pixels = self.log_ax.get_window_extent().width
        char_width = LOG_FONT_SIZE * self.fig.dpi / 72 * 0.62
        width = max(30, int(width_pixels / char_width) - 2)
        wrapped_lines: list[str] = []
        for line in lines:
            wrapped_lines.extend(
                textwrap.wrap(
                    line,
                    width=width,
                    subsequent_indent="  ",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                or [""]
            )
        return wrapped_lines


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
