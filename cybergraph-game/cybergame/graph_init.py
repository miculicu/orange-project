"""Graph initialization utilities for the cybergraph-game project."""

import random

import networkx as nx

from .model import apply_initial_node_attributes


def random_graph_init(
    num_nodes: int,
    num_edges: int,
    num_entry_points: int,
    seed: int | None,
    security_level_min: int,
    security_level_max: int,
) -> nx.Graph:
    """Create a connected random graph with initial cybersecurity attributes."""
    _validate_random_graph_inputs(num_nodes, num_edges, num_entry_points)
    _validate_security_range(security_level_min, security_level_max)

    rng = random.Random(seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))

    nodes = list(graph.nodes)
    rng.shuffle(nodes)
    for index in range(1, num_nodes):
        parent = rng.choice(nodes[:index])
        graph.add_edge(nodes[index], parent)

    possible_extra_edges = [
        (u, v)
        for index, u in enumerate(range(num_nodes))
        for v in range(index + 1, num_nodes)
        if not graph.has_edge(u, v)
    ]
    rng.shuffle(possible_extra_edges)

    needed_extra_edges = num_edges - graph.number_of_edges()
    graph.add_edges_from(possible_extra_edges[:needed_extra_edges])

    entry_points = set(rng.sample(list(graph.nodes), num_entry_points))
    security_levels = {
        node: rng.randint(security_level_min, security_level_max)
        for node in graph.nodes
    }
    apply_initial_node_attributes(
        graph,
        entry_points,
        security_levels=security_levels,
    )

    return graph


def from_networkx_graph(
    graph: nx.Graph,
    entry_points: list | set | None = None,
    default_security_level: int | None = None,
) -> nx.Graph:
    """Copy a graph and add the initial cybersecurity node attributes."""
    entry_point_set = set(entry_points or [])
    invalid_entry_points = entry_point_set - set(graph.nodes)
    if invalid_entry_points:
        raise ValueError(
            f"Entry points are not nodes in the graph: {sorted(invalid_entry_points)!r}"
        )

    initialized_graph = graph.copy()
    if default_security_level is None:
        raise ValueError("default_security_level must be provided explicitly.")
    apply_initial_node_attributes(
        initialized_graph,
        entry_point_set,
        default_security_level=default_security_level,
    )
    return initialized_graph


def _validate_random_graph_inputs(
    num_nodes: int,
    num_edges: int,
    num_entry_points: int,
) -> None:
    if num_nodes < 1:
        raise ValueError("num_nodes must be at least 1.")
    if not 0 <= num_entry_points <= num_nodes:
        raise ValueError("num_entry_points must be between 0 and num_nodes.")

    min_edges = num_nodes - 1
    if num_edges < min_edges:
        raise ValueError(
            f"num_edges must be at least {min_edges} for a connected graph."
        )

    max_edges = num_nodes * (num_nodes - 1) // 2
    if num_edges > max_edges:
        raise ValueError(
            f"num_edges must be at most {max_edges} for a simple undirected graph."
        )


def _validate_security_range(security_level_min: int, security_level_max: int) -> None:
    if security_level_min < 1:
        raise ValueError("security_level_min must be at least 1.")
    if security_level_max < security_level_min:
        raise ValueError("security_level_max must be >= security_level_min.")
