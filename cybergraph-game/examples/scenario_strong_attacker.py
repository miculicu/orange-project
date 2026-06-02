"""Scenario: strong attacker with a high capture probability."""

from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cybergame.attacker import always_attack_policy
from cybergame.defender import random_defender_policy_factory
from cybergame.game import advance_game, initialize_game_state
from cybergame.graph_init import random_graph_init
from cybergame.model import (
    AddNode,
    AdjustSecurityLevel,
    AttackNode,
    NoAction,
    RestoreNode,
    RuleConfig,
)
from cybergame.visualization import GraphLiveView

SCENARIO_NAME = "Strong attacker"

NUM_NODES = 2
NUM_EDGES = 1
NUM_ENTRY_POINTS = 1
GRAPH_SEED = 5

RANDOM_SEED = 19
NUM_TIME_STEPS = 25

ATTACK_SUCCESS_PROBABILITY = 0.95
ATTACK_SEEN_PROBABILITY = 0.35

SECURITY_LEVEL_MIN = 1
SECURITY_LEVEL_MAX = 3
ADDED_NODE_SECURITY_LEVEL = 3

DEFENDER_ACTION_PROBABILITY = 0.25
DEFENDER_ADD_NODE_PROBABILITY = 0.25
DEFENDER_ADJUST_SECURITY_PROBABILITY = 0.35
SECURITY_ADJUST_DELTA = 1
ADDED_NODE_EDGE_COUNT = 2

ATTACKER_POLICY = always_attack_policy
DEFENDER_POLICY = random_defender_policy_factory(
    action_probability=DEFENDER_ACTION_PROBABILITY,
    add_node_probability=DEFENDER_ADD_NODE_PROBABILITY,
    adjust_security_probability=DEFENDER_ADJUST_SECURITY_PROBABILITY,
    added_node_edge_count=ADDED_NODE_EDGE_COUNT,
    added_node_security_level=ADDED_NODE_SECURITY_LEVEL,
    security_adjust_delta=SECURITY_ADJUST_DELTA,
)


def main() -> None:
    rules = RuleConfig(
        attack_success_probability=ATTACK_SUCCESS_PROBABILITY,
        attack_seen_probability=ATTACK_SEEN_PROBABILITY,
        security_level_min=SECURITY_LEVEL_MIN,
        security_level_max=SECURITY_LEVEL_MAX,
        added_node_security_level=ADDED_NODE_SECURITY_LEVEL,
    )
    graph = random_graph_init(
        num_nodes=NUM_NODES,
        num_edges=NUM_EDGES,
        num_entry_points=NUM_ENTRY_POINTS,
        seed=GRAPH_SEED,
        security_level_min=SECURITY_LEVEL_MIN,
        security_level_max=SECURITY_LEVEL_MAX,
    )
    state = initialize_game_state(graph, rules)
    rng = random.Random(RANDOM_SEED)
    view = GraphLiveView(graph)

    status_text = _initial_status(state)
    print(status_text)
    view.update(state.graph, time_step=state.time_step, status_text=status_text)

    for _ in range(NUM_TIME_STEPS):
        if not view.wait_for_next_step_or_quit():
            print("Simulation stopped by user.")
            break

        result = advance_game(
            state,
            ATTACKER_POLICY,
            DEFENDER_POLICY,
            rules,
            rng,
        )
        status_text = _format_time_step_status(result, state)
        print(status_text)
        view.update(state.graph, time_step=state.time_step, status_text=status_text)

    view.show_until_closed()


def _initial_status(state) -> str:
    return (
        f"{SCENARIO_NAME}\n"
        f"Attacker known nodes: {len(state.attacker_knowledge.known_nodes)}\n"
        f"Defender known captured: {len(state.defender_knowledge.known_captured_nodes)}\n"
        "Enter: next step | q: quit"
    )


def _format_time_step_status(result, state) -> str:
    return (
        f"{_format_action_result('Attacker', result.attacker_result)}\n"
        f"{_format_action_result('Defender', result.defender_result)}\n"
        f"Attacker known nodes: {len(state.attacker_knowledge.known_nodes)}\n"
        f"Defender known captured: {len(state.defender_knowledge.known_captured_nodes)}"
    )


def _format_action_result(label: str, result) -> str:
    action_text = _format_action(result.action)
    detail = f"{label}: {action_text}, success={result.success}"
    if result.probability is not None:
        detail += f", p={result.probability:.3f}"
    if result.roll is not None:
        detail += f", roll={result.roll:.3f}"
    if label == "Attacker":
        detail += f", seen={result.seen_by_defender}"
    return f"{detail}, {result.reason}"


def _format_action(action) -> str:
    if isinstance(action, NoAction):
        return "no action"
    if isinstance(action, AttackNode):
        return f"attack {action.target}"
    if isinstance(action, RestoreNode):
        return f"restore {action.target}"
    if isinstance(action, AddNode):
        return f"add node linked to {action.neighbors}"
    if isinstance(action, AdjustSecurityLevel):
        return f"adjust security of {action.target} by {action.delta}"
    return str(action)


if __name__ == "__main__":
    main()
