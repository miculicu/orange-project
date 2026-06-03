"""Scenario: static graph with random attacker and random reimaging defender."""

from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cybergame.attacker import random_attacker_policy
from cybergame.defender import random_reimage_policy_factory
from cybergame.game import advance_game, initialize_game_state
from cybergame.graph_init import random_graph_init
from cybergame.model import AttackNode, NoAction, RestoreNode, RuleConfig
from cybergame.visualization import GraphLiveView

SCENARIO_NAME = "Fully connected static 4-node graph with random attacker and random reimaging defender"

NUM_NODES = 4
NUM_EDGES = 6
NUM_ENTRY_POINTS = 4
GRAPH_SEED = 1

RANDOM_SEED = 7
NUM_TIME_STEPS = 10

ATTACK_SUCCESS_PROBABILITY = 0.5
ATTACK_SEEN_PROBABILITY = 0.5

SECURITY_LEVEL_MIN = 1
SECURITY_LEVEL_MAX = 1
ADDED_NODE_SECURITY_LEVEL = 1

DEFENDER_REIMAGE_PROBABILITY = 0.5

ATTACKER_POLICY = random_attacker_policy
DEFENDER_POLICY = random_reimage_policy_factory(
    action_probability=DEFENDER_REIMAGE_PROBABILITY,
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
        f"Attacker known nodes: {_format_node_list(state.attacker_knowledge.known_nodes)}\n"
        f"Defender known captures: {_format_node_list(state.defender_knowledge.known_captured_nodes)}\n"
        "Enter: next step | q: quit"
    )


def _format_time_step_status(result, state) -> str:
    return (
        f"Step {result.time_step}\n"
        f"{_format_action_result('Attacker', result.attacker_result)}\n"
        f"{_format_action_result('Defender', result.defender_result)}\n"
        f"Attacker known nodes: {_format_node_list(state.attacker_knowledge.known_nodes)}\n"
        f"Defender known captures: {_format_node_list(state.defender_knowledge.known_captured_nodes)}"
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
    return detail


def _format_action(action) -> str:
    if isinstance(action, NoAction):
        return "no action"
    if isinstance(action, AttackNode):
        return f"attack {action.target}"
    if isinstance(action, RestoreNode):
        return f"reimage {action.target}"
    return str(action)


def _format_node_list(nodes, limit: int = 10) -> str:
    sorted_nodes = sorted(nodes, key=str)
    displayed_nodes = [str(node) for node in sorted_nodes[:limit]]
    if len(sorted_nodes) > limit:
        displayed_nodes.append("...")
    return ",".join(displayed_nodes)


if __name__ == "__main__":
    main()
