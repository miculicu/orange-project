"""Reusable scenario runner for live cybergraph-game demos."""

from dataclasses import dataclass
import random

from .engine import AttackerPolicy, DefenderPolicy, advance_with_policies
from .graph_init import random_graph_init
from .model import (
    AddNode,
    AttackNode,
    GameState,
    NoAction,
    RestoreNode,
    RuleConfig,
    StepResult,
)
from .observations import initial_attacker_knowledge
from .visualization import GraphLiveView


@dataclass(frozen=True)
class GraphConfig:
    num_nodes: int
    num_edges: int
    num_entry_points: int
    graph_seed: int


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    graph: GraphConfig
    rules: RuleConfig
    random_seed: int
    num_time_steps: int
    attacker_policy: AttackerPolicy
    defender_policy: DefenderPolicy


def run_live_scenario(config: ScenarioConfig) -> None:
    """Run a live scenario controlled by Enter/q/Escape in the graph window."""
    graph = random_graph_init(
        num_nodes=config.graph.num_nodes,
        num_edges=config.graph.num_edges,
        num_entry_points=config.graph.num_entry_points,
        seed=config.graph.graph_seed,
        security_level_min=config.rules.security_level_min,
        security_level_max=config.rules.security_level_max,
    )
    state = GameState(
        graph=graph,
        attacker_knowledge=initial_attacker_knowledge(graph),
    )
    rng = random.Random(config.random_seed)
    view = GraphLiveView(graph)

    status_text = _initial_status(config, state)
    print(status_text)
    view.update(state.graph, time_step=state.time_step, status_text=status_text)

    for _ in range(config.num_time_steps):
        if not view.wait_for_next_step_or_quit():
            print("Simulation stopped by user.")
            break

        result = advance_with_policies(
            state,
            config.attacker_policy,
            config.defender_policy,
            config.rules,
            rng,
        )

        status_text = _format_time_step_status(result, state)
        print(status_text)
        view.update(state.graph, time_step=state.time_step, status_text=status_text)

    view.show_until_closed()


def _initial_status(config: ScenarioConfig, state: GameState) -> str:
    quit_key = "\u00fc"
    return (
        f"{config.name}\n"
        f"Known attacker nodes: {len(state.attacker_knowledge.known_nodes)}\n"
        f"Enter: next step | {quit_key}/q/Esc: quit"
    )


def _format_time_step_status(result: StepResult, state: GameState) -> str:
    return (
        f"{_format_action_result('Attacker', result.attacker_result)}\n"
        f"{_format_action_result('Defender', result.defender_result)}\n"
        f"Known attacker nodes: {len(state.attacker_knowledge.known_nodes)}"
    )


def _format_action_result(label: str, result) -> str:
    action_text = _format_action(result.action)
    detail = f"{label}: {action_text}, success={result.success}"
    if result.probability is not None:
        detail += f", p={result.probability:.3f}"
    if result.roll is not None:
        detail += f", roll={result.roll:.3f}"
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
    return str(action)
