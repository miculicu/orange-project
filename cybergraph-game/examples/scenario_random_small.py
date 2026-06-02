"""Scenario: random attacker and random defender on a small graph."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cybergame.config import (
    ADDED_NODE_EDGE_COUNT,
    ADDED_NODE_SECURITY_LEVEL,
    ATTACK_SUCCESS_PROBABILITY,
    DEFAULT_SECURITY_LEVEL,
    DEFENDER_ACTION_PROBABILITY,
    DEFENDER_ADD_NODE_PROBABILITY,
    GRAPH_SEED,
    NUM_EDGES,
    NUM_ENTRY_POINTS,
    NUM_NODES,
    NUM_TIME_STEPS,
    RANDOM_SEED,
    SECURITY_LEVEL_MAX,
    SECURITY_LEVEL_MIN,
)
from cybergame.policies import make_random_defender_policy, random_attacker_policy
from cybergame.model import RuleConfig
from cybergame.scenario import GraphConfig, ScenarioConfig, run_live_scenario

DEFENDER_POLICY = make_random_defender_policy(
    action_probability=DEFENDER_ACTION_PROBABILITY,
    add_node_probability=DEFENDER_ADD_NODE_PROBABILITY,
    added_node_edge_count=ADDED_NODE_EDGE_COUNT,
    added_node_security_level=ADDED_NODE_SECURITY_LEVEL,
)

SCENARIO = ScenarioConfig(
    name="Random small graph",
    graph=GraphConfig(
        num_nodes=NUM_NODES,
        num_edges=NUM_EDGES,
        num_entry_points=NUM_ENTRY_POINTS,
        graph_seed=GRAPH_SEED,
    ),
    rules=RuleConfig(
        attack_success_probability=ATTACK_SUCCESS_PROBABILITY,
        security_level_min=SECURITY_LEVEL_MIN,
        security_level_max=SECURITY_LEVEL_MAX,
        default_security_level=DEFAULT_SECURITY_LEVEL,
        added_node_security_level=ADDED_NODE_SECURITY_LEVEL,
    ),
    random_seed=RANDOM_SEED,
    num_time_steps=NUM_TIME_STEPS,
    attacker_policy=random_attacker_policy,
    defender_policy=DEFENDER_POLICY,
)


if __name__ == "__main__":
    run_live_scenario(SCENARIO)
