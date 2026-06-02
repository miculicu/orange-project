"""Scenario: attacker always attacks and defender never defends on a big graph."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cybergame.policies import always_attack_policy, never_defend_policy
from cybergame.model import RuleConfig
from cybergame.scenario import GraphConfig, ScenarioConfig, run_live_scenario

SCENARIO = ScenarioConfig(
    name="Attacker always, defender never",
    graph=GraphConfig(
        num_nodes=35,
        num_edges=55,
        num_entry_points=3,
        graph_seed=11,
    ),
    rules=RuleConfig(
        attack_success_probability=0.7,
        security_level_min=1,
        security_level_max=3,
        default_security_level=1,
        added_node_security_level=3,
    ),
    random_seed=23,
    num_time_steps=40,
    attacker_policy=always_attack_policy,
    defender_policy=never_defend_policy,
)


if __name__ == "__main__":
    run_live_scenario(SCENARIO)
