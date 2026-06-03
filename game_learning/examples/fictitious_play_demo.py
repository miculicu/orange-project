"""Policy-gradient fictitious play: defender vs. two coordinating attackers.

Rotates best responses (defender threshold, then each attacker's REINFORCE
update) toward a Nash equilibrium, then prints where the two attackers learned
to focus their probes -- the coordination bonus rewards piling onto one node.
"""

from __future__ import annotations

import numpy as np

from game_learning import FictitiousPlayConfig, run_fictitious_play


def main() -> None:
    config = FictitiousPlayConfig(
        num_nodes=3,
        num_attackers=2,
        alpha=0.4,
        probe_miss_probability=0.2,
        attacker_cost=0.05,
        defender_cost=0.1,
        max_defend_nodes=1,
        horizon=25,
        gamma=0.95,
    )
    defender, attackers, history = run_fictitious_play(config, rounds=6, seed=0)

    print("\nfinal defender threshold:", round(defender.threshold, 3))
    for i, attacker in enumerate(attackers):
        probs = np.exp(attacker.logits - attacker.logits.max())
        probs /= probs.sum()
        print(f"attacker {i} node-preference:", np.round(probs, 3).tolist())


if __name__ == "__main__":
    main()
