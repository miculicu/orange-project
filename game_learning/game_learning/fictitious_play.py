"""Policy-gradient fictitious play for the multi-attacker MTD game.

This realises Algorithm 1 of Datar & Dujardin (CoDIT 2025), generalised to
several attackers: hold every player but one fixed, improve that player's
best response, then rotate. The defender is a threshold policy over its belief
marginals (the structure the paper proves optimal); each attacker is a
closed-form softmax-over-nodes policy whose logits are improved by REINFORCE.

Keeping the attacker policies closed-form means the defender's Bayes filter
stays exact at every round, and the whole loop runs without a deep-learning
dependency. PPO over the belief observation is available separately via
``CyberGraphDefenseEnv`` if a neural defender is preferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .belief import (
    BeliefUpdater,
    enumerate_binary_states,
    node_compromise_probabilities,
    node_marginals,
)
from .policies import (
    IDLE,
    FocusedAttackerPolicy,
    ThresholdDefenderPolicy,
    _allowed_nodes,
)


@dataclass
class FictitiousPlayConfig:
    """Game and training parameters for fictitious play (edge-free, K attackers)."""

    num_nodes: int
    num_attackers: int = 2
    alpha: float | list[float] | np.ndarray = 0.5
    probe_miss_probability: float = 0.2
    attacker_cost: float = 0.05
    defender_cost: float = 0.1
    control_reward: float = 1.0
    max_defend_nodes: int | None = 1
    horizon: int = 30
    gamma: float = 0.95

    def alpha_vector(self) -> np.ndarray:
        if np.isscalar(self.alpha):
            return np.full(self.num_nodes, float(self.alpha), dtype=np.float64)
        values = np.asarray(self.alpha, dtype=np.float64)
        if values.shape != (self.num_nodes,):
            raise ValueError("alpha must be scalar or have one entry per node.")
        return values


@dataclass
class RoundResult:
    """Returns recorded after one fictitious-play round."""

    defender_return: float
    attacker_returns: list[float]
    defender_threshold: float


def _updater(config: FictitiousPlayConfig, attackers) -> BeliefUpdater:
    from .policies import AttackerEnsemble

    ensemble = AttackerEnsemble(num_nodes=config.num_nodes, attackers=list(attackers))
    return BeliefUpdater(
        num_nodes=config.num_nodes,
        alpha=config.alpha_vector(),
        probe_miss_probability=config.probe_miss_probability,
        attacker_model=ensemble,
    )


def _initial_belief(states: np.ndarray) -> np.ndarray:
    belief = np.zeros(len(states), dtype=np.float64)
    belief[0] = 1.0  # all-clean state is index 0
    return belief


def rollout(
    config: FictitiousPlayConfig,
    defender: ThresholdDefenderPolicy,
    attackers,
    rng: np.random.Generator,
    record_attacker: int | None = None,
):
    """Simulate one episode; optionally record REINFORCE terms for one attacker.

    Returns ``(defender_return, attacker_returns, trajectory)`` where
    ``trajectory`` (only when ``record_attacker`` is set) is a list of
    ``(grad_logits, reward)`` for the recorded attacker at each step.
    """
    states = enumerate_binary_states(config.num_nodes)
    alpha = config.alpha_vector()
    updater = _updater(config, attackers)

    state = np.zeros(config.num_nodes, dtype=np.int8)
    belief = _initial_belief(states)
    detect_prob = 1.0 - config.probe_miss_probability

    defender_return = 0.0
    attacker_returns = [0.0 for _ in attackers]
    trajectory: list[tuple[np.ndarray, float]] = []
    discount = 1.0

    for _ in range(config.horizon):
        marginals = node_marginals(belief, states)
        defense = defender.sample(marginals)

        choices = [a.sample(state, defense, rng) for a in attackers]
        probe_counts = np.zeros(config.num_nodes, dtype=np.int64)
        for choice in choices:
            if choice != IDLE:
                probe_counts[choice] += 1

        q = node_compromise_probabilities(state, probe_counts, defense, alpha)
        next_state = (rng.random(config.num_nodes) < q).astype(np.int8)
        detected = rng.binomial(probe_counts, detect_prob).astype(np.int64)
        belief = updater.update(belief, detected, defense)

        compromised = int(next_state.sum())
        d_reward = config.control_reward * (config.num_nodes - compromised) - (
            config.defender_cost * int(defense.sum())
        )
        defender_return += discount * d_reward

        for i, choice in enumerate(choices):
            a_reward = config.control_reward * compromised - (
                config.attacker_cost * (1 if choice != IDLE else 0)
            )
            attacker_returns[i] += discount * a_reward
            if record_attacker is not None and i == record_attacker:
                grad = _softmax_logit_gradient(
                    attackers[i], state, defense, choice, config.num_nodes
                )
                trajectory.append((grad, a_reward))

        state = next_state
        discount *= config.gamma

    return defender_return, attacker_returns, trajectory


def _softmax_logit_gradient(
    attacker: FocusedAttackerPolicy,
    state: np.ndarray,
    defense: np.ndarray,
    choice: int,
    num_nodes: int,
) -> np.ndarray:
    """grad_logits log pi(choice | state) for a softmax-over-allowed-nodes policy."""
    grad = np.zeros(num_nodes, dtype=np.float64)
    if choice == IDLE:
        return grad
    allowed = _allowed_nodes(state, defense, attacker.clean_only)
    if len(allowed) == 0:
        return grad
    logits = np.asarray(attacker.logits, dtype=np.float64)[allowed]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    # d log p_a / d logit_j = 1{j==a} - p_j   for j in allowed
    grad[allowed] = -probs
    grad[choice] += 1.0
    return grad


def evaluate(
    config: FictitiousPlayConfig,
    defender: ThresholdDefenderPolicy,
    attackers,
    episodes: int,
    rng: np.random.Generator,
) -> tuple[float, list[float]]:
    """Mean discounted returns for the defender and each attacker."""
    d_total = 0.0
    a_total = np.zeros(len(attackers), dtype=np.float64)
    for _ in range(episodes):
        d_ret, a_ret, _ = rollout(config, defender, attackers, rng)
        d_total += d_ret
        a_total += np.asarray(a_ret)
    return d_total / episodes, (a_total / episodes).tolist()


def best_response_defender(
    config: FictitiousPlayConfig,
    attackers,
    rng: np.random.Generator,
    episodes: int = 30,
    grid: np.ndarray | None = None,
) -> ThresholdDefenderPolicy:
    """Grid-search the belief threshold that maximises the defender return."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)
    best_threshold = float(grid[0])
    best_return = -np.inf
    for threshold in grid:
        defender = ThresholdDefenderPolicy(
            num_nodes=config.num_nodes,
            threshold=float(threshold),
            max_defend_nodes=config.max_defend_nodes,
        )
        d_ret, _ = evaluate(config, defender, attackers, episodes, rng)
        if d_ret > best_return:
            best_return = d_ret
            best_threshold = float(threshold)
    return ThresholdDefenderPolicy(
        num_nodes=config.num_nodes,
        threshold=best_threshold,
        max_defend_nodes=config.max_defend_nodes,
    )


def best_response_attacker(
    config: FictitiousPlayConfig,
    defender: ThresholdDefenderPolicy,
    attackers: list,
    index: int,
    rng: np.random.Generator,
    iterations: int = 20,
    episodes: int = 20,
    learning_rate: float = 0.2,
) -> FocusedAttackerPolicy:
    """Improve attacker ``index`` by REINFORCE, holding all others fixed."""
    attackers = list(attackers)
    logits = np.asarray(attackers[index].logits, dtype=np.float64).copy()

    for _ in range(iterations):
        attackers[index] = FocusedAttackerPolicy(
            num_nodes=config.num_nodes, logits=logits, idle_probability=0.0
        )
        grad_acc = np.zeros(config.num_nodes, dtype=np.float64)
        returns: list[float] = []
        per_episode: list[tuple[list[np.ndarray], list[float]]] = []
        for _ in range(episodes):
            _, _, trajectory = rollout(
                config, defender, attackers, rng, record_attacker=index
            )
            grads = [g for g, _ in trajectory]
            rewards = [r for _, r in trajectory]
            returns_to_go = _discounted_returns_to_go(rewards, config.gamma)
            per_episode.append((grads, returns_to_go))
            returns.append(sum(rewards))

        baseline = float(np.mean(returns)) if returns else 0.0
        # REINFORCE with a constant baseline for variance reduction.
        ep_baseline = baseline / max(1, len(per_episode[0][0])) if per_episode else 0.0
        for grads, returns_to_go in per_episode:
            for grad, g_to_go in zip(grads, returns_to_go):
                grad_acc += grad * (g_to_go - ep_baseline)
        grad_acc /= max(1, episodes)
        logits = logits + learning_rate * grad_acc

    return FocusedAttackerPolicy(
        num_nodes=config.num_nodes, logits=logits, idle_probability=0.0
    )


def _discounted_returns_to_go(rewards: list[float], gamma: float) -> list[float]:
    out = [0.0] * len(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def run_fictitious_play(
    config: FictitiousPlayConfig,
    rounds: int = 8,
    seed: int | None = None,
    verbose: bool = True,
) -> tuple[ThresholdDefenderPolicy, list[FocusedAttackerPolicy], list[RoundResult]]:
    """Rotate best-responses (defender, then each attacker) for ``rounds`` rounds."""
    rng = np.random.default_rng(seed)
    attackers: list[FocusedAttackerPolicy] = [
        FocusedAttackerPolicy(
            num_nodes=config.num_nodes,
            logits=np.zeros(config.num_nodes),
            idle_probability=0.0,
        )
        for _ in range(config.num_attackers)
    ]
    defender = ThresholdDefenderPolicy(
        num_nodes=config.num_nodes,
        threshold=0.5,
        max_defend_nodes=config.max_defend_nodes,
    )

    history: list[RoundResult] = []
    for round_index in range(rounds):
        defender = best_response_defender(config, attackers, rng)
        for i in range(config.num_attackers):
            attackers[i] = best_response_attacker(config, defender, attackers, i, rng)

        d_ret, a_ret = evaluate(config, defender, attackers, episodes=50, rng=rng)
        result = RoundResult(
            defender_return=d_ret,
            attacker_returns=a_ret,
            defender_threshold=defender.threshold,
        )
        history.append(result)
        if verbose:
            a_str = ", ".join(f"{r:.2f}" for r in a_ret)
            print(
                f"round {round_index + 1}/{rounds} "
                f"defender_thr={defender.threshold:.2f} "
                f"defender_return={d_ret:.2f} attacker_returns=[{a_str}]"
            )

    return defender, attackers, history
