import numpy as np

from game_learning.belief import (
    BeliefUpdater,
    node_compromise_probabilities,
    observation_likelihood,
)
from game_learning.policies import AttackerEnsemble, UniformAttackerPolicy


def test_observation_likelihood_rejects_more_detections_than_probes():
    assert observation_likelihood(
        observation=np.array([1, 0]),
        probe_counts=np.array([0, 1]),
        probe_miss_probability=0.2,
    ) == 0.0


def test_observation_likelihood_binomial_value():
    # 2 probes, 1 detected, detect prob 0.8: C(2,1)*0.8*0.2 = 0.32
    value = observation_likelihood(
        observation=np.array([1]),
        probe_counts=np.array([2]),
        probe_miss_probability=0.2,
    )
    assert np.isclose(value, 0.32)


def test_coordination_increases_compromise_probability():
    alpha = np.array([0.5])
    state = np.array([0], dtype=np.int8)
    defense = np.array([0], dtype=np.int8)
    one = node_compromise_probabilities(state, np.array([1]), defense, alpha)[0]
    two = node_compromise_probabilities(state, np.array([2]), defense, alpha)[0]
    three = node_compromise_probabilities(state, np.array([3]), defense, alpha)[0]
    assert one < two < three
    assert np.isclose(two, 1.0 - np.exp(-1.0))


def test_reimage_resets_and_compromised_stays():
    alpha = np.array([0.5, 0.5])
    state = np.array([1, 1], dtype=np.int8)
    # node 0 reimaged -> clean (q=0); node 1 undefended & compromised -> stays (q=1)
    q = node_compromise_probabilities(state, np.array([3, 0]), np.array([1, 0]), alpha)
    assert np.allclose(q, [0.0, 1.0])


def test_belief_update_normalizes():
    ensemble = AttackerEnsemble(
        num_nodes=2,
        attackers=[UniformAttackerPolicy(num_nodes=2), UniformAttackerPolicy(num_nodes=2)],
    )
    updater = BeliefUpdater(
        num_nodes=2,
        alpha=np.array([0.5, 0.5]),
        probe_miss_probability=0.2,
        attacker_model=ensemble,
    )
    belief = np.array([1.0, 0.0, 0.0, 0.0])
    updated = updater.update(
        belief=belief,
        observation=np.array([1, 0]),
        defense=np.array([0, 0]),
    )
    assert np.isclose(updated.sum(), 1.0)
    assert updated.shape == (4,)


def test_joint_action_distribution_sums_to_one():
    ensemble = AttackerEnsemble(
        num_nodes=3,
        attackers=[UniformAttackerPolicy(num_nodes=3) for _ in range(2)],
    )
    dist = ensemble.joint_action_distribution(
        state=np.zeros(3, dtype=np.int8), defense=np.zeros(3, dtype=np.int8)
    )
    total = sum(prob for _, prob in dist)
    assert np.isclose(total, 1.0)
    # two attackers, three nodes -> probe counts sum to 2 everywhere
    assert all(int(counts.sum()) == 2 for counts, _ in dist)
