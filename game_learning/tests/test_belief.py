import numpy as np

from game_learning.belief import BeliefUpdater, observation_likelihood
from game_learning.policies import UniformAttackerPolicy


def test_observation_likelihood_rejects_false_positive():
    assert observation_likelihood(
        observation=np.array([1, 0]),
        attack=np.array([0, 1]),
        probe_miss_probability=0.2,
    ) == 0.0


def test_belief_update_normalizes():
    attacker = UniformAttackerPolicy(num_nodes=2, max_attack_nodes=1)
    updater = BeliefUpdater(
        num_nodes=2,
        beta=np.array([0.5, 0.5]),
        probe_miss_probability=0.2,
        attacker_policy=attacker,
    )
    belief = np.array([1.0, 0.0, 0.0, 0.0])
    updated = updater.update(
        belief=belief,
        observation=np.array([1, 0]),
        defense=np.array([0, 0]),
    )
    assert np.isclose(updated.sum(), 1.0)
    assert updated.shape == (4,)

