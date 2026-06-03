# Game Learning

This is a reinforcement-learning starting point for the graph cybersecurity game.
It is intentionally closer to the mathematical formalization than the existing
`cybergraph-game` engine:

- graph: `G = (V, E)` with `n = |V|`
- hidden state: `s in {0, 1}^n`, where `1` means attacker-controlled
- attacker action: `A in {0, 1}^n`, a subset of probed nodes
- defender action: `D in {0, 1}^n`, a subset of reimaged nodes
- defender observation: a belief vector over all `2^n` possible hidden states

The first environment is single-agent from the defender's perspective. The
attacker follows a fixed policy `pi_A(A | s, D)`, and a defender RL algorithm
learns which nodes to reimage from the belief state.

## Configs

Experiments live in `configs/*.toml`. Each config has two blocks:

```toml
[env]
name = "path_graph_7"
graph_type = "path"
num_nodes = 7
beta = 0.1
probe_miss_probability = 0.2
attacker_cost = 0.05
defender_cost = 0.5
max_steps = 50
max_attack_nodes = 2
max_defend_nodes = 2
initial_compromised_probability = 0.0

[training]
algorithm = "ppo"
total_timesteps = 10000
seed = 7
device = "cpu"
verbose = 1
model_name = "ppo_defender"
output_root = "outputs"
learning_rate = 0.0003
n_steps = 2048
batch_size = 64
gamma = 0.99
```

The graph belongs to `[env]` because it determines the observation and action
spaces. Training settings and output paths belong to `[training]`.

Config-specific outputs are written to:

```text
outputs/<env.name>/
```

For the default config, PPO saves to:

```text
outputs/path_graph_7/ppo_defender.zip
```

## Quick Start

Run a random non-learned rollout:

```bash
python examples/random_rollout.py
```

Train PPO from a config:

```bash
python examples/train_ppo.py --config configs/path_graph_7.toml
```

Visualize the trained PPO defender for that same config:

```bash
python examples/visualize_ppo.py --config configs/path_graph_7.toml
```

Frames are written to:

```text
outputs/path_graph_7/ppo_rollout_frames/
```

Visualize a random-policy rollout baseline:

```bash
python examples/visual_rollout.py
```

Baseline frames are written to `rollout_frames/`.

## Transition Model

For node `v`, define `beta_v` as the probability that a probe succeeds when the
node is currently clean and not reimaged. With defender action `D` and attacker
action `A`:

```text
q_v(s, A, D) = P(s'_v = 1 | s, A, D)

q_v = 0       if v in D
q_v = 1       if v not in D and s_v = 1
q_v = beta_v  if v not in D, s_v = 0, and v in A
q_v = 0       otherwise
```

The joint transition probability factorizes over nodes.

## Observation and Belief

The defender observes a subset `Y` of attacker probes. There are no false
positives. Each actual probe is missed independently with probability `nu`, so:

```text
L(Y | A) = (1 - nu)^|Y| nu^(|A|-|Y|),  if Y subset A
         = 0,                           otherwise
```

The belief update sums over prior states and attacker actions:

```text
b'(s') proportional to
  sum_s b(s) sum_A pi_A(A | s, D) L(Y | A) K^{A,D}(s' | s)
```

## Next Extensions

- restrict attacks to graph-reachable frontier nodes
- add per-node costs or security levels
- train the attacker too, then alternate fixed-opponent updates
- replace the full `2^n` belief vector with a factored or learned belief model
