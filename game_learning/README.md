# Game Learning

This is a reinforcement-learning starting point for the graph cybersecurity game.

- graph: `G = (V, E)` with `n = |V|`
- hidden state: `s in {0, 1}^n`, where `1` means attacker-controlled
- attacker action: `A in {0, 1}^n`, a subset of probed nodes
- defender action: `D in {0, 1}^n`, a subset of reimaged nodes
- defender observation: a belief vector over all `2^n` possible hidden states

The first environment is single-agent from the defender's perspective. The
attacker follows a fixed policy `pi_A(A | s, D)`, and a defender RL algorithm
learns which nodes to reimage from the belief state.

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

## Quick Start

```bash
python examples/random_rollout.py
```

Visual rollout with saved PNG frames and a Matplotlib live window:

```bash
python examples/visual_rollout.py
```

Frames are written to `rollout_frames/`.

Visualize a trained PPO defender after running `examples/train_ppo.py`:

```bash
python examples/visualize_ppo.py
```

Frames are written to `ppo_rollout_frames/`.

Optional PPO training with Stable-Baselines3:

```bash
pip install stable-baselines3
python examples/train_ppo.py
```

## Next Extensions

- restrict attacks to graph-reachable frontier nodes
- add per-node costs or security levels
- train the attacker too, then alternate fixed-opponent updates
- replace the full `2^n` belief vector with a factored or learned belief model

