# Game Learning — Multi-Attacker Moving Target Defence

A reinforcement-learning environment for a **partially observable stochastic
game** between a defender and **several attackers**, following Datar & Dujardin,
*"Adaptive Learning for Moving Target Defence"* (CoDIT 2025), generalised to
multiple nodes and multiple attackers.

- **nodes**: `n` independent nodes — **no edges**. Each node is its own
  attack/defence surface; the transition and observation kernels factorise over
  nodes.
- **hidden state**: `s in {0, 1}^n`, where `1` means attacker-controlled.
- **attackers**: `K` independent attackers. Each step every attacker probes at
  most one node (or stays idle). What matters for the dynamics is the per-node
  **probe count** `rho_v` = how many attackers hit node `v` this step.
- **defender action**: `D in {0, 1}^n`, the subset of reimaged nodes.
- **defender observation**: a Bayes-exact belief over all `2^n` hidden states,
  driven by detected probe counts.

## Coordination bonus

The compromise model is the paper's memoryless exponential, with `rho_v` being
the number of **simultaneous** probes on node `v`:

```text
q_v(s, rho, D) = P(s'_v = 1 | s, rho, D)

q_v = 0                      if v in D            (reimaged -> clean)
q_v = 1                      if v not in D, s_v = 1 (stays compromised)
q_v = 1 - exp(-alpha_v * rho_v)  if v not in D, s_v = 0 (clean, probed)
q_v = 0                      otherwise            (rho_v = 0 => q_v = 0)
```

Because `rho_v` adds up the probes, **attackers that coordinate on the same node
in the same step are strictly more likely to compromise it** than attackers that
spread out. Larger `alpha_v` makes a single probe more dangerous.

## Observation and belief

Each individual probe is detected independently with probability `1 - nu`
(`nu = probe_miss_probability`), with no false positives. So the number detected
on node `v` is `Binomial(rho_v, 1 - nu)`, and the likelihood of an observed
detected-count vector `Y` given the probe counts `rho` is:

```text
L(Y | rho) = prod_v  C(rho_v, Y_v) (1 - nu)^{Y_v} nu^{rho_v - Y_v},  Y_v <= rho_v
           = 0                                                        otherwise
```

The HMM belief filter marginalises over prior states and the joint attacker
action (the exact distribution over probe-count vectors from the attacker
roster):

```text
b'(s') proportional to
  sum_s b(s) sum_rho pi_A(rho | s, D) L(Y | rho) K^{rho,D}(s' | s)
```

Keeping the attacker policies closed-form (see below) keeps this filter exact.

## Players and learning

The package supports two ways to learn:

1. **PPO defender** (neural). `CyberGraphDefenseEnv` is a Gymnasium env where the
   defender acts on the belief vector while a fixed attacker roster probes.
2. **Fictitious play** (paper Algorithm 1, generalised to `D, A_1, ..., A_K`).
   Rotate best responses: improve the defender, then each attacker, then repeat.
   - **Defender**: a *threshold policy over belief marginals* — the structure the
     paper proves optimal — reimaging the most-suspect nodes up to budget.
   - **Attackers**: closed-form *softmax-over-nodes* policies whose logits are
     improved by REINFORCE. Being closed-form, they keep the belief filter exact,
     and the whole loop runs without a deep-learning dependency.

## Configs

Experiments live in `configs/*.toml` with an `[env]` and a `[training]` block:

```toml
[env]
name = "path_graph_7"
graph_type = "path"          # edge-free; only labels/positions nodes for plots
num_nodes = 7
alpha = 0.4                  # per-node compromise rate (scalar or list)
probe_miss_probability = 0.2 # nu: chance a probe goes undetected
attacker_cost = 0.05
defender_cost = 0.5
control_reward = 1.0         # reward per controlled node per step
max_steps = 50
num_attackers = 2            # K independent attackers
attacker_idle_probability = 0.0
max_defend_nodes = 2         # reimage budget per step
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

Load a config programmatically:

```python
from game_learning.experiment_config import load_experiment_config, build_game_config
from game_learning import CyberGraphDefenseEnv

cfg = load_experiment_config("configs/path_graph_7.toml")
graph, game_config = build_game_config(cfg)
env = CyberGraphDefenseEnv(game_config)
```

Config-specific outputs are written to `outputs/<env.name>/`.

> Note: the belief vector has length `2^n`, so the exact filter is meant for
> modest `n` (roughly `n <= 8`). Larger graphs need a factored/learned belief.

## Quick start

All commands assume the project venv (see the repo root `.venv`). Run from this
`game_learning/` directory.

Random-defender rollout against multiple attackers:

```bash
python examples/random_rollout.py
```

Fictitious play — defender vs. two coordinating attackers (prints where the
attackers learn to focus their probes):

```bash
python examples/fictitious_play_demo.py
```

Train a PPO defender (requires `stable-baselines3`):

```bash
python examples/train_ppo.py
```

Visualise a random rollout, or a trained PPO defender:

```bash
python examples/visual_rollout.py
python examples/visualize_ppo.py     # after train_ppo.py
```

## Code map

| Module | Role |
|--------|------|
| `belief.py` | exponential compromise model, binomial observation likelihood, exact HMM belief filter, belief marginals |
| `policies.py` | single-attacker policies (`UniformAttackerPolicy`, `FocusedAttackerPolicy`), `AttackerEnsemble` (joint probe-count distribution), `ThresholdDefenderPolicy` |
| `env.py` | `CyberGraphDefenseEnv` (Gymnasium) + `GameConfig` |
| `fictitious_play.py` | rotating best-response training for `D, A_1, ..., A_K` |
| `experiment_config.py` | load `[env]`/`[training]` TOMLs, build envs and output paths |
| `visualization.py` | per-step plots and a live view |

## Next extensions

- per-node heterogeneity: distinct `alpha_v`, per-node costs or security levels
- richer (state-dependent) attacker policies while keeping the filter tractable
- a neural attacker that exposes its action distribution to the belief filter
- replace the full `2^n` belief with a factored or learned belief model
