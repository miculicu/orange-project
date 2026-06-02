# cybergraph-game

`cybergraph-game` is a graph-based cybersecurity simulation playground.

The current milestone implements graph initialization, live visualization, and simple time-step actions. Nodes can be defended or captured, and entry points are visualized differently because they represent possible initial attack surfaces.

The attacker can attempt to capture legal target nodes with a global success probability. Initially, defended entry points are attackable. After the attacker captures a node, defended neighbors of any captured node become attackable too.

The defender can restore one captured node to the defended state or add a new defended node with edges to the graph. In each time step, the attacker and defender may each take one action or choose no action. The demo visualizes the graph continuously in one Matplotlib window and shows the current time step with the latest actions.

Each node has a `security_level` from 1 to 3. Higher-security nodes are drawn with more boundary rings. The attacker's effective success probability is:

```text
effective_probability = ATTACK_SUCCESS_PROBABILITY / security_level
```

Simulation defaults such as graph size, random seeds, number of time steps, attack success probability, and defender action probability live in `cybergame/config.py`.

The demo advances one time step whenever you press Enter in the Matplotlib window, so you can read the action log before continuing. Press `ü`, `q`, or Escape in the graph window to quit the simulation.

Future milestones may add attacker learning, noisy observations, reimaging, stochastic attack models, costs, rewards, and richer game dynamics.

## Scenarios

The `examples/` folder contains scenario scripts with local settings that can be edited independently, including graph size, seeds, time steps, security range, attack success probability, and defender behavior.

```bash
python examples/scenario_random_small.py
python examples/scenario_attacker_always_big.py
```

`scenario_random_small.py` uses the default small graph settings and random attacker/defender policies.

`scenario_attacker_always_big.py` uses a larger graph, an attacker that always attacks a known legal target, and a defender that never acts.

Attacker policies are given limited knowledge. The attacker initially knows entry points and their neighboring nodes. When the attacker captures a node, the attacker permanently learns that node's neighbors, even if the defender later restores the node. The attacker does not use hidden node security levels for policy choice, though the real attack success probability still depends on security level.

The live layout keeps existing node positions fixed when the defender adds new nodes, so the graph should not rotate or jump as much between time steps.

## Architecture

The simulation is organized around a clean information boundary:

```text
true GameState -> Observation -> Policy -> Action -> Rules/Engine -> updated GameState
```

Core modules:

- `model.py`: shared enums, action dataclasses, results, rule config, and game state.
- `state.py`: safe helpers for graph node attributes.
- `rules.py`: legal actions and state transitions.
- `observations.py`: attacker and defender observations.
- `policies.py`: policies that consume observations only.
- `engine.py`: one-step simulation loop.
- `scenario.py`: live scenario runner and scenario configuration.

This means future attacker or defender policies should not receive the true graph directly. They should receive an observation object. This keeps hidden information, noisy information, reinforcement learning, and optimization experiments easier to add later.

## Installation

```bash
pip install -r requirements.txt
```

## Demo

```bash
python examples/demo_graph_init.py
```

The original demo entry point now runs the random small scenario.
