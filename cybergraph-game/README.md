# cybergraph-game

`cybergraph-game` is a graph-based cybersecurity simulation playground.

The project models a true graph state, limited attacker/defender knowledge, simple actions, and live Matplotlib visualization. Nodes can be defended or captured. Entry points are possible initial attack surfaces, and node `security_level` values reduce the attacker's success probability.

```text
effective_probability = ATTACK_SUCCESS_PROBABILITY / security_level
```

The example advances one time step whenever you press Enter in the Matplotlib window. Press `q` in the graph window to quit.

## Examples

The project currently keeps two self-contained examples:

```bash
python examples/scenario_random_standard.py
python examples/scenario_strong_attacker.py
```

`scenario_random_standard.py` runs a static graph where the attacker randomly attacks known legal nodes and the defender randomly reimages known captured nodes or does nothing.

`scenario_strong_attacker.py` uses a high attack success probability and an attacker that always attacks a known legal target.

All scenario values live at the top of each file: graph size, seeds, time steps, security levels, attack success probability, defender alert probability, and defender behavior.

## Information Model

The attacker initially knows only entry points. The attacker can only attack known nodes that are legal attack targets. When the attacker captures a node, the attacker permanently learns that node's neighbors. Restoring a node does not erase attacker knowledge.

The defender knows the full graph topology and security levels, but does not automatically know which nodes are captured. After each attack, the game rolls `ATTACK_SEEN_PROBABILITY`; if the attack is seen and succeeded, the defender records that node as known captured.

Policies should depend only on their observation objects, not on the hidden true graph state.

## Actions

The attacker can attack a known legal target or do nothing.

The defender can restore a known captured node, add a defended node with edges, increase or decrease a node's security level, or do nothing.

## Architecture

Core modules:

- `model.py`: shared data types plus graph node attribute helpers.
- `attacker.py`: attacker knowledge, observations, and attacker policies.
- `defender.py`: defender knowledge, observations, and defender policies.
- `game.py`: legal actions, probabilities, knowledge updates, and state transitions.
- `graph_init.py`: graph creation.
- `visualization.py`: static and live graph drawing.

Future milestones may add noisy observations, richer defender alerts, rewards, costs, reimaging, attacker learning, reinforcement learning, and optimization experiments.

## Installation

```bash
pip install -r requirements.txt
```
