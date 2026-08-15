# REINFORCE Maze Solver

A small reinforcement learning project that trains a REINFORCE policy to solve grid mazes. The project started as a terminal experiment and now includes a Tkinter interface for drawing mazes, generating random mazes, training the agent, and watching the learning metrics update live.

The focus is intentionally educational: the implementation keeps the algorithm visible instead of hiding it behind a larger RL framework.

## What It Does

- Trains a policy-gradient agent with REINFORCE.
- Solves padded grid mazes with legal-action masking.
- Uses a tabular softmax policy over maze cells and actions.
- Uses distance-progress reward shaping to make learning practical from scratch.
- Decays sampling temperature during training.
- Shows training success, returns, path length, and elapsed time.
- Provides a Tkinter desktop UI.

## Project Layout

```text
.
├── global_defs.py          # Training constants and reward settings
├── policy.py               # Tabular policy model
├── reinforce.py            # REINFORCE rollout, reward, and loss logic
├── main.py                 # Terminal training entry point
├── ui/
│   ├── shared/training.py  # Shared training adapter for UI workflows
│   └── tk_app.py           # Tkinter desktop app
└── requirements.txt
```

## Maze Format

Mazes are stored as integer matrices:

| Value | Meaning |
| --- | --- |
| `0` | Open cell |
| `1` | Wall |
| `2` | Start |
| `3` | End |

The maze includes a one-cell padding border. The start and end must be placed on that border. The rest of the padding border is treated as wall.

## How The Agent Learns

The policy is a trainable table of logits:

```text
cell row x cell column x action
```

For each reachable cell, the policy learns a probability distribution over four actions:

```text
right, left, up, down
```

The training loop samples trajectories, computes reward-to-go, normalizes returns, and applies the REINFORCE loss:

```text
loss = -log_prob(action) * normalized_return
```

The reward is shaped by shortest-path distance to the goal:

```text
reward = step_penalty + progress_reward * (old_distance - new_distance)
```

Reaching the goal adds a terminal success reward. This keeps the project recognizably REINFORCE while giving the agent useful feedback before it randomly solves the full maze.

## Running Terminal Training

```bash
python main.py
```

The terminal output shows a compact training table:

```text
epoch        loss   success   avg return  best return     temp
```

Training stops early once the sampled batch reaches `100%` success. A final low-temperature evaluation is printed at the end.

## Running The Tkinter UI

```bash
python ui/tk_app.py
```

The Tkinter app provides the interactive workflow:

- draw a padded maze
- move start and end around the border
- generate a random maze
- train the policy
- watch live stats
- view the final path

## Training Constants

The main constants live in `global_defs.py`:

```python
EPISODE_LIMIT = 100
SAMPLING_QUANTITY = 64
DISCOUNT_VALUE = 0.99
EPOCHS = 120

SUCCESS_REWARD = 10.0
PENALTY = -0.01
PROGRESS_REWARD = 1.0

LEARNING_RATE = 0.03

INITIAL_TEMP = 1.0
MIN_TEMP = 0.1
TEMP_DECAY = 0.02
EVAL_TEMP = 0.05
```

These settings were tuned for fast, reliable learning on `20x20` inner mazes.

## Requirements

Python dependencies are listed in `requirements.txt`:

```text
tensorflow
numpy
maze-utils
tqdm
```

If you use the included `devenv` setup, the project environment is defined in:

```text
devenv.nix
devenv.yaml
```

## Why This Project Is Useful

This project is small enough to read end-to-end, but complete enough to show the real moving parts of policy-gradient learning:

- stochastic rollout sampling
- legal-action masking
- reward shaping
- reward-to-go
- return normalization
- temperature decay
- final policy evaluation

Keep in mind this was done to learn and to demonstrate the fundamental concepts of reinforcement learning, and specifically policy gradient learning.
