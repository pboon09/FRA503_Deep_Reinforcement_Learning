# HW1: Multi-Armed Bandit

## Setup

```bash
pip install -r requirements.txt
```

## Configure Experiments

Edit `EXPERIMENTS` in `simulation.py`:

```python
EXPERIMENTS = {
    "exp1": {
        "strategy": "epsilon_greedy",
        "epsilon": 0.0,
        "initial_q": 0.0,
    },
    "exp2": {
        "strategy": "epsilon_greedy",
        "epsilon": 0.1,
        "initial_q": 0.0,
    },
    "exp3": {
        "strategy": "ucb",
        "c": 2.0,
        "initial_q": 0.0,
    },
    "exp4": {
        "strategy": "epsilon_greedy",
        "epsilon": 0.0,
        "initial_q": 5.0,
        "alpha": 0.1,
    },
}
```

### Available Options

| Key | Description |
|-----|-------------|
| `strategy` | `"epsilon_greedy"` or `"ucb"` |
| `epsilon` | Exploration rate (0.0 - 1.0) |
| `c` | UCB exploration parameter |
| `initial_q` | Initial Q-value for all arms |
| `alpha` | Constant step-size (optional, uses 1/n if not set) |

### Global Parameters

```python
N_ARMS = 10
N_RUNS = 2000
N_STEPS = 1000
```

## Run

```bash
python simulation.py
```

## Results

```
figures/
├── exp1/
│   ├── rewards.png
│   ├── optimal_action.png
│   ├── actions.png
│   ├── q_comparison.png
│   ├── reward_distribution.png
│   └── exp1_log.json
├── exp2/
│   └── ...
├── combined_rewards.png
└── combined_optimal.png
```
