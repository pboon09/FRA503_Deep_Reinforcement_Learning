# HW1: Multi-Armed Bandit

## Part 1: Setup

```bash
pip3 install -r requirements.txt
```

---

## Part 2: Understanding Parameters

### Global Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_ARMS` | 10 | Number of slot machine arms |
| `N_RUNS` | 2000 | Number of independent experiments |
| `N_STEPS` | 1000 | Number of timesteps per experiment |
| `SEED` | 42 | Base random seed for reproducibility |

### N_RUNS vs N_STEPS

- **N_STEPS**: How many times the agent pulls an arm in one experiment
- **N_RUNS**: How many times we repeat the entire experiment with a fresh bandit and agent

We average results across all runs to get smooth curves.

---

## Part 3: Configure Experiments

Edit `EXPERIMENTS` in `simulation.py`:

```python
EXPERIMENTS = [
    {"strategy": "epsilon_greedy", "epsilon": 0.0, "seed": SEED},
    {"strategy": "epsilon_greedy", "epsilon": 0.1, "seed": SEED},
    {"strategy": "ucb", "c": 1.0, "seed": SEED},
    {"strategy": "epsilon_greedy", "epsilon": 0.0, "initial_q": 5.0, "alpha": 0.1, "seed": SEED},
]
```

### Available Options

| Key | Description |
|-----|-------------|
| `strategy` | `"epsilon_greedy"` or `"ucb"` |
| `epsilon` | Exploration rate (0.0 - 1.0) or a function |
| `c` | UCB exploration parameter |
| `initial_q` | Initial Q-value for all arms (default: 0.0) |
| `alpha` | Constant step-size (optional, uses 1/n if None) |
| `seed` | Random seed for reproducibility |

### Epsilon Decay Functions

You can define epsilon as a function of time (`t`) and cumulative reward (`r`):

```python
def time_decay(t, r):
    return 1.0 / (1 + 0.01 * t)

EXPERIMENTS = [
    {"strategy": "epsilon_greedy", "epsilon": 0.1, "seed": SEED},           # Constant
    {"strategy": "epsilon_greedy", "epsilon": time_decay, "seed": SEED},    # Decay with time
    {"strategy": "epsilon_greedy", "epsilon": lambda t, r: 1/(1+t), "seed": SEED},  # Inline
]
```

Parameters passed to the function:
- `t`: Current timestep (starts at 0)
- `r`: Cumulative reward so far

---

## Part 4: Run Experiments

```bash
python3 simulation.py
```

---

## Part 5: Results

All outputs are saved to `figures/` and `logs/` folders:

```
figures/
├── epsilon_greedy_eps0.0/
│   ├── q_comparison.png
│   ├── reward_distribution.png
│   ├── action_counts.png
│   └── q_error.png
├── ucb_c1.0/
│   └── ...
├── combined_rewards.png
├── combined_optimal.png
├── combined_regret.png
├── combined_q_error.png
├── group_epsilon_greedy.png
├── group_ucb.png
├── group_optimistic_greedy.png
├── group_optimistic_ucb.png
└── best_overlay.png

logs/
├── epsilon_greedy_eps0.0/
│   ├── epsilon_greedy_eps0.0_log.json
│   └── epsilon_greedy_eps0.0_results.csv
└── ucb_c1.0/
    └── ...
```

---

## Part 6: Log File Structure

### JSON Log (summary)

Each experiment saves a JSON log with summary data:

```json
{
  "experiment": "epsilon_greedy_eps0.0",
  "label": "epsilon_greedy: ε=0.0",
  "config": { ... },
  "parameters": {
    "n_arms": 10,
    "n_runs": 2000,
    "n_steps": 1000
  },
  "results": {
    "final_avg_reward": 1.23,
    "final_optimal_pct": 91.5,
    "final_cumulative_regret": 58.5,
    "final_q_error": 0.45,
    "total_optimal_selections": 1725033,
    "total_suboptimal_selections": 274967
  },
  "last_run": {
    "true_means": [...],
    "q_estimates": [...],
    "optimal_arm": 3,
    "action_counts": [1, 2, 5, 980, ...]
  }
}
```

### CSV Log (time series)

Each experiment also saves a CSV with step-by-step data for analysis:

```csv
step,avg_reward,optimal_pct,cumulative_regret,q_error
1,0.123,10.5,0.5,1.2
2,0.456,15.2,0.8,1.1
...
1000,1.234,91.5,58.5,0.45
```

Use this CSV for generating reports or custom plots.
