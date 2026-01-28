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

### N_RUNS vs N_STEPS

- **N_STEPS**: How many times the agent pulls an arm in one experiment
- **N_RUNS**: How many times we repeat the entire experiment with a fresh bandit and agent

We average results across all runs to get smooth curves.

---

## Part 3: Configure Experiments

Edit `EXPERIMENTS` in `simulation.py`:

```python
EXPERIMENTS = [
    {"strategy": "epsilon_greedy", "epsilon": 0.0},
    {"strategy": "epsilon_greedy", "epsilon": 0.1},
    {"strategy": "ucb", "c": 2.0},
    {"strategy": "epsilon_greedy", "epsilon": 0.0, "initial_q": 5.0, "alpha": 0.1},
]
```

### Available Options

| Key | Description |
|-----|-------------|
| `strategy` | `"epsilon_greedy"` or `"ucb"` |
| `epsilon` | Exploration rate (0.0 - 1.0) |
| `c` | UCB exploration parameter |
| `initial_q` | Initial Q-value for all arms |
| `alpha` | Constant step-size (optional, uses 1/n if not set) |

---

## Part 4: Run Experiments

```bash
python3 simulation.py
```

---

## Part 5: Results

All outputs are saved to `figures/` folder:

```
figures/
├── epsilon_greedy_eps0.0/
│   ├── q_comparison.png         # Estimated Q vs true Q
│   ├── reward_distribution.png  # Violin plot of rewards
│   └── epsilon_greedy_eps0.0_log.json
├── epsilon_greedy_eps0.1/
│   └── ...
├── ucb_c2.0/
│   └── ...
├── combined_rewards.png         # All experiments overlayed
└── combined_optimal.png         # All experiments overlayed
```

---

## Part 6: Log File Structure

Each experiment saves a JSON log with full data:

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
    "final_optimal_pct": 91.5
  },
  "last_run": {
    "true_means": [...],
    "q_estimates": [...],
    "optimal_arm": 3
  }
}
```
