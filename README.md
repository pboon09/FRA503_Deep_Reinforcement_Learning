# HW2: Tabular RL for Cart-Pole Stabilization

FRA503 Deep Reinforcement Learning for Robotics

## Setup

```bash
conda activate env_isaaclab
cd ~/FRA503_Deep_Reinforcement_Learning/CartPole_4.5.0
```

## Configuration

Edit `scripts/RL_Algorithm/configs/rl_config.json`:

| Parameter                  | Default      | Description                                                   |
| -------------------------- | ------------ | ------------------------------------------------------------- |
| `num_of_action`            | 5            | Number of discrete actions                                    |
| `action_range`             | [-5.0, 5.0]  | Continuous force range                                        |
| `discretize_state_weight`  | [1, 8, 1, 8] | State discretization weights for [x, x_dot, theta, theta_dot] |
| `n_episodes`               | 10000        | Total training episodes                                       |
| `start_epsilon`            | 1.0          | Initial exploration rate                                      |
| `epsilon_decay`            | 0.999        | Per-step decay rate                                           |
| `final_epsilon`            | 0.01         | Minimum epsilon                                               |
| `discount`                 | 0.99         | Discount factor (gamma)                                       |

Per-algorithm learning rates are under `algorithms`:

| Algorithm         | Learning Rate |
| ----------------- | ------------- |
| MC                | 0.1           |
| SARSA             | 0.1           |
| Q_Learning        | 0.1           |
| Double_Q_Learning | 0.1           |

## Train (Single Algorithm)

```bash
RL_ALGORITHM=MC python scripts/RL_Algorithm/train.py \
    --task Stabilize-Isaac-Cartpole-v0 --num_envs 256 --headless
```

Replace `MC` with `SARSA`, `Q_Learning`, or `Double_Q_Learning`.

Outputs:

- CSV log: `logs/Stabilize/<algorithm>/training_log_*.csv`
- Q-table: `q_value/Stabilize/<algorithm>/<algorithm>_10000_*.json`

## Play (Deploy Trained Policy)

```bash
RL_ALGORITHM=MC python scripts/RL_Algorithm/play.py \
    --task Stabilize-Isaac-Cartpole-v0 --num_envs 1

# With video recording
RL_ALGORITHM=MC python scripts/RL_Algorithm/play.py \
    --task Stabilize-Isaac-Cartpole-v0 --num_envs 1 \
    --video --video_length 1000 --video_dir videos/MC
```

## Run All Experiments (Single Command)

```bash
python run_experiments.py
```

This runs all 4 suites automatically:

| Suite                 | What It Does                               | Output                            |
| --------------------- | ------------------------------------------ | --------------------------------- |
| 1 — Baseline          | Train 4 algorithms with default config     | `experiments/suite_1_baseline/`   |
| 2 — Action Resolution | Train 4 algos x {5, 25, 50} actions       | `experiments/suite_2_action/`     |
| 3 — State Resolution  | Train 4 algos x {low, mid, high} weights   | `experiments/suite_3_state/`      |
| 4 — Deployment        | Evaluate each baseline Q-table (epsilon=0) | `experiments/suite_4_deployment/` |

## Visualize

```bash
# Training curves (single algorithm)
python scripts/RL_Algorithm/visualize/plot_training.py \
    --logs "logs/Stabilize/MC/training_log_*.csv" \
    --output figures/

# Training curves (compare all)
python scripts/RL_Algorithm/visualize/plot_training.py \
    --logs "logs/Stabilize/MC/training_log_*.csv" \
           "logs/Stabilize/SARSA/training_log_*.csv" \
           "logs/Stabilize/Q_Learning/training_log_*.csv" \
           "logs/Stabilize/Double_Q_Learning/training_log_*.csv" \
    --output figures/

# Q-table surface + policy heatmap
python scripts/RL_Algorithm/visualize/plot_q_surface.py \
    --qtable "q_value/Stabilize/MC/MC_10000_*.json" \
    --output figures/

# Deployment bar charts
python scripts/RL_Algorithm/visualize/plot_deployment.py \
    --csv experiments/suite_4_deployment/evaluation_results.csv \
    --output figures/suite_4_deployment/

# TensorBoard
tensorboard --logdir logs/Stabilize/
```

## Results

```
experiments/
├── suite_1_baseline/          # 4 CSVs + 4 Q-table JSONs
├── suite_2_action/            # 12 CSVs + 12 JSONs (4 algos x 3 action configs)
├── suite_3_state/             # 12 CSVs + 12 JSONs (4 algos x 3 weight configs)
└── suite_4_deployment/
    └── evaluation_results.csv # Mean reward & length per algorithm

figures/
├── suite_1_baseline/
│   ├── comparison/            # reward_curve, episode_length, epsilon_decay, etc.
│   ├── <algorithm>/           # Per-algorithm heatmaps
│   └── q_surface/<algorithm>/ # 3D Q-surface, policy heatmaps
├── suite_2_action/
│   ├── comparison/            # Cross-algorithm-action comparison plots
│   └── <algorithm>_act_<N>/   # Per-config plots
├── suite_3_state/
│   ├── comparison/            # Cross-algorithm-weight comparison plots
│   └── <algorithm>_<weight>/  # Per-config plots
└── suite_4_deployment/        # deployment_summary.png, bar charts

q_value/Stabilize/<algorithm>/ # Trained Q-tables (JSON)
```

## Project Structure

| File / Folder | Purpose |
|---------------|---------|
| `scripts/RL_Algorithm/train.py` | Training loop |
| `scripts/RL_Algorithm/play.py` | Deployment / video recording |
| `scripts/RL_Algorithm/configs/rl_config.json` | Hyperparameters |
| `scripts/RL_Algorithm/visualize/` | Plotting scripts |
| `source/CartPole/CartPole/tasks/` | IsaacLab task definition (reward, termination) |
| `run_experiments.py` | Automated 4-suite experiment runner |
| `RL_Algorithm/` | Algorithm implementations (MC, SARSA, Q_Learning, Double_Q_Learning) |
