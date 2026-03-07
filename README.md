# HW2: Tabular RL for Cart-Pole Stabilization

FRA503 Deep Reinforcement Learning for Robotics

## Setup

```bash
conda activate env_isaaclab
cd ~/FRA503_Deep_Reinforcement_Learning/CartPole_4.5.0
```

## Configuration

Edit `scripts/RL_Algorithm/configs/rl_config.json`:

| Parameter                  | Default          | Description                                                   |
| -------------------------- | ---------------- | ------------------------------------------------------------- |
| `num_of_action`            | 5                | Number of discrete actions                                    |
| `action_range`             | [-5.0, 5.0]      | Continuous force range (N)                                    |
| `discretize_state_weight`  | [1, 8, 1, 8]     | State discretization weights for [x, theta, x_dot, theta_dot] |
| `n_episodes`               | 20000            | Total training episodes                                       |
| `start_epsilon`            | 1.0              | Initial exploration rate                                      |
| `epsilon_decay`            | 0.9995           | Decay rate applied per episode or per step                    |
| `epsilon_decay_mode`       | per_step         | Decay mode: `per_step`, `per_episode`, or `fixed`             |
| `final_epsilon`            | 0.01             | Minimum epsilon                                               |
| `discount`                 | 0.99             | Discount factor (gamma)                                       |
| `q_init`                   | 0.0              | Initial Q-value for all state-action pairs                    |

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
- Q-table: `q_value/Stabilize/<algorithm>/<algorithm>_20000_*.json`

## Play (Deploy Trained Policy)

```bash
python scripts/RL_Algorithm/play.py \
    --task Stabilize-Isaac-Cartpole-v0 \
    --algorithm MC \
    --qtable_path experiments/suite_1_baseline/MC.json \
    --num_episodes 10 \
    --output_csv evaluation_results.csv

# With video recording
python scripts/RL_Algorithm/play.py \
    --task Stabilize-Isaac-Cartpole-v0 \
    --algorithm MC \
    --qtable_path experiments/suite_1_baseline/MC.json \
    --num_episodes 10 \
    --video --video_dir videos/MC

# With trajectory logging (for phase portrait plots)
python scripts/RL_Algorithm/play.py \
    --task Stabilize-Isaac-Cartpole-v0 \
    --algorithm MC \
    --qtable_path experiments/suite_1_baseline/MC.json \
    --num_episodes 10 \
    --trajectory_dir trajectories/
```

## Run Experiments

```bash
python run_experiments.py              # run ALL suites
python run_experiments.py 1            # baseline only
python run_experiments.py 1 4 plots    # baseline + deployment + figures
python run_experiments.py plots        # just regenerate figures
python run_experiments.py 1 2 3        # suites 1-3
```

Available suites:

| Suite                  | What It Does                                                        | Output                            |
| ---------------------- | ------------------------------------------------------------------- | --------------------------------- |
| 1 - Baseline           | Train 4 algorithms with default config                              | `experiments/suite_1_baseline/`   |
| 2 - Action Resolution  | Train 4 algos x {3, 5, 11, 21} actions                             | `experiments/suite_2_action/`     |
| 3 - State Resolution   | Train 4 algos x {[1,4,1,4], [1,8,1,8], [2,16,2,16]}               | `experiments/suite_3_state/`      |
| 4 - Deployment         | Evaluate each baseline Q-table (epsilon=0) + video + trajectories   | `experiments/suite_4_deployment/` |
| 5 - Learning Rate      | Train 4 algos x {0.01, 0.05, 0.1, 0.3, 0.5, 0.9}                  | `experiments/suite_5_lr/`         |
| 6 - Epsilon Schedule   | 3 per-step + 2 per-episode + 1 fixed decay configs                  | `experiments/suite_6_epsilon/`    |
| 7 - Discount Factor    | Train 4 algos x {0.9, 0.95, 0.99, 0.999, 1.0}                     | `experiments/suite_7_gamma/`      |
| 8 - Q-init             | Train 4 algos x {0, 10, 50, 100}                                   | `experiments/suite_8_q_init/`     |

After all suites complete, report figures are generated automatically via `plot_report_figures.py`.

## Visualize

```bash
# Generate all 12 report figures at once
python scripts/RL_Algorithm/visualize/plot_report_figures.py --output figures/
```

This generates:

| Figure                       | Content                                          |
| ---------------------------- | ------------------------------------------------ |
| `fig1_feedback_loop.png`     | 1x2: Total Reward + State Coverage               |
| `fig2_credit_assignment.png` | 1x2: Running Max Q-value + TD Error              |
| `fig3_representation.png`    | 2x2: Value heatmaps + Policy heatmaps            |
| `fig4_action_sweep.png`      | 2x2: Action resolution per algorithm             |
| `fig5_state_sweep.png`       | 2x2: State resolution per algorithm              |
| `fig6_deployment.png`        | 2x2: Phase portraits (deployment, best episode)  |
| `fig7_q_surface.png`         | 3D Q-value surfaces per algorithm                |
| `fig8_policy_surface.png`    | 3D Policy surfaces per algorithm                 |
| `fig9_lr_sweep.png`          | 2x2: Learning rate sensitivity per algorithm     |
| `fig10_epsilon_sweep.png`    | 2x2: Epsilon schedule sensitivity per algorithm  |
| `fig11_gamma_sweep.png`      | 2x2: Discount factor sensitivity per algorithm   |
| `fig12_q_init_sweep.png`     | 2x2: Q-init sensitivity per algorithm            |

## Results

```
experiments/
├── suite_1_baseline/          # 4 CSVs + 4 Q-table JSONs
├── suite_2_action/            # 16 CSVs + 16 JSONs (4 algos x 4 action configs)
├── suite_3_state/             # 12 CSVs + 12 JSONs (4 algos x 3 weight configs)
├── suite_4_deployment/
│   ├── evaluation_results.csv # Mean reward & length per algorithm
│   ├── videos/<algorithm>/    # Recorded evaluation videos
│   └── trajectories/          # Per-step trajectory CSVs
├── suite_5_lr/                # 24 CSVs + 24 JSONs (4 algos x 6 LR values)
├── suite_6_epsilon/           # 28 CSVs + 28 JSONs (4 algos x 7 epsilon configs)
├── suite_7_gamma/             # 20 CSVs + 20 JSONs (4 algos x 5 gamma values)
└── suite_8_q_init/            # 16 CSVs + 16 JSONs (4 algos x 4 Q-init values)

figures/
├── fig1_feedback_loop.png
├── fig2_credit_assignment.png
├── fig3_representation.png
├── fig4_action_sweep.png
├── fig5_state_sweep.png
├── fig6_deployment.png
├── fig7_q_surface.png
├── fig8_policy_surface.png
├── fig9_lr_sweep.png
├── fig10_epsilon_sweep.png
├── fig11_gamma_sweep.png
└── fig12_q_init_sweep.png
```

## Project Structure

| File / Folder                                      | Purpose                                            |
| -------------------------------------------------- | -------------------------------------------------- |
| `scripts/RL_Algorithm/train.py`                    | Training loop (algorithm selected via `RL_ALGORITHM` env var) |
| `scripts/RL_Algorithm/play.py`                     | Deployment evaluation (epsilon=0, video, trajectories) |
| `scripts/RL_Algorithm/configs/rl_config.json`      | Hyperparameters                                    |
| `scripts/RL_Algorithm/visualize/plot_report_figures.py` | Generate all report figures                   |
| `run_experiments.py`                               | Automated 8-suite experiment runner + figure generation |
| `RL_Algorithm/Algorithm/MC.py`                     | Monte Carlo (first-visit)                          |
| `RL_Algorithm/Algorithm/SARSA.py`                  | SARSA (on-policy TD)                               |
| `RL_Algorithm/Algorithm/Q_Learning.py`             | Q-Learning (off-policy TD)                         |
| `RL_Algorithm/Algorithm/Double_Q_Learning.py`      | Double Q-Learning                                  |
| `RL_Algorithm/RL_base.py`                          | Base class for tabular algorithms                  |
| `source/CartPole/CartPole/tasks/`                  | IsaacLab task definition (reward, termination, MDP) |
