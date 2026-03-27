# HW3: Function Approximation RL for Cart-Pole Stabilization

FRA503 Deep Reinforcement Learning for Robotics

## Setup

```bash
conda activate env_isaaclab
cd ~/FRA503_Deep_Reinforcement_Learning/CartPole_4.5.0
```

## Configuration

Edit `scripts/Function_based/configs/rl_config.json`:

### Shared Parameters

| Parameter         | Default      | Description                              |
| ----------------- | ------------ | ---------------------------------------- |
| `n_episodes`      | 20000        | Total training episodes                  |
| `n_observations`  | 4            | Observation space dimension              |
| `action_range`    | [-2.5, 2.5]  | Continuous force range (N)               |
| `discount_factor` | 0.99         | Discount factor (gamma)                  |

### Per-Algorithm Learning Rates

| Algorithm     | Learning Rate |
| ------------- | ------------- |
| Linear_Q      | 0.1           |
| DQN           | 0.0001        |
| MC_REINFORCE  | 0.0005        |
| AC            | 0.0007        |
| PPO           | 0.0003        |

### Key Algorithm-Specific Parameters

| Algorithm    | Key Parameters                                                                         |
| ------------ | -------------------------------------------------------------------------------------- |
| Linear_Q     | `num_of_action`: 11, `lr_decay`: 0.9999, `epsilon_decay`: 9.5e-6                      |
| DQN          | `num_of_action`: 11, `hidden_dim`: 128, `tau`: 0.005, `buffer_size`: 1000000, `batch_size`: 128, `learning_starts`: 10000 |
| MC_REINFORCE | `action_type`: continuous, `hidden_dim`: 64, `rollout_steps`: 64, `entropy_coef`: 0.01 |
| AC           | `action_type`: continuous, `hidden_dims`: [64, 64], `rollout_steps`: 16, `init_noise_std`: 0.6 |
| PPO          | `action_type`: continuous, `hidden_dims`: [64, 64], `clip_param`: 0.2, `lam`: 0.95, `num_transitions_per_env`: 24 |

## Train (Single Algorithm)

```bash
RL_ALGORITHM=PPO python scripts/Function_based/train.py \
    --task Stabilize-Isaac-Cartpole-v0 --num_envs 256 --headless
```

Recommended `--num_envs` per algorithm:

| Algorithm    | num_envs |
| ------------ | -------- |
| PPO          | 256      |
| MC_REINFORCE | 16       |
| AC           | 8        |
| DQN          | 32       |
| Linear_Q     | 32       |

Replace `PPO` with `Linear_Q`, `DQN`, `MC_REINFORCE`, or `AC`.

Outputs:

- CSV log: `experiments/suite_1_baseline/<algorithm>.csv`
- Metrics log: `experiments/suite_1_baseline/<algorithm>_metrics.csv`
- Model: `model/Stabilize/<algorithm>/`

## Play (Deploy Trained Policy)

```bash
RL_ALGORITHM=PPO python scripts/Function_based/play.py \
    --task Stabilize-Isaac-Cartpole-v0 --num_envs 1 --headless
```

Output:

- Deployment CSV: `experiments/suite_1_baseline/<algorithm>_deploy.csv`

With video recording:

```bash
RL_ALGORITHM=PPO python scripts/Function_based/play.py \
    --task Stabilize-Isaac-Cartpole-v0 --num_envs 1 --headless --video
```

## Run Experiments

```bash
python run_experiments.py              # train + deploy + plots (all algorithms)
python run_experiments.py train        # train only
python run_experiments.py deploy       # deploy only
python run_experiments.py plots        # just regenerate figures
python run_experiments.py train plots  # train then generate figures
python run_experiments.py --algos DQN PPO  # specific algorithms only
```

## Visualize

```bash
# Generate all report figures
python scripts/Function_based/visualize/plot_report_figures.py --output figures/

# Generate metrics figures (loss, entropy, grad norm, etc.)
python scripts/Function_based/visualize/plot_metrics_figures.py --output figures/
```

Report figures generated:

| Figure                          | Content                                             |
| ------------------------------- | --------------------------------------------------- |
| `fig1_learning_curves.png`      | Return vs episode (all algos, mean ± std band)      |
| `fig2_sample_efficiency.png`    | Return vs total env steps (fair x-axis)             |
| `fig3_deployment.png`           | (a) bar chart + (b) per-episode scatter             |
| `fig4_convergence_speed.png`    | Horizontal bar: steps to reach threshold            |
| `fig5_reward_per_step.png`      | Reward efficiency (return/length) over episodes     |
| `fig6_epsilon_decay.png`        | Epsilon decay schedule per algorithm                |
| `fig7_episode_length.png`       | Episode length over training                        |
| `fig8_deployment_boxplot.png`   | Deployment reward distribution (box plot)           |
| `fig9_policy_surface.png`       | Policy surface visualisation                        |
| `fig10_value_surface.png`       | Value function surface visualisation                |
| `fig11_radar_chart.png`         | Multi-metric radar chart (all algos)                |
| `fig12_loss_curves.png`         | Training loss curves                                |
| `fig13_entropy.png`             | Policy entropy over training                        |
| `fig14_grad_norm.png`           | Gradient norm over training                         |
| `fig15_explained_variance.png`  | Explained variance of value function                |
| `fig16_ppo_clip_fraction.png`   | PPO clip fraction over training                     |
| `fig17_dqn_q_values.png`        | DQN Q-value estimates over training                 |
| `fig18_wall_clock.png`          | Wall-clock training time comparison                 |

## Results

```
experiments/
└── suite_1_baseline/
    ├── AC.csv
    ├── AC_deploy.csv
    ├── AC_metrics.csv
    ├── DQN.csv
    ├── DQN_deploy.csv
    ├── DQN_metrics.csv
    ├── Linear_Q.csv
    ├── Linear_Q_deploy.csv
    ├── MC_REINFORCE.csv
    ├── MC_REINFORCE_deploy.csv
    ├── PPO.csv
    ├── PPO_deploy.csv
    ├── PPO_metrics.csv
    └── timing_log.txt

model/
└── Stabilize/
    ├── AC/
    ├── DQN/
    ├── Linear_Q/
    ├── MC_REINFORCE/
    └── PPO/

figures/
├── fig1_learning_curves.png
├── fig2_sample_efficiency.png
├── fig3_deployment.png
├── fig4_convergence_speed.png
├── fig5_reward_per_step.png
├── fig6_epsilon_decay.png
├── fig7_episode_length.png
├── fig8_deployment_boxplot.png
├── fig9_policy_surface.png
├── fig10_value_surface.png
├── fig11_radar_chart.png
├── fig12_loss_curves.png
├── fig13_entropy.png
├── fig14_grad_norm.png
├── fig15_explained_variance.png
├── fig16_ppo_clip_fraction.png
├── fig17_dqn_q_values.png
└── fig18_wall_clock.png
```

## Project Structure

| File / Folder                                                        | Purpose                                                              |
| -------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `scripts/Function_based/train.py`                                    | Training loop (algorithm selected via `RL_ALGORITHM` env var)        |
| `scripts/Function_based/play.py`                                     | Deployment evaluation (deterministic inference)                      |
| `scripts/Function_based/configs/rl_config.json`                      | Hyperparameters for all algorithms                                   |
| `scripts/Function_based/visualize/plot_report_figures.py`            | Generate main report figures (fig1–fig10)                            |
| `scripts/Function_based/visualize/plot_metrics_figures.py`           | Generate detailed metrics figures (fig11–fig18)                      |
| `run_experiments.py`                                                 | Automated train → deploy → plot runner                               |
| `RL_Algorithm/RL_base_function.py`                                   | Base class: `scale_action`, `decay_epsilon`, `plot_durations`        |
| `RL_Algorithm/storage/buffers.py`                                    | `RolloutBuffer` (on-policy) and `ReplayBuffer` (off-policy)          |
| `RL_Algorithm/storage/on_policy.py`                                  | `OnPolicyAlgorithm` base for PPO / A2C                               |
| `RL_Algorithm/storage/off_policy.py`                                 | `OffPolicyAlgorithm` base for DQN / TD3 / SAC                        |
| `RL_Algorithm/networks/mlp.py`                                       | Shared `MLP` backbone (relu / elu / tanh)                            |
| `RL_Algorithm/Function_based/Linear_Q.py`                            | Linear function approximation Q-Learning                             |
| `RL_Algorithm/Function_based/DQN.py`                                 | Deep Q-Network with experience replay                                |
| `RL_Algorithm/Function_based/MC_REINFORCE.py`                        | MC REINFORCE policy gradient                                         |
| `RL_Algorithm/Function_based/AC.py`                                  | Actor-Critic (MC episodic)                                           |
| `RL_Algorithm/Function_based/PPO.py`                                 | Proximal Policy Optimization (clipped surrogate + GAE)               |
| `source/CartPole/CartPole/tasks/`                                    | IsaacLab task definition (reward, termination, MDP)                  |
