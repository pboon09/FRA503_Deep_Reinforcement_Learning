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
| `total_steps`     | 20000        | Total batch steps per algorithm          |
| `n_observations`  | 4            | Observation space dimension              |
| `action_range`    | [-2.5, 2.5]  | Continuous force range (N)               |
| `discount_factor` | 0.99         | Discount factor (gamma)                  |
| `num_envs`        | 256          | Number of parallel environments          |

### Per-Algorithm Learning Rates

| Algorithm     | Learning Rate       |
| ------------- | ------------------- |
| Linear_Q      | 0.01                |
| DQN           | 0.0001              |
| MC_REINFORCE  | 0.0005              |
| AC            | 0.0003              |
| A2C           | 0.0005              |
| PPO           | 0.0003              |
| SAC           | 0.0003              |
| TD3           | 0.0003              |

### Key Algorithm-Specific Parameters

| Algorithm    | Key Parameters                                                                         |
| ------------ | -------------------------------------------------------------------------------------- |
| Linear_Q     | `num_of_action`: 11, `epsilon_decay`: 5e-5, `final_epsilon`: 0.01                     |
| DQN          | `num_of_action`: 11, `hidden_dim`: 128, `target_update_freq`: 1000, `buffer_size`: 1000000, `batch_size`: 128, `learning_starts`: 5000 |
| MC_REINFORCE | `action_type`: continuous, `hidden_dim`: 64                                            |
| AC           | `action_type`: continuous, `hidden_dims`: [256, 256], `init_noise_std`: 1.0            |
| A2C          | `action_type`: continuous, `hidden_dims`: [256, 256], `num_transitions_per_env`: 4     |
| PPO          | `action_type`: continuous, `hidden_dims`: [256, 256], `clip_param`: 0.2, `lam`: 0.95, `num_transitions_per_env`: 20 |
| SAC          | `hidden_dim`: 256, `tau`: 0.005, `buffer_size`: 1000000, `batch_size`: 256, `auto_alpha`: true, `learning_starts`: 1000 |
| TD3          | `hidden_dim`: 256, `tau`: 0.005, `buffer_size`: 2000000, `batch_size`: 256, `exploration_noise`: 0.1, `policy_update_freq`: 2, `learning_starts`: 1000 |

## Train & Deploy (All Algorithms)

```bash
python run_experiment.py --task Stabilize-Isaac-Cartpole-v0 --headless --num_envs 256
```

Single algorithm:

```bash
python run_experiment.py --task Stabilize-Isaac-Cartpole-v0 --headless --num_envs 256 --algo SAC
```

Replace `SAC` with `Linear_Q`, `DQN`, `MC_REINFORCE`, `AC`, `A2C`, `PPO`, or `TD3`.

Outputs:

- Training CSV: `experiments/suite_1_baseline/<algorithm>.csv`
- Loss CSV: `experiments/suite_1_baseline/<algorithm>_loss.csv`
- Deployment CSV: `experiments/suite_1_baseline/<algorithm>_deploy.csv`
- Trajectory CSV: `experiments/suite_1_baseline/<algorithm>_traj.csv`
- Model: `model/Stabilize/<algorithm>/`

## Results

```
experiments/
└── suite_1_baseline/
    ├── Linear_Q.csv
    ├── Linear_Q_loss.csv
    ├── Linear_Q_deploy.csv
    ├── Linear_Q_traj.csv
    ├── DQN.csv
    ├── DQN_loss.csv
    ├── DQN_deploy.csv
    ├── DQN_traj.csv
    ├── MC_REINFORCE.csv
    ├── MC_REINFORCE_deploy.csv
    ├── MC_REINFORCE_traj.csv
    ├── AC.csv
    ├── AC_loss.csv
    ├── AC_deploy.csv
    ├── AC_traj.csv
    ├── A2C.csv
    ├── A2C_loss.csv
    ├── A2C_deploy.csv
    ├── A2C_traj.csv
    ├── PPO.csv
    ├── PPO_loss.csv
    ├── PPO_deploy.csv
    ├── PPO_traj.csv
    ├── SAC.csv
    ├── SAC_loss.csv
    ├── SAC_deploy.csv
    ├── SAC_traj.csv
    ├── TD3.csv
    ├── TD3_loss.csv
    ├── TD3_deploy.csv
    └── TD3_traj.csv

model/
└── Stabilize/
    ├── Linear_Q/
    ├── DQN/
    ├── MC_REINFORCE/
    ├── AC/
    ├── A2C/
    ├── PPO/
    ├── SAC/
    └── TD3/

figures/
├── fig1_learning_curves.png
├── fig2_sample_efficiency.png
├── fig3_deployment.png
├── fig6_epsilon_decay.png
├── fig7_episode_length.png
├── fig8_deployment_boxplot.png
├── fig9_policy_surface.png
├── fig10_value_surface.png
├── fig3_policy_value_surfaces.png
├── fig_phase_portrait.png
├── fig_train_deploy_gap.png
└── fig_action_histogram.png
```

## Project Structure

| File / Folder                                                        | Purpose                                                              |
| -------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `run_experiment.py`                                                  | Unified train + deploy + plot pipeline                               |
| `scripts/Function_based/configs/rl_config.json`                      | Hyperparameters for all algorithms                                   |
| `RL_Algorithm/RL_base_function.py`                                   | Base class: `scale_action`, `decay_epsilon`, `plot_durations`        |
| `RL_Algorithm/storage/buffers.py`                                    | `RolloutBuffer` (on-policy) and `ReplayBuffer` (off-policy)         |
| `RL_Algorithm/storage/on_policy.py`                                  | `OnPolicyAlgorithm` base for PPO / A2C                              |
| `RL_Algorithm/storage/off_policy.py`                                 | `OffPolicyAlgorithm` base for DQN / TD3 / SAC                       |
| `RL_Algorithm/networks/mlp.py`                                       | Shared `MLP` backbone (relu / elu / tanh)                            |
| `RL_Algorithm/Function_based/Linear_Q.py`                            | Linear function approximation Q-Learning                             |
| `RL_Algorithm/Function_based/DQN.py`                                 | Deep Q-Network with experience replay                                |
| `RL_Algorithm/Function_based/MC_REINFORCE.py`                        | MC REINFORCE policy gradient with baseline                           |
| `RL_Algorithm/Function_based/AC.py`                                  | Actor-Critic (per-step TD, S&B Ch. 13.5)                            |
| `RL_Algorithm/Function_based/A2C.py`                                 | Advantage Actor-Critic (GAE lambda=1.0)                              |
| `RL_Algorithm/Function_based/PPO.py`                                 | Proximal Policy Optimization (clipped surrogate + GAE)               |
| `RL_Algorithm/Function_based/SAC.py`                                 | Soft Actor-Critic (max entropy, auto-alpha)                          |
| `RL_Algorithm/Function_based/TD3.py`                                 | Twin Delayed DDPG (deterministic, twin critics)                      |
| `source/CartPole/CartPole/tasks/`                                    | IsaacLab task definition (reward, termination, MDP)                  |
