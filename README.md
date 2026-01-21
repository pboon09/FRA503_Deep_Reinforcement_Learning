# HW0: Cartpole RL Agent

FRA503 Deep Reinforcement Learning for Robotics

## Files to Fix

This homework involves modifying files in the **IsaacLab** installation (located at `~/IsaacLab/`):

| File | Purpose |
|------|---------|
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_env_cfg.py` | Environment configuration (rewards, terminations, observations) |
| `source/isaaclab_tasks/.../cartpole/agents/sb3_ppo_cfg.yaml` | PPO hyperparameter configuration |
| `source/isaaclab/isaaclab/managers/reward_manager.py` | Change how Isaaclab calculates reward |

## Files Involved in This Homework

### IsaacLab Structure

| Folder/File | Purpose |
|-------------|---------|
| `source/` | Core library code: environments, tasks, robot assets, managers |
| `scripts/` | Training and evaluation scripts for RL libraries (sb3, rsl_rl, skrl) |
| `logs/` | Training outputs: checkpoints, tensorboard logs |
| `isaaclab.sh` | Launcher script for Linux |

### Key Configuration Files

| File | Description |
|------|-------------|
| `cartpole_env_cfg.py` | Defines `CartpoleEnvCfg`, `CartpoleSceneCfg`, `ActionsCfg`, `ObservationsCfg`, `RewardsCfg`, `TerminationsCfg`, `EventCfg` |
| `sb3_ppo_cfg.yaml` | PPO hyperparameters: `n_steps`, `batch_size`, `n_epochs`, `learning_rate` |
| `reward_manager.py` | Handles reward computation (time-based vs step-based) |

## Commands Used

### Environment Setup

```bash
# Activate conda environment
conda activate env_isaaclab

# Navigate to IsaacLab directory
cd ~/IsaacLab
```

### Training

```bash
# Train Cartpole with 64 parallel environments (headless for faster training)
python scripts/reinforcement_learning/sb3/train.py --task=Isaac-Cartpole-v0 --num_envs 64 --headless

# Train with GUI (remove --headless)
python scripts/reinforcement_learning/sb3/train.py --task=Isaac-Cartpole-v0 --num_envs 64
```

### Play Trained Policy

```bash
python scripts/reinforcement_learning/sb3/play.py --task=Isaac-Cartpole-v0 --num_envs 64
```

### TensorBoard Visualization

```bash
python -m tensorboard.main --logdir logs/sb3/Isaac-Cartpole-v0
```

### View Training Results

```bash
# Check trained model location
ls logs/sb3/Isaac-Cartpole-v0/

# View the latest training folder
ls -la logs/sb3/Isaac-Cartpole-v0/$(ls -t logs/sb3/Isaac-Cartpole-v0/ | head -1)/
```

### Delete Training Logs

```bash
rm -rf logs/sb3/Isaac-Cartpole-v0/
```

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | None | Environment name (e.g., `Isaac-Cartpole-v0`) |
| `--num_envs` | From config | Number of parallel environments |
| `--headless` | False | Run without GUI for faster training |
| `--seed` | From config | Random seed for reproducibility |
| `--max_iterations` | None | Number of policy updates |
