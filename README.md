# FRA503 Deep Reinforcement Learning for Robotics

## HW2 — Tabular RL Training

```bash
cd ~/FRA503_Deep_Reinforcement_Learning/CartPole_4.5.0
conda activate env_isaaclab
```

### Train

```bash
# Monte Carlo
RL_ALGORITHM=MC python scripts/RL_Algorithm/train.py --task Stabilize-Isaac-Cartpole-v0 --num_envs 256 --headless

# SARSA
RL_ALGORITHM=SARSA python scripts/RL_Algorithm/train.py --task Stabilize-Isaac-Cartpole-v0 --num_envs 256 --headless

# Q-Learning
RL_ALGORITHM=Q_Learning python scripts/RL_Algorithm/train.py --task Stabilize-Isaac-Cartpole-v0 --num_envs 256 --headless

# Double Q-Learning
RL_ALGORITHM=Double_Q_Learning python scripts/RL_Algorithm/train.py --task Stabilize-Isaac-Cartpole-v0 --num_envs 256 --headless
```

### TensorBoard

```bash
tensorboard --logdir logs/Stabilize/
```

### Visualize CSV Logs

```bash
# Single algorithm
python scripts/RL_Algorithm/visualize/plot_training.py \
    --logs "logs/Stabilize/MC/training_log_*.csv" \
    --output figures/

# Compare all algorithms
python scripts/RL_Algorithm/visualize/plot_training.py \
    --logs "logs/Stabilize/MC/training_log_*.csv" \
           "logs/Stabilize/SARSA/training_log_*.csv" \
           "logs/Stabilize/Q_Learning/training_log_*.csv" \
           "logs/Stabilize/Double_Q_Learning/training_log_*.csv" \
    --output figures/ --show
```