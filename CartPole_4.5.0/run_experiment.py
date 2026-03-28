"""
Run all 8 function-based RL algorithms sequentially on the CartPole Stabilize task.
All algorithms use 256 parallel environments for fast training.

For each algorithm:
  1. Train for n_episodes iterations (with progress bar)
  2. Deploy (evaluate) for 10 episodes with deterministic policy (1 env)
  3. Save training CSV, deployment CSV, and final model

Usage (Isaac Lab):
    python run_experiment.py --task Stabilize-Isaac-Cartpole-v0 --headless
"""

import argparse
import sys
import os
import json
import csv
import time

from isaaclab.app import AppLauncher

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "RL_Algorithm"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "source", "CartPole"))

# ------------------------------------------------------------------ #
# CLI arguments                                                        #
# ------------------------------------------------------------------ #
parser = argparse.ArgumentParser(description="Run all RL experiments.")
parser.add_argument("--task", type=str, default="Stabilize-Isaac-Cartpole-v0",
                    help="Isaac Lab task name.")
parser.add_argument("--num_envs", type=int, default=256,
                    help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--config", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "scripts", "Function_based", "configs", "rl_config.json"),
                    help="Path to rl_config.json.")
parser.add_argument("--deploy_episodes", type=int, default=10,
                    help="Number of deployment (evaluation) episodes.")
parser.add_argument("--algo", type=str, default=None,
                    help="Run only this algorithm (e.g. --algo SAC). Runs all if omitted.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------------ #
# Imports that require the simulator to be running                     #
# ------------------------------------------------------------------ #
import gymnasium as gym
import torch
import numpy as np
import random
from tqdm import tqdm

from isaaclab.envs import (
    DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg,
    ManagerBasedRLEnvCfg, multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------------ #
# Algorithm imports                                                    #
# ------------------------------------------------------------------ #
from RL_Algorithm.Function_based.Linear_Q import Linear_QN
from RL_Algorithm.Function_based.DQN import DQN
from RL_Algorithm.Function_based.MC_REINFORCE import MC_REINFORCE
from RL_Algorithm.Function_based.AC import AC
from RL_Algorithm.Function_based.A2C import A2C
from RL_Algorithm.Function_based.PPO import PPO
from RL_Algorithm.Function_based.SAC import SAC
from RL_Algorithm.Function_based.TD3 import TD3

# ------------------------------------------------------------------ #
# Helpers                                                               #
# ------------------------------------------------------------------ #
ALGO_ORDER = ["Linear_Q", "DQN", "MC_REINFORCE", "AC", "A2C", "PPO", "SAC", "TD3"]

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def make_env(task: str, env_cfg, num_envs: int = 1):
    """Create and return an Isaac Lab environment."""
    env_cfg.scene.num_envs = num_envs
    env = gym.make(task, cfg=env_cfg)
    return env


def extract_obs(obs):
    """Extract observation tensor from Isaac Lab obs (handles dict or tensor)."""
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


def build_agent(algo_name: str, cfg: dict, shared: dict, device: torch.device):
    """Construct an agent from config."""
    action_range = shared["action_range"]
    n_obs = shared["n_observations"]
    gamma = shared["discount_factor"]
    ac = cfg

    if algo_name == "Linear_Q":
        return Linear_QN(
            num_of_action=ac["num_of_action"],
            action_range=action_range,
            learning_rate=ac["learning_rate"],
            initial_epsilon=ac["initial_epsilon"],
            epsilon_decay=ac["epsilon_decay"],
            final_epsilon=ac["final_epsilon"],
            discount_factor=gamma,
        )
    elif algo_name == "DQN":
        return DQN(
            device=device,
            num_of_action=ac["num_of_action"],
            action_range=action_range,
            n_observations=n_obs,
            hidden_dim=ac["hidden_dim"],
            dropout=ac.get("dropout", 0.0),
            learning_rate=ac["learning_rate"],
            tau=ac["tau"],
            initial_epsilon=ac["initial_epsilon"],
            epsilon_decay=ac["epsilon_decay"],
            final_epsilon=ac["final_epsilon"],
            discount_factor=gamma,
            buffer_size=ac["buffer_size"],
            batch_size=ac["batch_size"],
            learning_starts=ac.get("learning_starts", 1000),
        )
    elif algo_name == "MC_REINFORCE":
        return MC_REINFORCE(
            device=device,
            num_of_action=ac["num_of_action"],
            action_range=action_range,
            n_observations=n_obs,
            hidden_dim=ac["hidden_dim"],
            dropout=ac.get("dropout", 0.0),
            action_type=ac["action_type"],
            learning_rate=ac["learning_rate"],
            discount_factor=gamma,
        )
    elif algo_name == "AC":
        return AC(
            device=device,
            num_of_action=ac["num_of_action"],
            action_range=action_range,
            n_observations=n_obs,
            hidden_dims=ac["hidden_dims"],
            activation=ac.get("activation", "elu"),
            action_type=ac["action_type"],
            init_noise_std=ac.get("init_noise_std", 1.0),
            learning_rate=ac["learning_rate"],
            discount_factor=gamma,
            value_loss_coef=ac.get("value_loss_coef", 0.5),
            entropy_coef=ac.get("entropy_coef", 0.01),
            max_grad_norm=ac.get("max_grad_norm", 0.5),
        )
    elif algo_name == "A2C":
        return A2C(
            device=device,
            num_of_action=ac["num_of_action"],
            action_range=action_range,
            n_observations=n_obs,
            hidden_dims=ac["hidden_dims"],
            activation=ac.get("activation", "elu"),
            action_type=ac["action_type"],
            init_noise_std=ac.get("init_noise_std", 1.0),
            learning_rate=ac["learning_rate"],
            discount_factor=gamma,
            value_loss_coef=ac.get("value_loss_coef", 0.5),
            entropy_coef=ac.get("entropy_coef", 0.01),
            max_grad_norm=ac.get("max_grad_norm", 0.5),
        )
    elif algo_name == "PPO":
        return PPO(
            device=device,
            num_of_action=ac["num_of_action"],
            action_range=action_range,
            n_observations=n_obs,
            hidden_dims=ac["hidden_dims"],
            activation=ac.get("activation", "elu"),
            action_type=ac["action_type"],
            init_noise_std=ac.get("init_noise_std", 1.0),
            num_learning_epochs=ac["num_learning_epochs"],
            num_mini_batches=ac["num_mini_batches"],
            clip_param=ac["clip_param"],
            gamma=gamma,
            lam=ac["lam"],
            value_loss_coef=ac.get("value_loss_coef", 0.5),
            entropy_coef=ac.get("entropy_coef", 0.01),
            learning_rate=ac["learning_rate"],
            max_grad_norm=ac.get("max_grad_norm", 0.5),
            desired_kl=ac.get("desired_kl", 0.0),
        )
    elif algo_name == "SAC":
        agent = SAC(
            device=device,
            num_of_action=ac["num_of_action"],
            action_range=action_range,
            n_observations=n_obs,
            hidden_dim=ac["hidden_dim"],
            learning_rate=ac["learning_rate"],
            alpha_lr=ac.get("alpha_lr", 0.0003),
            tau=ac["tau"],
            discount_factor=gamma,
            buffer_size=ac["buffer_size"],
            batch_size=ac["batch_size"],
            init_alpha=ac.get("init_alpha", 0.2),
            auto_alpha=ac.get("auto_alpha", True),
            target_entropy=ac.get("target_entropy", None),
        )
        agent.learning_starts = ac.get("learning_starts", 0)
        return agent
    elif algo_name == "TD3":
        return TD3(
            device=device,
            num_of_action=ac["num_of_action"],
            action_range=action_range,
            n_observations=n_obs,
            hidden_dim=ac["hidden_dim"],
            learning_rate=ac["learning_rate"],
            tau=ac["tau"],
            discount_factor=gamma,
            buffer_size=ac["buffer_size"],
            batch_size=ac["batch_size"],
            exploration_noise=ac.get("exploration_noise", 0.1),
            target_noise=ac.get("target_noise", 0.2),
            target_noise_clip=ac.get("target_noise_clip", 0.5),
            policy_update_freq=ac.get("policy_update_freq", 2),
            learning_starts=ac.get("learning_starts", 1000),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


# ------------------------------------------------------------------ #
# Unified training loop (all algos use 256 parallel envs)              #
# ONE continuous loop - no env.reset() per iteration                   #
# ------------------------------------------------------------------ #

def _select_action_batch(agent, algo_name, obs, device):
    """Select actions for all 256 envs in one batch call."""
    if algo_name == "Linear_Q":
        # Initialise GPU tensors on first call
        if not (hasattr(agent, '_w_gpu') and agent._w_gpu is not None):
            agent._w_gpu = torch.tensor(agent.w, dtype=torch.float32, device=device)
            agent._device = device
        if not hasattr(agent, '_obs_scale_gpu') or agent._obs_scale_gpu is None:
            agent._obs_scale_gpu = torch.tensor(agent.obs_scale, dtype=torch.float32, device=device)
        obs_norm = obs / agent._obs_scale_gpu
        all_q = obs_norm @ agent._w_gpu
        greedy = all_q.argmax(dim=1)
        rand = torch.randint(0, agent.num_of_action, (obs.shape[0],), device=device)
        mask = torch.rand(obs.shape[0], device=device) < agent.epsilon
        indices = torch.where(mask, rand, greedy)
        a_min, a_max = agent.action_range
        action = a_min + (indices.float() / (agent.num_of_action - 1)) * (a_max - a_min)
        return action.unsqueeze(-1), indices

    elif algo_name == "DQN":
        with torch.no_grad():
            q = agent.policy_net(obs)
            greedy = q.argmax(dim=1)
        rand = torch.randint(0, agent.num_of_action, (obs.shape[0],), device=device)
        mask = torch.rand(obs.shape[0], device=device) < agent.epsilon
        indices = torch.where(mask, rand, greedy)
        a_min, a_max = agent.action_range
        action = a_min + (indices.float() / (agent.num_of_action - 1)) * (a_max - a_min)
        return action.unsqueeze(-1), indices

    elif algo_name == "MC_REINFORCE":
        dist = agent._get_distribution(obs)
        action, log_prob = agent._sample_action(dist)
        if agent.action_type == "continuous":
            env_action = action.clamp(agent.action_range[0], agent.action_range[1])
        else:
            env_action = agent._scale_action_batch(action)
        return env_action, (action, log_prob)

    elif algo_name == "AC":
        action = agent.policy.act(obs)
        value = agent.policy.evaluate(obs)
        log_prob = agent.policy.get_actions_log_prob(action)
        if agent.action_type == "continuous":
            env_action = action.clamp(agent.action_range[0], agent.action_range[1])
        else:
            env_action = action
        return env_action, (action, log_prob, value)

    elif algo_name == "SAC":
        return agent.select_action(obs), None

    elif algo_name == "TD3":
        return agent.select_action(obs), None

    return None, None


def train_algorithm(agent, env, algo_name, algo_cfg, shared_cfg, n_episodes_unused,
                    num_envs, csv_writer, device):
    """
    Train using 256 parallel envs. Budget = total_steps batch steps.
    All algorithms use the same budget for fair comparison.
    Each completed episode is logged to CSV.
    """
    total_steps = algo_cfg.get("total_steps", shared_cfg.get("total_steps", 20000))
    global_step = 0
    completed = 0

    obs, _ = env.reset()
    obs = extract_obs(obs)

    # Per-env episode tracking (shared by all algorithms)
    ep_returns = torch.zeros(num_envs, device=device)
    ep_lengths = torch.zeros(num_envs, device=device)

    # --- On-policy parallel algorithms (A2C, PPO) ---
    if algo_name in ("A2C", "PPO"):
        num_transitions = algo_cfg["num_transitions_per_env"]
        n_obs = obs.shape[-1]
        action_type = algo_cfg.get("action_type", "continuous")
        actions_shape = (agent.num_of_action,) if action_type == "continuous" else (1,)
        agent._init_storage(num_envs, num_transitions, (n_obs,), actions_shape, device)

        n_iters = total_steps // num_transitions
        pbar = tqdm(total=n_iters, desc=f"Training {algo_name}", ncols=100)

        for iteration in range(n_iters):
            with torch.no_grad():
                for t_step in range(num_transitions):
                    actions = agent.act(obs)
                    action_min, action_max = agent.action_range
                    if action_min is not None and action_type == "continuous":
                        env_actions = actions.clamp(action_min, action_max)
                    else:
                        env_actions = actions
                    next_obs, rewards, terminated, truncated, _info = env.step(env_actions)
                    next_obs = extract_obs(next_obs)
                    dones = terminated | truncated
                    agent.process_env_step(rewards, dones)
                    obs = next_obs

                    # Track per-env episodes
                    ep_returns += rewards
                    ep_lengths += 1
                    done_mask = dones.bool()
                    if done_mask.any():
                        done_idx = done_mask.nonzero(as_tuple=True)[0]
                        for idx in done_idx:
                            global_step_now = (iteration * num_transitions + t_step + 1) * num_envs
                            csv_writer.writerow({
                                "episode": completed,
                                "ep_return": ep_returns[idx].item(),
                                "ep_length": int(ep_lengths[idx].item()),
                                "global_step": global_step_now,
                                "epsilon": 0.0,
                            })
                            completed += 1
                        ep_returns[done_mask] = 0.0
                        ep_lengths[done_mask] = 0.0

                agent.compute_returns(obs)

            agent.update()
            pbar.update(1)

        pbar.close()

    # --- All other algorithms: single continuous loop ---
    else:
        # Per-env storage for MC algorithms (REINFORCE, AC)
        if algo_name in ("MC_REINFORCE", "AC"):
            env_obs = [[] for _ in range(num_envs)]
            env_actions = [[] for _ in range(num_envs)]
            env_rewards = [[] for _ in range(num_envs)]
            num_updates = 0
            update_every = 2  # very frequent updates for short episodes
            agent.optimizer.zero_grad()

        pbar = tqdm(total=total_steps, desc=f"Training {algo_name}", ncols=100)

        for step in range(total_steps):
            # --- Select actions ---
            env_action, extra = _select_action_batch(agent, algo_name, obs, device)

            # --- Step all 256 envs ---
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            next_obs = extract_obs(next_obs)
            done = (terminated | truncated).bool()

            # --- Algorithm-specific per-step logic ---
            if algo_name == "Linear_Q":
                # Vectorized TD update on GPU (with obs normalization)
                indices = extra  # action indices
                obs_norm      = obs      / agent._obs_scale_gpu
                next_obs_norm = next_obs / agent._obs_scale_gpu
                next_q = next_obs_norm @ agent._w_gpu
                next_max = next_q.max(dim=1).values
                cur_q = (obs_norm @ agent._w_gpu).gather(1, indices.unsqueeze(1)).squeeze(1)
                td_target = reward + agent.discount_factor * next_max * (~terminated).float()
                td_error = (td_target - cur_q).clamp(-5.0, 5.0)
                for a in range(agent.num_of_action):
                    mask = (indices == a)
                    if mask.any():
                        grads = agent.lr * td_error[mask].unsqueeze(1) * obs_norm[mask]
                        agent._w_gpu[:, a] += grads.mean(dim=0)
                agent.decay_epsilon()

            elif algo_name == "DQN":
                indices = extra
                rew_cpu = reward.cpu().numpy()
                term_cpu = terminated.cpu().numpy()
                done_cpu = done.cpu().numpy()
                for i in range(num_envs):
                    ns = None if done_cpu[i] else next_obs[i]
                    agent.store_transition(obs[i], int(indices[i].item()), float(rew_cpu[i]),
                                           ns, bool(term_cpu[i]))
                # Single gradient step (UTD=1) after learning_starts warmup
                if step >= agent.learning_starts:
                    agent.update_policy()
                    agent.update_target_networks()
                agent.decay_epsilon()

            elif algo_name == "MC_REINFORCE":
                action, _ = extra
                rew_cpu = reward.cpu().numpy()
                done_cpu = done.cpu().numpy()
                for i in range(num_envs):
                    env_obs[i].append(obs[i].detach())
                    env_actions[i].append(action[i].detach())
                    env_rewards[i].append(float(rew_cpu[i]))
                    if done_cpu[i] and len(env_rewards[i]) > 1:
                        # Recompute log_probs with fresh graph
                        ep_obs = torch.stack(env_obs[i])
                        ep_act = torch.stack(env_actions[i])
                        dist = agent._get_distribution(ep_obs)
                        if agent.action_type == "continuous":
                            lp = dist.log_prob(ep_act).sum(dim=-1)
                        else:
                            lp = dist.log_prob(ep_act.squeeze(-1))
                        returns = agent.calculate_stepwise_returns(env_rewards[i])
                        loss = agent.calculate_loss(returns, lp, ep_obs) / update_every
                        loss.backward()
                        num_updates += 1
                    if done_cpu[i]:
                        env_obs[i] = []
                        env_actions[i] = []
                        env_rewards[i] = []
                if num_updates >= update_every:
                    torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 0.5)
                    if hasattr(agent, 'value_net'):
                        torch.nn.utils.clip_grad_norm_(agent.value_net.parameters(), 0.5)
                    agent.optimizer.step()
                    agent.optimizer.zero_grad()
                    num_updates = 0

            elif algo_name == "AC":
                action, _, _ = extra
                done_cpu = done.cpu().numpy()
                rew_cpu = reward.cpu().numpy()
                for i in range(num_envs):
                    env_obs[i].append(obs[i].detach())
                    env_actions[i].append(action[i].detach())
                    env_rewards[i].append(float(rew_cpu[i]))
                    if done_cpu[i] and len(env_rewards[i]) > 1:
                        # Recompute log_probs & values with fresh graph
                        ep_obs = torch.stack(env_obs[i])
                        ep_act = torch.stack(env_actions[i])
                        agent.policy._update_distribution(ep_obs)
                        lp = agent.policy.get_actions_log_prob(ep_act)
                        vals = agent.policy.evaluate(ep_obs).squeeze(-1)
                        returns = agent.compute_returns(
                            torch.tensor(env_rewards[i], device=device))
                        al, cl = agent.calculate_loss(lp, vals, returns)
                        loss = (al + agent.value_loss_coef * cl) / update_every
                        loss.backward()
                        num_updates += 1
                    if done_cpu[i]:
                        env_obs[i] = []
                        env_actions[i] = []
                        env_rewards[i] = []
                if num_updates >= update_every:
                    torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), agent.max_grad_norm)
                    agent.optimizer.step()
                    agent.optimizer.zero_grad()
                    num_updates = 0

            elif algo_name == "SAC":
                a_min, a_max = agent.action_range
                raw = (env_action - a_min) / (a_max - a_min) * 2.0 - 1.0
                rew_cpu = reward.cpu().numpy()
                done_cpu = done.float().cpu().numpy()
                for i in range(num_envs):
                    agent.store_transition(obs[i], raw[i], float(rew_cpu[i]),
                                           next_obs[i], float(done_cpu[i]))
                # SAC: UTD=1 after learning_starts warmup
                if step >= getattr(agent, 'learning_starts', 0):
                    agent.update_policy()

            elif algo_name == "TD3":
                a_min, a_max = agent.action_range
                raw = (env_action - a_min) / (a_max - a_min) * 2.0 - 1.0
                rew_cpu = reward.cpu().numpy()
                done_cpu = done.float().cpu().numpy()
                for i in range(num_envs):
                    agent.store_transition(obs[i], raw[i], float(rew_cpu[i]),
                                           next_obs[i], float(done_cpu[i]))
                # TD3: UTD=1 after learning_starts (deterministic actor is fragile at high UTD)
                if step >= agent.learning_starts:
                    agent.update_policy()

            # --- Track completed episodes ---
            ep_returns += reward
            ep_lengths += 1
            global_step += num_envs

            if done.any():
                done_idx = done.nonzero(as_tuple=True)[0]
                for idx in done_idx:
                    csv_writer.writerow({
                        "episode": completed,
                        "ep_return": ep_returns[idx].item(),
                        "ep_length": int(ep_lengths[idx].item()),
                        "global_step": global_step,
                        "epsilon": getattr(agent, 'epsilon', 0.0) or 0.0,
                    })
                    completed += 1
                ep_returns[done] = 0.0
                ep_lengths[done] = 0.0

            obs = next_obs
            pbar.update(1)

        pbar.close()

        # Final gradient step for MC algorithms (flush accumulated grads)
        if algo_name in ("MC_REINFORCE", "AC") and num_updates > 0:
            if algo_name == "MC_REINFORCE":
                torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 0.5)
                if hasattr(agent, 'value_net'):
                    torch.nn.utils.clip_grad_norm_(agent.value_net.parameters(), 0.5)
            elif algo_name == "AC":
                torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), agent.max_grad_norm)
            agent.optimizer.step()
            agent.optimizer.zero_grad()

        # Sync Linear_Q GPU weights back to numpy
        if algo_name == "Linear_Q":
            agent.w = agent._w_gpu.cpu().numpy()


def deploy(agent, env, algo_name, n_episodes, max_steps, csv_writer, device):
    """Deploy (evaluate) using the same 256-env. Track env[0] for episode returns."""
    print(f"\n  Deploying {algo_name} for {n_episodes} episodes...")

    obs, _ = env.reset()
    obs = extract_obs(obs)
    num_envs = obs.shape[0]
    ep_returns = torch.zeros(num_envs, device=device)
    ep_lengths = torch.zeros(num_envs, device=device)
    completed = 0

    while completed < n_episodes:
        with torch.no_grad():
            if algo_name == "Linear_Q":
                # Use GPU weights if available, else numpy (with obs normalization)
                if hasattr(agent, '_w_gpu') and agent._w_gpu is not None:
                    if not hasattr(agent, '_obs_scale_gpu') or agent._obs_scale_gpu is None:
                        agent._obs_scale_gpu = torch.tensor(agent.obs_scale, dtype=torch.float32, device=device)
                    all_q = (obs / agent._obs_scale_gpu) @ agent._w_gpu
                else:
                    obs_np = obs.cpu().numpy()
                    q_np = (obs_np / agent.obs_scale) @ agent.w
                    all_q = torch.tensor(q_np, device=device)
                action_indices = all_q.argmax(dim=1)
                action_min, action_max = agent.action_range
                action = action_min + (action_indices.float() / (agent.num_of_action - 1)) * (action_max - action_min)
                action = action.unsqueeze(-1)
            elif algo_name == "DQN":
                q_vals = agent.policy_net(obs)
                action_indices = q_vals.argmax(dim=1)
                action_min, action_max = agent.action_range
                action = action_min + (action_indices.float() / (agent.num_of_action - 1)) * (action_max - action_min)
                action = action.unsqueeze(-1)
            elif algo_name == "MC_REINFORCE":
                if agent.action_type == "continuous":
                    mean = agent.policy_net(obs)
                    action = mean.clamp(agent.action_range[0], agent.action_range[1])
                else:
                    logits = agent.policy_net(obs)
                    action_indices = logits.argmax(dim=-1)
                    action_min, action_max = agent.action_range
                    action = action_min + (action_indices.float() / (agent.num_of_action - 1)) * (action_max - action_min)
                    action = action.unsqueeze(-1)
            elif algo_name in ("AC", "A2C", "PPO"):
                action = agent.select_action(obs)
                if hasattr(agent, 'action_type') and agent.action_type == "continuous":
                    action = action.clamp(agent.action_range[0], agent.action_range[1])
                if action.dim() == 1:
                    action = action.unsqueeze(-1)
            elif algo_name == "SAC":
                action = agent.select_action(obs, evaluate=True)
            elif algo_name == "TD3":
                action = agent.select_action(obs, add_noise=False)

            # Ensure (num_envs, action_dim)
            if action.dim() == 1:
                action = action.unsqueeze(-1)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = extract_obs(next_obs)
            done = (terminated | truncated).bool()

            ep_returns += reward
            ep_lengths += 1

            # Log completed episodes
            if done.any():
                done_idx = done.nonzero(as_tuple=True)[0]
                for idx in done_idx:
                    if completed < n_episodes:
                        ret = ep_returns[idx].item()
                        length = int(ep_lengths[idx].item())
                        csv_writer.writerow({
                            "episode": completed,
                            "ep_return": ret,
                            "ep_length": length,
                        })
                        print(f"    Episode {completed}: return={ret:.1f}, length={length}")
                        completed += 1
                ep_returns[done] = 0.0
                ep_lengths[done] = 0.0

            obs = next_obs


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlOnPolicyRunnerCfg):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    config = load_config(args_cli.config)
    shared = config["shared"]
    num_envs = shared.get("num_envs", args_cli.num_envs)
    task_name = str(args_cli.task).split("-")[0]

    # Output directories
    exp_dir = os.path.join(ROOT, "experiments", "suite_1_baseline")
    model_base = os.path.join(ROOT, "model", task_name)
    os.makedirs(exp_dir, exist_ok=True)

    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Create ONE environment and reuse for all algorithms (don't close/recreate)
    env = make_env(args_cli.task, env_cfg, num_envs=num_envs)

    algo_filter = [args_cli.algo] if args_cli.algo else ALGO_ORDER
    for algo_name in algo_filter:
        if algo_name not in config["algorithms"]:
            print(f"\nSkipping {algo_name} (not in config)")
            continue

        algo_cfg = config["algorithms"][algo_name]
        t_steps = algo_cfg.get("total_steps", shared.get("total_steps", 20000))
        print(f"\n{'='*60}")
        print(f"  Algorithm: {algo_name} ({num_envs} envs, {t_steps} batch steps)")
        print(f"{'='*60}")

        try:
            # Build agent
            agent = build_agent(algo_name, algo_cfg, shared, device)

            # Model save directory
            model_dir = os.path.join(model_base, algo_name)
            os.makedirs(model_dir, exist_ok=True)

            # ---- Train ---- #
            train_csv_path = os.path.join(exp_dir, f"{algo_name}.csv")
            with open(train_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "episode", "ep_return", "ep_length", "global_step", "epsilon"])
                writer.writeheader()
                train_algorithm(agent, env, algo_name, algo_cfg, shared,
                                t_steps, num_envs, writer, device)

            # Save final model
            if algo_name == "Linear_Q":
                agent.save_model(model_dir, f"{algo_name}_final.npy")
            else:
                agent.save_model(model_dir, f"{algo_name}_final.pth")

            print(f"  Training complete. Model saved to {model_dir}")

            # ---- Deploy (reuse same env, don't close) ---- #
            deploy_csv_path = os.path.join(exp_dir, f"{algo_name}_deploy.csv")
            with open(deploy_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["episode", "ep_return", "ep_length"])
                writer.writeheader()
                deploy(agent, env, algo_name, args_cli.deploy_episodes, 1000, writer, device)

            print(f"  Deployment complete. Results saved to {deploy_csv_path}")

            # Save training summary
            summary_path = os.path.join(exp_dir, f"{algo_name}_summary.txt")
            with open(summary_path, "w") as f:
                f.write(f"Algorithm: {algo_name}\n")
                f.write(f"Task: {args_cli.task}\n")
                f.write(f"Num Envs: {num_envs}\n")
                f.write(f"Episodes: {algo_cfg.get('n_episodes', 1000)}\n")
                f.write(f"Config: {json.dumps(algo_cfg, indent=2)}\n")
                f.write(f"Shared: {json.dumps(shared, indent=2)}\n")

        except Exception as e:
            import traceback
            print(f"\n  ERROR in {algo_name}:")
            traceback.print_exc()
            print(f"  Skipping {algo_name}, continuing with next algorithm...\n")

    # Close env only after ALL algorithms are done
    env.close()

    print(f"\n{'='*60}")
    print("  All experiments complete!")
    print(f"  Results: {exp_dir}")
    print(f"{'='*60}")

    # Auto-generate visualization figures
    print("\n  Generating figures...")
    try:
        import subprocess
        vis_script = os.path.join(ROOT, "scripts", "Function_based", "visualize", "plot_report_figures.py")
        subprocess.run([sys.executable, vis_script], cwd=ROOT, check=True)
        print("  Figures saved to figures/")
    except Exception as e:
        print(f"  Warning: Failed to generate figures: {e}")
        print(f"  Run manually: python {vis_script}")


if __name__ == "__main__":
    main()
    simulation_app.close()
