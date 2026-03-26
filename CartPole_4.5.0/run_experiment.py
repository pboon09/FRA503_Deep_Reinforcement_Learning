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
        return SAC(
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
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


# ------------------------------------------------------------------ #
# Unified training loop (all algos use 256 parallel envs)              #
# ------------------------------------------------------------------ #

def train_algorithm(agent, env, algo_name, algo_cfg, shared_cfg, n_episodes,
                    num_envs, csv_writer, device):
    """Train any algorithm using 256 parallel envs."""
    max_steps = algo_cfg.get("max_steps", shared_cfg.get("max_steps", 200))
    global_step = 0

    # --- On-policy parallel algorithms (A2C, PPO) ---
    if algo_name in ("A2C", "PPO"):
        num_transitions = algo_cfg["num_transitions_per_env"]
        obs, _ = env.reset()
        obs = extract_obs(obs)
        n_obs = obs.shape[-1]
        action_type = algo_cfg.get("action_type", "continuous")
        actions_shape = (agent.num_of_action,) if action_type == "continuous" else (1,)
        agent._init_storage(num_envs, num_transitions, (n_obs,), actions_shape, device)

        for episode in tqdm(range(n_episodes), desc=f"Training {algo_name}", ncols=100):
            ep_return = 0.0

            with torch.inference_mode():
                for _ in range(num_transitions):
                    actions = agent.act(obs)
                    action_min, action_max = agent.action_range
                    if action_min is not None and action_type == "continuous":
                        env_actions = actions.clamp(action_min, action_max)
                    else:
                        env_actions = actions
                    next_obs, rewards, terminated, truncated, _ = env.step(env_actions)
                    next_obs = extract_obs(next_obs)
                    dones = terminated | truncated
                    agent.process_env_step(rewards, dones)
                    obs = next_obs
                    ep_return += rewards.mean().item()

                agent.compute_returns(obs)

            agent.update()
            global_step += num_envs * num_transitions

            csv_writer.writerow({
                "episode": episode,
                "ep_return": ep_return,
                "ep_length": num_transitions,
                "global_step": global_step,
                "epsilon": 0.0,
            })

    # --- All other algorithms (vectorized learn()) ---
    else:
        for episode in tqdm(range(n_episodes), desc=f"Training {algo_name}", ncols=100):
            if algo_name == "Linear_Q":
                ep_return, timestep = agent.learn(env, max_steps=max_steps,
                                                   num_agents=num_envs)
            elif algo_name == "DQN":
                ep_return, timestep = agent.learn(env, num_agents=num_envs,
                                                   max_steps=max_steps)
            elif algo_name == "MC_REINFORCE":
                result = agent.learn(env, num_agents=num_envs,
                                     max_steps=max_steps)
                ep_return = result[0]
                timestep = result[2] if len(result) > 2 and isinstance(result[2], (int, float)) else max_steps
            elif algo_name == "AC":
                result = agent.learn(env, max_steps=max_steps,
                                     num_agents=num_envs)
                ep_return = result[0]
                timestep = result[2] if len(result) > 2 else max_steps
            elif algo_name in ("SAC", "TD3"):
                ep_return, timestep = agent.learn(env, num_agents=num_envs,
                                                   max_steps=max_steps)

            global_step += num_envs * timestep
            epsilon = getattr(agent, 'epsilon', 0.0) or 0.0

            csv_writer.writerow({
                "episode": episode,
                "ep_return": ep_return,
                "ep_length": timestep,
                "global_step": global_step,
                "epsilon": epsilon,
            })

            # # Live plot (commented out)
            # agent.plot_durations(timestep)


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
                # Use GPU weights if available, else numpy
                if hasattr(agent, '_w_gpu') and agent._w_gpu is not None:
                    all_q = obs @ agent._w_gpu
                else:
                    obs_np = obs.cpu().numpy()
                    q_np = obs_np @ agent.w
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

    for algo_name in ALGO_ORDER:
        if algo_name not in config["algorithms"]:
            print(f"\nSkipping {algo_name} (not in config)")
            continue

        algo_cfg = config["algorithms"][algo_name]
        n_episodes = algo_cfg.get("n_episodes", 1000)
        print(f"\n{'='*60}")
        print(f"  Algorithm: {algo_name} ({num_envs} envs, {n_episodes} episodes)")
        print(f"{'='*60}")

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
                            n_episodes, num_envs, writer, device)

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

    # Close env only after ALL algorithms are done
    env.close()

    print(f"\n{'='*60}")
    print("  All experiments complete!")
    print(f"  Results: {exp_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
    simulation_app.close()
