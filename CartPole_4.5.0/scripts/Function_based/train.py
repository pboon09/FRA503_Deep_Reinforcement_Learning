"""Script to train RL agent."""
"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import json

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../RL_Algorithm")))

parser = argparse.ArgumentParser(description="Train an RL agent.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import gymnasium as gym
import torch
import random
import matplotlib.pyplot as plt
from datetime import datetime

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

_ALGORITHM_NAME = os.environ.get("RL_ALGORITHM", "PPO")


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    Algorithm_name = _ALGORITHM_NAME
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    task_name = str(args_cli.task).split("-")[0]
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1

    config_path = os.path.join(os.path.dirname(__file__), "configs", "rl_config.json")
    with open(config_path, "r") as f:
        full_cfg = json.load(f)
    shared = full_cfg["shared"]
    algo_cfg = full_cfg["algorithms"].get(Algorithm_name, {})
    n_episodes = shared["n_episodes"]

    if Algorithm_name == "Linear_Q":
        from RL_Algorithm.Function_based.Linear_Q import Linear_QN
        agent = Linear_QN(
            num_of_action=algo_cfg["num_of_action"], action_range=shared["action_range"],
            learning_rate=algo_cfg["learning_rate"], initial_epsilon=algo_cfg["initial_epsilon"],
            epsilon_decay=algo_cfg["epsilon_decay"], final_epsilon=algo_cfg["final_epsilon"],
            discount_factor=shared["discount_factor"],
        )
    elif Algorithm_name == "DQN":
        from RL_Algorithm.Function_based.DQN import DQN
        agent = DQN(
            device=device, num_of_action=algo_cfg["num_of_action"], action_range=shared["action_range"],
            n_observations=shared["n_observations"], hidden_dim=algo_cfg["hidden_dim"],
            dropout=algo_cfg.get("dropout", 0.0), learning_rate=algo_cfg["learning_rate"],
            tau=algo_cfg["tau"], initial_epsilon=algo_cfg["initial_epsilon"],
            epsilon_decay=algo_cfg["epsilon_decay"], final_epsilon=algo_cfg["final_epsilon"],
            discount_factor=shared["discount_factor"], buffer_size=algo_cfg["buffer_size"],
            batch_size=algo_cfg["batch_size"],
        )
    elif Algorithm_name == "MC_REINFORCE":
        from RL_Algorithm.Function_based.MC_REINFORCE import MC_REINFORCE
        agent = MC_REINFORCE(
            device=device, num_of_action=algo_cfg.get("num_of_action", 1),
            action_range=shared["action_range"], n_observations=shared["n_observations"],
            hidden_dim=algo_cfg["hidden_dim"], dropout=algo_cfg.get("dropout", 0.1),
            action_type=algo_cfg.get("action_type", "continuous"),
            learning_rate=algo_cfg["learning_rate"], discount_factor=shared["discount_factor"],
        )
    elif Algorithm_name == "AC":
        from RL_Algorithm.Function_based.AC import AC
        agent = AC(
            device=device, num_of_action=algo_cfg.get("num_of_action", 1),
            action_range=shared["action_range"], n_observations=shared["n_observations"],
            hidden_dims=algo_cfg["hidden_dims"], activation=algo_cfg.get("activation", "elu"),
            action_type=algo_cfg.get("action_type", "continuous"),
            init_noise_std=algo_cfg.get("init_noise_std", 1.0),
            learning_rate=algo_cfg["learning_rate"], discount_factor=shared["discount_factor"],
            value_loss_coef=algo_cfg.get("value_loss_coef", 0.5),
            entropy_coef=algo_cfg.get("entropy_coef", 0.01),
            max_grad_norm=algo_cfg.get("max_grad_norm", 0.5),
        )
    elif Algorithm_name == "PPO":
        from RL_Algorithm.Function_based.PPO import PPO
        agent = PPO(
            device=device, num_of_action=algo_cfg.get("num_of_action", 1),
            action_range=shared["action_range"], n_observations=shared["n_observations"],
            hidden_dims=algo_cfg["hidden_dims"], activation=algo_cfg.get("activation", "elu"),
            action_type=algo_cfg.get("action_type", "continuous"),
            init_noise_std=algo_cfg.get("init_noise_std", 1.0),
            num_learning_epochs=algo_cfg["num_learning_epochs"],
            num_mini_batches=algo_cfg["num_mini_batches"],
            clip_param=algo_cfg["clip_param"], gamma=shared["discount_factor"],
            lam=algo_cfg["lam"], value_loss_coef=algo_cfg.get("value_loss_coef", 1.0),
            entropy_coef=algo_cfg.get("entropy_coef", 0.005),
            learning_rate=algo_cfg["learning_rate"],
            max_grad_norm=algo_cfg.get("max_grad_norm", 1.0),
            desired_kl=algo_cfg.get("desired_kl", 0.01),
        )
    else:
        raise ValueError(f"Unknown algorithm: {Algorithm_name}")

    model_dir = os.path.join("model", task_name, Algorithm_name)
    os.makedirs(model_dir, exist_ok=True)

    while simulation_app.is_running():
        if Algorithm_name == "PPO":
            num_transitions_per_env = algo_cfg.get("num_transitions_per_env", 16)
            ppo_iters = algo_cfg.get("max_iterations", 1000)
            agent.learn(env, num_envs=num_envs, num_transitions_per_env=num_transitions_per_env, max_episodes=ppo_iters)
        elif Algorithm_name == "Linear_Q":
            agent.learn(env, num_agents=num_envs, n_episodes=n_episodes)
        elif Algorithm_name == "DQN":
            agent.learn(env, num_agents=num_envs, n_episodes=n_episodes)
        elif Algorithm_name == "MC_REINFORCE":
            agent.learn(env, num_agents=num_envs, n_episodes=n_episodes)
        elif Algorithm_name == "AC":
            agent.learn(env, num_agents=num_envs, n_episodes=n_episodes)

        ext = ".npy" if Algorithm_name == "Linear_Q" else ".pth"
        agent.save_model(model_dir, f"{Algorithm_name}_final{ext}")
        print("Training complete.")

        # Save training curve plot to figures/
        fig_dir = os.path.join("figures")
        os.makedirs(fig_dir, exist_ok=True)
        agent.plot_durations(show_result=True)
        plt.savefig(os.path.join(fig_dir, f"{Algorithm_name}_training_curve.png"), dpi=150)
        print(f"Saved training curve to {fig_dir}/{Algorithm_name}_training_curve.png")
        plt.close('all')

        # Save per-episode CSV to experiments/ (like HW2)
        exp_dir = os.path.join("experiments", "suite_1_baseline")
        os.makedirs(exp_dir, exist_ok=True)
        csv_path = os.path.join(exp_dir, f"{Algorithm_name}.csv")
        if agent.episode_log:
            fieldnames = list(agent.episode_log[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(agent.episode_log)
        print(f"Saved CSV ({len(agent.episode_log)} episodes) to {csv_path}")

        break
    # ==================================================================== #
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
