"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import csv
import time
import json

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../RL_Algorithm")))

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
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
    # ========================= Can be modified ========================== #

    Algorithm_name = _ALGORITHM_NAME

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print("device:", device)

    task_name = str(args_cli.task).split("-")[0]
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1

    config_path = os.path.join(os.path.dirname(__file__), "configs", "rl_config.json")
    with open(config_path, "r") as f:
        full_cfg = json.load(f)

    shared = full_cfg["shared"]
    algo_cfg = full_cfg["algorithms"].get(Algorithm_name, {})

    n_episodes = shared["n_episodes"]
    n_observations = shared["n_observations"]

    if Algorithm_name == "Linear_Q":
        from RL_Algorithm.Function_based.Linear_Q import Linear_QN as Algorithm
        agent = Algorithm(
            num_of_action=algo_cfg["num_of_action"],
            action_range=shared["action_range"],
            learning_rate=algo_cfg["learning_rate"],
            initial_epsilon=algo_cfg["initial_epsilon"],
            epsilon_decay=algo_cfg["epsilon_decay"],
            final_epsilon=algo_cfg["final_epsilon"],
            discount_factor=shared["discount_factor"],
        )
    elif Algorithm_name == "DQN":
        from RL_Algorithm.Function_based.DQN import DQN as Algorithm
        agent = Algorithm(
            device=device,
            num_of_action=algo_cfg["num_of_action"],
            action_range=shared["action_range"],
            n_observations=n_observations,
            hidden_dim=algo_cfg["hidden_dim"],
            dropout=algo_cfg.get("dropout", 0.0),
            learning_rate=algo_cfg["learning_rate"],
            tau=algo_cfg["tau"],
            initial_epsilon=algo_cfg["initial_epsilon"],
            epsilon_decay=algo_cfg["epsilon_decay"],
            final_epsilon=algo_cfg["final_epsilon"],
            discount_factor=shared["discount_factor"],
            buffer_size=algo_cfg["buffer_size"],
            batch_size=algo_cfg["batch_size"],
        )
    elif Algorithm_name == "MC_REINFORCE":
        from RL_Algorithm.Function_based.MC_REINFORCE import MC_REINFORCE as Algorithm
        agent = Algorithm(
            device=device,
            num_of_action=algo_cfg.get("num_of_action", 1),
            action_range=shared["action_range"],
            n_observations=n_observations,
            hidden_dim=algo_cfg["hidden_dim"],
            dropout=algo_cfg.get("dropout", 0.1),
            action_type=algo_cfg.get("action_type", "continuous"),
            learning_rate=algo_cfg["learning_rate"],
            discount_factor=shared["discount_factor"],
        )
    elif Algorithm_name == "AC":
        from RL_Algorithm.Function_based.AC import AC as Algorithm
        agent = Algorithm(
            device=device,
            num_of_action=algo_cfg.get("num_of_action", 1),
            action_range=shared["action_range"],
            n_observations=n_observations,
            hidden_dims=algo_cfg["hidden_dims"],
            activation=algo_cfg.get("activation", "elu"),
            action_type=algo_cfg.get("action_type", "continuous"),
            init_noise_std=algo_cfg.get("init_noise_std", 1.0),
            learning_rate=algo_cfg["learning_rate"],
            discount_factor=shared["discount_factor"],
            value_loss_coef=algo_cfg.get("value_loss_coef", 0.5),
            entropy_coef=algo_cfg.get("entropy_coef", 0.01),
            max_grad_norm=algo_cfg.get("max_grad_norm", 0.5),
        )
    elif Algorithm_name == "PPO":
        from RL_Algorithm.Function_based.PPO import PPO as Algorithm
        agent = Algorithm(
            device=device,
            num_of_action=algo_cfg.get("num_of_action", 1),
            action_range=shared["action_range"],
            n_observations=n_observations,
            hidden_dims=algo_cfg["hidden_dims"],
            activation=algo_cfg.get("activation", "elu"),
            action_type=algo_cfg.get("action_type", "continuous"),
            init_noise_std=algo_cfg.get("init_noise_std", 1.0),
            num_learning_epochs=algo_cfg["num_learning_epochs"],
            num_mini_batches=algo_cfg["num_mini_batches"],
            clip_param=algo_cfg["clip_param"],
            gamma=shared["discount_factor"],
            lam=algo_cfg["lam"],
            value_loss_coef=algo_cfg.get("value_loss_coef", 1.0),
            entropy_coef=algo_cfg.get("entropy_coef", 0.005),
            learning_rate=algo_cfg["learning_rate"],
            max_grad_norm=algo_cfg.get("max_grad_norm", 1.0),
            desired_kl=algo_cfg.get("desired_kl", 0.01),
        )
    else:
        raise ValueError(f"Unknown algorithm: {Algorithm_name}")

    save_interval = 5000
    model_dir = os.path.join("model", task_name, Algorithm_name)
    os.makedirs(model_dir, exist_ok=True)

    csv_dir = os.path.join("logs", task_name, Algorithm_name)
    os.makedirs(csv_dir, exist_ok=True)
    csv_filename = os.path.join(csv_dir, f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    csv_file = open(csv_filename, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=["episode", "ep_length", "ep_return", "loss", "epsilon", "lr"])
    csv_writer.writeheader()

    obs, _ = env.reset()
    timestep = 0

    while simulation_app.is_running():

        if Algorithm_name == "PPO":
            num_transitions_per_env = algo_cfg.get("num_transitions_per_env", 16)
            agent.learn(env, num_envs=num_envs, num_transitions_per_env=num_transitions_per_env, max_episodes=n_episodes)
            agent.save_model(model_dir, f"{Algorithm_name}_final.pth")
            print("Training complete.")

        else:
            for episode in tqdm(range(n_episodes)):

                if Algorithm_name == "Linear_Q":
                    ep_return, ep_len = agent.learn(env, max_steps=1000)
                    loss_val = 0.0
                elif Algorithm_name == "DQN":
                    ep_return, ep_len = agent.learn(env, num_agents=num_envs, max_steps=1000)
                    loss_val = 0.0
                elif Algorithm_name == "MC_REINFORCE":
                    ep_return, loss_val, _ = agent.learn(env, num_agents=num_envs)
                    ep_len = 0
                elif Algorithm_name == "AC":
                    ep_return, loss_val, ep_len = agent.learn(env, max_steps=1000, num_agents=num_envs)
                else:
                    break

                agent.plot_durations(timestep=ep_len)

                csv_writer.writerow({
                    "episode": episode,
                    "ep_length": ep_len,
                    "ep_return": ep_return,
                    "loss": loss_val,
                    "epsilon": getattr(agent, 'epsilon', 0.0),
                    "lr": getattr(agent, 'lr', getattr(agent, 'LR', 0.0)),
                })

                if episode % 100 == 0:
                    print(f"[{Algorithm_name}] episode {episode} | return={ep_return:.2f}")

                if episode % save_interval == 0 and episode > 0:
                    ext = ".npy" if Algorithm_name == "Linear_Q" else ".pth"
                    agent.save_model(model_dir, f"{Algorithm_name}_{episode}{ext}")

            ext = ".npy" if Algorithm_name == "Linear_Q" else ".pth"
            agent.save_model(model_dir, f"{Algorithm_name}_final{ext}")
            print("Training complete.")

        csv_file.close()

        agent.plot_durations(show_result=True)
        plt.ioff()
        plt.show()

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        break
    # ==================================================================== #

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
