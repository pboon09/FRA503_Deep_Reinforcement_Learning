"""Script to play a trained RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import sys
import os
import json

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../RL_Algorithm")))

parser = argparse.ArgumentParser(description="Play a trained RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
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
    n_episodes = 10

    config_path = os.path.join(os.path.dirname(__file__), "configs", "rl_config.json")
    with open(config_path, "r") as f:
        full_cfg = json.load(f)

    shared = full_cfg["shared"]
    algo_cfg = full_cfg["algorithms"].get(Algorithm_name, {})
    n_observations = shared["n_observations"]

    if Algorithm_name == "Linear_Q":
        from RL_Algorithm.Function_based.Linear_Q import Linear_QN
        agent = Linear_QN(
            num_of_action=algo_cfg["num_of_action"],
            action_range=shared["action_range"],
        )
    elif Algorithm_name == "DQN":
        from RL_Algorithm.Function_based.DQN import DQN
        agent = DQN(
            device=device,
            num_of_action=algo_cfg["num_of_action"],
            action_range=shared["action_range"],
            n_observations=n_observations,
            hidden_dim=algo_cfg["hidden_dim"],
            dropout=algo_cfg.get("dropout", 0.0),
            learning_rate=algo_cfg["learning_rate"],
            tau=algo_cfg["tau"],
            initial_epsilon=0.0,
            epsilon_decay=0.0,
            final_epsilon=0.0,
            discount_factor=shared["discount_factor"],
            buffer_size=algo_cfg["buffer_size"],
            batch_size=algo_cfg["batch_size"],
        )
    elif Algorithm_name == "MC_REINFORCE":
        from RL_Algorithm.Function_based.MC_REINFORCE import MC_REINFORCE
        agent = MC_REINFORCE(
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
        from RL_Algorithm.Function_based.AC import AC
        agent = AC(
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
        from RL_Algorithm.Function_based.PPO import PPO
        agent = PPO(
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
            learning_rate=algo_cfg["learning_rate"],
        )
    else:
        raise ValueError(f"Unknown algorithm: {Algorithm_name}")

    model_dir = os.path.join("model", task_name, Algorithm_name)
    ext = ".npy" if Algorithm_name == "Linear_Q" else ".pth"
    model_filename = f"{Algorithm_name}_final{ext}"
    agent.load_model(model_dir, model_filename)
    print(f"Loaded: {os.path.join(model_dir, model_filename)}")

    obs, _ = env.reset()
    timestep = 0

    # Deployment CSV
    exp_dir = os.path.join("experiments", "suite_1_baseline")
    os.makedirs(exp_dir, exist_ok=True)
    deploy_csv_path = os.path.join(exp_dir, f"{Algorithm_name}_deploy.csv")
    deploy_rows = []

    while simulation_app.is_running():
        with torch.inference_mode():

            for episode in range(n_episodes):
                obs, _ = env.reset()
                done = False
                ep_return = 0.0
                ep_len = 0

                while not done:
                    state = obs['policy'].to(device)

                    if Algorithm_name == "Linear_Q":
                        action, _ = agent.select_action(state)
                    elif Algorithm_name == "DQN":
                        agent.epsilon = 0.0
                        action, _ = agent.select_action(state)
                    elif Algorithm_name in ("MC_REINFORCE",):
                        agent.policy_net.eval()
                        if agent.action_type == "continuous":
                            action = torch.clamp(
                                agent.policy_net(state), shared["action_range"][0], shared["action_range"][1]
                            )
                        else:
                            logits = agent.policy_net(state)
                            action = agent.scale_action(logits.argmax(dim=-1).item())
                    elif Algorithm_name in ("AC", "PPO"):
                        action = agent.select_action(state)
                        if algo_cfg.get("action_type", "continuous") == "continuous":
                            action = torch.clamp(action, shared["action_range"][0], shared["action_range"][1])
                    else:
                        break

                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = (terminated | truncated).any().item()
                    ep_return += reward.mean().item()
                    ep_len += 1

                deploy_rows.append({
                    "algorithm": Algorithm_name,
                    "episode": episode,
                    "ep_return": ep_return,
                    "ep_length": ep_len,
                })
                print(f"[{Algorithm_name}] Episode {episode}: return={ep_return:.2f}, length={ep_len}")

        # Save deployment CSV
        with open(deploy_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["algorithm", "episode", "ep_return", "ep_length"])
            writer.writeheader()
            writer.writerows(deploy_rows)
        print(f"Saved deployment CSV to {deploy_csv_path}")

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
