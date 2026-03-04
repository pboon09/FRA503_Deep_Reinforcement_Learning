"""Script to evaluate a trained RL agent (deployment mode, epsilon=0)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import json

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a trained RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--video_dir", type=str, default="videos", help="Directory to save recorded videos.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--algorithm", type=str, default="Q_Learning",
                    choices=["MC", "SARSA", "Q_Learning", "Double_Q_Learning"],
                    help="Algorithm class to instantiate.")
parser.add_argument("--qtable_path", type=str, required=True,
                    help="Full path to Q-table JSON file.")
parser.add_argument("--num_episodes", type=int, default=10,
                    help="Number of evaluation episodes.")
parser.add_argument("--output_csv", type=str, default="evaluation_results.csv",
                    help="Path for evaluation results CSV (appended to).")
parser.add_argument("--trajectory_dir", type=str, default=None,
                    help="Directory to save per-step trajectory CSVs for analysis plots.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import gymnasium as gym
import torch
import numpy as np

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_tasks.utils import parse_env_cfg

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Evaluate a trained RL agent with pure exploitation (epsilon=0)."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": args_cli.video_dir,
            "episode_trigger": lambda episode_id: True,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording videos to {args_cli.video_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # Load hyperparameters from rl_config.json
    config_path = os.path.join(os.path.dirname(__file__), "configs", "rl_config.json")
    with open(config_path, "r") as f:
        full_cfg = json.load(f)

    shared_cfg = full_cfg["shared"]
    algo_cfg = full_cfg["algorithms"].get(args_cli.algorithm, {})

    num_of_action           = shared_cfg["num_of_action"]
    action_range            = shared_cfg["action_range"]
    discretize_state_weight = shared_cfg["discretize_state_weight"]
    discount                = shared_cfg["discount"]
    learning_rate = algo_cfg.get("learning_rate", shared_cfg.get("learning_rate", 0.1))

    Algorithm_name = args_cli.algorithm

    # Build agent with epsilon=0 (pure exploitation)
    match Algorithm_name:
        case "MC":
            from RL_Algorithm.Algorithm.MC import MC
            agent = MC(
                num_of_action=num_of_action, action_range=action_range,
                discretize_state_weight=discretize_state_weight,
                learning_rate=learning_rate,
                initial_epsilon=0.0, epsilon_decay=0.0, final_epsilon=0.0,
                discount_factor=discount,
            )
        case "SARSA":
            from RL_Algorithm.Algorithm.SARSA import SARSA
            agent = SARSA(
                num_of_action=num_of_action, action_range=action_range,
                discretize_state_weight=discretize_state_weight,
                learning_rate=learning_rate,
                initial_epsilon=0.0, epsilon_decay=0.0, final_epsilon=0.0,
                discount_factor=discount,
            )
        case "Q_Learning":
            from RL_Algorithm.Algorithm.Q_Learning import Q_Learning
            agent = Q_Learning(
                num_of_action=num_of_action, action_range=action_range,
                discretize_state_weight=discretize_state_weight,
                learning_rate=learning_rate,
                initial_epsilon=0.0, epsilon_decay=0.0, final_epsilon=0.0,
                discount_factor=discount,
            )
        case "Double_Q_Learning":
            from RL_Algorithm.Algorithm.Double_Q_Learning import Double_Q_Learning
            agent = Double_Q_Learning(
                num_of_action=num_of_action, action_range=action_range,
                discretize_state_weight=discretize_state_weight,
                learning_rate=learning_rate,
                initial_epsilon=0.0, epsilon_decay=0.0, final_epsilon=0.0,
                discount_factor=discount,
            )

    # Force 100% exploitation regardless of config
    agent.epsilon = 0.0

    # Load Q-table
    qtable_dir = os.path.dirname(args_cli.qtable_path)
    qtable_file = os.path.basename(args_cli.qtable_path)
    agent.load_q_value(qtable_dir, qtable_file)

    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    trajectory_rows = []  # per-step data for analysis plots

    while simulation_app.is_running():
        with torch.inference_mode():
            for ep in range(args_cli.num_episodes):
                obs, _ = env.reset()
                done = False
                ep_reward = 0.0
                ep_steps = 0

                while not done:
                    # agent stepping
                    action, action_idx = agent.get_action(obs)

                    # record state before stepping
                    state = obs["policy"].cpu().numpy().flatten()
                    action_val = float(action.item())

                    # env stepping
                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    done = bool(terminated.item()) or bool(truncated.item())
                    r = float(reward.item())
                    ep_reward += r
                    ep_steps += 1

                    trajectory_rows.append({
                        "episode": ep,
                        "step": ep_steps,
                        "cart_pos": float(state[0]),
                        "pole_angle": float(state[1]),
                        "cart_vel": float(state[2]),
                        "pole_vel": float(state[3]),
                        "action_idx": action_idx,
                        "action_val": action_val,
                        "reward": r,
                    })

                    obs = next_obs

                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_steps)
                print(f"  Episode {ep+1}/{args_cli.num_episodes}: "
                      f"reward={ep_reward:.2f}, length={ep_steps}")

        break  # Exit simulation_app.is_running() loop after evaluation

    # Save trajectory CSV if directory specified
    if args_cli.trajectory_dir and trajectory_rows:
        os.makedirs(args_cli.trajectory_dir, exist_ok=True)
        traj_path = os.path.join(args_cli.trajectory_dir, f"{Algorithm_name}_trajectory.csv")
        with open(traj_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=trajectory_rows[0].keys())
            writer.writeheader()
            writer.writerows(trajectory_rows)
        print(f"Trajectory saved to: {traj_path}")

    # ==================================================================== #

    # Compute and print summary
    mean_reward = float(np.mean(episode_rewards))
    mean_length = float(np.mean(episode_lengths))
    print(f"\nEvaluation Summary ({Algorithm_name}):")
    print(f"  mean_reward = {mean_reward:.2f}")
    print(f"  mean_length = {mean_length:.1f}")

    # Append results to output CSV
    csv_path = args_cli.output_csv
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "algorithm", "qtable_path", "num_episodes",
            "mean_reward", "mean_length",
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "algorithm": Algorithm_name,
            "qtable_path": args_cli.qtable_path,
            "num_episodes": args_cli.num_episodes,
            "mean_reward": mean_reward,
            "mean_length": mean_length,
        })
    print(f"Results appended to: {csv_path}")

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
