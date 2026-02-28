"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import json

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# --------------------------------------------------------------------------- #
# Algorithm selection via environment variable.
# Hydra has its own internal argparse that rejects unknown --flags, so we
# cannot use a normal CLI argument. Use RL_ALGORITHM env var instead:
#
#   RL_ALGORITHM=MC python scripts/RL_Algorithm/train.py --task ... --headless
# --------------------------------------------------------------------------- #
_ALGORITHM_CHOICES = ["MC", "SARSA", "Q_Learning", "Double_Q_Learning"]
_algorithm_name = os.environ.get("RL_ALGORITHM", "Q_Learning")

if _algorithm_name not in _ALGORITHM_CHOICES:
    raise ValueError(f"RL_ALGORITHM must be one of {_ALGORITHM_CHOICES}, got '{_algorithm_name}'")

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

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
import numpy as np
import gymnasium as gym
import torch
from datetime import datetime
import random
from torch.utils.tensorboard import SummaryWriter

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    log_dir = os.path.join("logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    Algorithm_name = _algorithm_name
    task_name = str(args_cli.task).split('-')[0]  # e.g. Stabilize, SwingUp

    # Load per-algorithm hyperparameters from config file
    config_path = os.path.join(os.path.dirname(__file__), "configs", f"{Algorithm_name}.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    num_of_action           = cfg["num_of_action"]
    action_range            = cfg["action_range"]            # [min, max]
    discretize_state_weight = cfg["discretize_state_weight"] # [pose_cart, pose_pole, vel_cart, vel_pole]
    learning_rate           = cfg["learning_rate"]
    n_episodes              = cfg["n_episodes"]
    start_epsilon           = cfg["start_epsilon"]
    epsilon_decay           = cfg["epsilon_decay"]
    final_epsilon           = cfg["final_epsilon"]
    discount                = cfg["discount"]

    # Build agent based on selected algorithm
    match Algorithm_name:
        case "MC":
            from RL_Algorithm.Algorithm.MC import MC
            agent = MC(
                num_of_action=num_of_action,
                action_range=action_range,
                discretize_state_weight=discretize_state_weight,
                learning_rate=learning_rate,
                initial_epsilon=start_epsilon,
                epsilon_decay=epsilon_decay,
                final_epsilon=final_epsilon,
                discount_factor=discount,
            )
        case "SARSA":
            from RL_Algorithm.Algorithm.SARSA import SARSA
            agent = SARSA(
                num_of_action=num_of_action,
                action_range=action_range,
                discretize_state_weight=discretize_state_weight,
                learning_rate=learning_rate,
                initial_epsilon=start_epsilon,
                epsilon_decay=epsilon_decay,
                final_epsilon=final_epsilon,
                discount_factor=discount,
            )
        case "Q_Learning":
            from RL_Algorithm.Algorithm.Q_Learning import Q_Learning
            agent = Q_Learning(
                num_of_action=num_of_action,
                action_range=action_range,
                discretize_state_weight=discretize_state_weight,
                learning_rate=learning_rate,
                initial_epsilon=start_epsilon,
                epsilon_decay=epsilon_decay,
                final_epsilon=final_epsilon,
                discount_factor=discount,
            )
        case "Double_Q_Learning":
            from RL_Algorithm.Algorithm.Double_Q_Learning import Double_Q_Learning
            agent = Double_Q_Learning(
                num_of_action=num_of_action,
                action_range=action_range,
                discretize_state_weight=discretize_state_weight,
                learning_rate=learning_rate,
                initial_epsilon=start_epsilon,
                epsilon_decay=epsilon_decay,
                final_epsilon=final_epsilon,
                discount_factor=discount,
            )

    # ---- CSV logging setup ----
    csv_dir = os.path.join("logs", task_name, Algorithm_name)
    os.makedirs(csv_dir, exist_ok=True)
    csv_filename = os.path.join(
        csv_dir, f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    )
    q_col_names = [f"q_{i}" for i in range(num_of_action)]
    csv_fieldnames = [
        "episode", "step",
        "cart_pos", "pole_angle", "cart_vel", "pole_vel",
        "cart_pos_dis", "pole_angle_dis", "cart_vel_dis", "pole_vel_dis",
        "action_idx", "action_val",
        "reward", "epsilon",
    ] + q_col_names
    csv_file = open(csv_filename, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
    csv_writer.writeheader()

    def log_step(episode, step, raw_obs, obs_dis, action_idx, action_val, reward):
        """Write one row to the CSV log."""
        state = raw_obs["policy"].cpu().numpy().flatten()
        q_vals = agent.q_values[obs_dis]
        row = {
            "episode":        episode,
            "step":           step,
            "cart_pos":       float(state[0]),
            "pole_angle":     float(state[1]),
            "cart_vel":       float(state[2]),
            "pole_vel":       float(state[3]),
            "cart_pos_dis":   obs_dis[0],
            "pole_angle_dis": obs_dis[1],
            "cart_vel_dis":   obs_dis[2],
            "pole_vel_dis":   obs_dis[3],
            "action_idx":     action_idx,
            "action_val":     float(action_val),
            "reward":         float(reward),
            "epsilon":        float(agent.epsilon),
        }
        for i, q in enumerate(q_vals):
            row[f"q_{i}"] = float(q)
        csv_writer.writerow(row)

    # ---- TensorBoard setup ----
    tb_dir = os.path.join("logs", task_name, Algorithm_name,
                          f"tensorboard_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(tb_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_dir)
    print(f"[TensorBoard] tensorboard --logdir {tb_dir}")

    # ---- Q-value save directory ----
    q_save_dir = os.path.join("q_value", task_name, Algorithm_name)
    os.makedirs(q_save_dir, exist_ok=True)

    from tqdm import tqdm

    # Number of parallel environments
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1

    # ---- Per-env state tracking ----
    episode_rewards = np.zeros(num_envs)   # cumulative reward for each env's current episode
    episode_steps   = np.zeros(num_envs, dtype=int)  # steps in current episode per env
    total_episodes  = 0                    # total completed episodes across all envs
    sum_reward      = 0.0                  # accumulates episode returns for periodic printing
    sum_ep_length   = 0.0                  # accumulates episode lengths for periodic logging
    last_log_ep     = 0                    # last episode count when we printed/saved
    global_step     = 0

    # MC: per-env episode histories (agent's single-list histories won't work for multi-env)
    if Algorithm_name == "MC":
        mc_obs_hist    = [[] for _ in range(num_envs)]
        mc_action_hist = [[] for _ in range(num_envs)]
        mc_reward_hist = [[] for _ in range(num_envs)]

    # SARSA: per-env current (obs_dis, action_idx) — needed for on-policy next-action tracking
    if Algorithm_name == "SARSA":
        sarsa_obs_dis    = [None] * num_envs
        sarsa_action_idx = [None] * num_envs

    # Initial environment reset
    obs, _ = env.reset()

    # SARSA: initialise first action for every env before the loop
    if Algorithm_name == "SARSA":
        for i in range(num_envs):
            obs_i = {"policy": obs["policy"][i : i + 1]}
            sarsa_obs_dis[i]    = agent.discretize_state(obs_i)
            sarsa_action_idx[i] = agent.get_discretize_action(sarsa_obs_dis[i])

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():

            pbar = tqdm(total=n_episodes, desc=f"[{Algorithm_name}] Episodes")

            while total_episodes < n_episodes:

                # ---- Select action for every env ----
                action_indices = []
                obs_dis_list   = []

                for i in range(num_envs):
                    obs_i = {"policy": obs["policy"][i : i + 1]}

                    if Algorithm_name == "SARSA":
                        # Reuse previously selected action (on-policy requirement)
                        obs_dis    = sarsa_obs_dis[i]
                        action_idx = sarsa_action_idx[i]
                    else:
                        obs_dis    = agent.discretize_state(obs_i)
                        action_idx = agent.get_discretize_action(obs_dis)

                    obs_dis_list.append(obs_dis)
                    action_indices.append(action_idx)

                    # MC: append this step to the per-env history
                    if Algorithm_name == "MC":
                        mc_obs_hist[i].append(obs_dis)
                        mc_action_hist[i].append(action_idx)

                # ---- Build batch action tensor [num_envs, 1] ----
                action_vals = [
                    action_range[0] + (action_range[1] - action_range[0]) * a / (num_of_action - 1)
                    for a in action_indices
                ]
                action_tensor = torch.tensor([[v] for v in action_vals], dtype=torch.float32)

                # ---- Step environment ----
                next_obs, reward, terminated, truncated, _ = env.step(action_tensor)
                done_flags = terminated | truncated   # bool tensor [num_envs]

                # ---- Per-env update ----
                ep_completed_this_step = 0

                for i in range(num_envs):
                    obs_i      = {"policy": obs["policy"][i : i + 1]}
                    next_obs_i = {"policy": next_obs["policy"][i : i + 1]}

                    r_i    = float(reward[i].item())
                    done_i = bool(done_flags[i].item())

                    episode_rewards[i] += r_i
                    episode_steps[i]   += 1

                    # Capture return before match/case may reset it (for TB logging)
                    ep_return_snapshot = float(episode_rewards[i])
                    ep_steps_snapshot  = int(episode_steps[i])

                    match Algorithm_name:

                        case "MC":
                            mc_reward_hist[i].append(r_i)

                            if done_i:
                                # Compute discounted returns backwards and update Q-table
                                G = 0.0
                                for t in reversed(range(len(mc_reward_hist[i]))):
                                    G = discount * G + mc_reward_hist[i][t]
                                    s = mc_obs_hist[i][t]
                                    a = mc_action_hist[i][t]
                                    agent.n_values[s][a] += 1
                                    error = G - agent.q_values[s][a]
                                    agent.q_values[s][a] += error / agent.n_values[s][a]
                                    agent.training_error.append(abs(error))

                                mc_obs_hist[i].clear()
                                mc_action_hist[i].clear()
                                mc_reward_hist[i].clear()

                                sum_reward += episode_rewards[i]
                                episode_rewards[i] = 0.0
                                total_episodes += 1
                                ep_completed_this_step += 1

                        case "SARSA":
                            next_obs_dis    = agent.discretize_state(next_obs_i)
                            next_action_idx = agent.get_discretize_action(next_obs_dis)

                            agent.update(obs_dis_list[i], action_indices[i], r_i,
                                         next_obs_dis, next_action_idx, done_i)

                            if done_i:
                                sum_reward += episode_rewards[i]
                                episode_rewards[i] = 0.0
                                total_episodes += 1
                                ep_completed_this_step += 1
                                # Isaac Lab auto-resets: next_obs_i is already the reset obs
                                sarsa_obs_dis[i]    = agent.discretize_state(next_obs_i)
                                sarsa_action_idx[i] = agent.get_discretize_action(sarsa_obs_dis[i])
                            else:
                                sarsa_obs_dis[i]    = next_obs_dis
                                sarsa_action_idx[i] = next_action_idx

                        case "Q_Learning":
                            next_obs_dis = agent.discretize_state(next_obs_i)
                            agent.update(obs_dis_list[i], action_indices[i], r_i, next_obs_dis, done_i)

                            if done_i:
                                sum_reward += episode_rewards[i]
                                episode_rewards[i] = 0.0
                                total_episodes += 1
                                ep_completed_this_step += 1

                        case "Double_Q_Learning":
                            next_obs_dis = agent.discretize_state(next_obs_i)
                            agent.update(obs_dis_list[i], action_indices[i], r_i, next_obs_dis, done_i)

                            if done_i:
                                sum_reward += episode_rewards[i]
                                episode_rewards[i] = 0.0
                                total_episodes += 1
                                ep_completed_this_step += 1

                    # TensorBoard: log once per completed episode (all envs)
                    if done_i:
                        tb_writer.add_scalar("episode/reward",  ep_return_snapshot, total_episodes)
                        tb_writer.add_scalar("episode/length",  ep_steps_snapshot,  total_episodes)
                        tb_writer.add_scalar("episode/epsilon", agent.epsilon,       total_episodes)
                        sum_ep_length += ep_steps_snapshot
                        episode_steps[i] = 0

                    # Log only env 0 every step to keep CSV size manageable
                    if i == 0:
                        log_step(total_episodes, global_step, obs_i,
                                 obs_dis_list[i], action_indices[i], action_vals[i], r_i)

                # ---- Progress bar update ----
                if ep_completed_this_step > 0:
                    pbar.update(ep_completed_this_step)

                # ---- Periodic print and Q-value save (every 100 completed episodes) ----
                if total_episodes - last_log_ep >= 100 and total_episodes > 0:
                    n_new    = total_episodes - last_log_ep
                    avg      = sum_reward / n_new
                    avg_len  = sum_ep_length / n_new
                    print(f"\n[Episode {total_episodes}] avg_score: {avg:.2f}  avg_len: {avg_len:.0f}  epsilon: {agent.epsilon:.4f}")

                    # TensorBoard: periodic averages
                    tb_writer.add_scalar("train/avg_episode_reward", avg,      total_episodes)
                    tb_writer.add_scalar("train/avg_episode_length", avg_len,  total_episodes)
                    tb_writer.add_scalar("train/epsilon",            agent.epsilon, total_episodes)
                    if agent.training_error:
                        mean_err = float(np.mean(agent.training_error[-1000:]))
                        tb_writer.add_scalar("train/mean_td_error", mean_err, total_episodes)

                    sum_reward  = 0.0
                    sum_ep_length = 0.0
                    last_log_ep = total_episodes

                    q_value_file = (
                        f"{Algorithm_name}_{total_episodes}"
                        f"_{num_of_action}_{action_range[1]}"
                        f"_{discretize_state_weight[0]}_{discretize_state_weight[1]}.json"
                    )
                    agent.save_q_value(q_save_dir, q_value_file)

                # ---- Epsilon decay: once per global step ----
                agent.decay_epsilon()

                obs = next_obs
                global_step += 1

            pbar.close()

        csv_file.close()
        tb_writer.close()
        print("!!! Training is complete !!!")
        print(f"CSV log saved to:  {csv_filename}")
        print(f"TensorBoard logs:  {tb_dir}")
        print(f"  → run: tensorboard --logdir {tb_dir}")
        break

    # ==================================================================== #

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
