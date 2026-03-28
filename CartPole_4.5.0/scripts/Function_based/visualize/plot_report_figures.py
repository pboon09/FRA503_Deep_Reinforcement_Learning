#!/usr/bin/env python3
"""Generate report figures for all 8 function-based RL algorithms.

Figures:
  1. fig1_learning_curves.png    -- Return vs episode (all algos, mean +/- std band)
  2. fig2_sample_efficiency.png  -- Return vs total env steps (fair x-axis)
  3. fig3_deployment.png         -- (a) bar chart + (b) per-episode scatter
  4. fig4_convergence_speed.png  -- Horizontal bar: steps to reach threshold
  5. fig5_reward_per_step.png    -- Reward efficiency (return/length) over episodes
  6. fig6_epsilon_decay.png      -- Epsilon decay (value-based only)
  7. fig7_episode_length.png     -- Episode length over training
  8. fig8_deployment_boxplot.png -- Deployment performance boxplot
  9. fig9_policy_surface.png     -- 3D policy surface (actor-critic algos)
  10. fig10_value_surface.png    -- 3D value surface

Data:
  - Training:   experiments/suite_1_baseline/{algo}.csv
  - Deployment: experiments/suite_1_baseline/{algo}_deploy.csv
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 1.8,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

ALGOS = ["Linear_Q", "DQN", "MC_REINFORCE", "AC", "A2C", "PPO", "SAC", "TD3"]
ALGO_DISPLAY = {
    "Linear_Q": "Linear Q",
    "DQN": "DQN",
    "MC_REINFORCE": "MC REINFORCE",
    "AC": "Actor-Critic",
    "A2C": "A2C",
    "PPO": "PPO",
    "SAC": "SAC",
    "TD3": "TD3",
}
ALGO_COLORS = {
    "Linear_Q": "#0072B2",
    "DQN": "#E69F00",
    "MC_REINFORCE": "#009E73",
    "AC": "#D55E00",
    "A2C": "#CC79A7",
    "PPO": "#56B4E9",
    "SAC": "#F0E442",
    "TD3": "#999999",
}

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.join(ROOT, "experiments", "suite_1_baseline")
CFG_PATH = os.path.join(ROOT, "scripts", "Function_based", "configs", "rl_config.json")


def _load_cfg():
    if os.path.isfile(CFG_PATH):
        with open(CFG_PATH) as f:
            return json.load(f)
    return {}


def load_csv(algo):
    p = os.path.join(EXP_DIR, f"{algo}.csv")
    return pd.read_csv(p) if os.path.isfile(p) else None


def load_deploy(algo):
    p = os.path.join(EXP_DIR, f"{algo}_deploy.csv")
    return pd.read_csv(p) if os.path.isfile(p) else None


def adaptive_window(n):
    return max(1, n // 15)


# --------------------------------------------------------------------------- #
# Fig 1: Learning Curves -- Return vs Batch Step
# --------------------------------------------------------------------------- #
def make_fig1(out):
    fig, ax = plt.subplots(figsize=(14, 5))
    cfg = _load_cfg()
    num_envs = cfg.get("shared", {}).get("num_envs", 256)
    total_steps = cfg.get("shared", {}).get("total_steps", 20000)
    for algo in ALGOS:
        df = load_csv(algo)
        if df is None or "ep_return" not in df.columns or "global_step" not in df.columns:
            continue
        # Convert global_step to batch step (0 .. total_steps)
        batch_step = df["global_step"] / num_envs
        y = df["ep_return"]
        w = adaptive_window(len(y))
        smoothed = y.rolling(w, min_periods=1).mean()
        std = y.rolling(w, min_periods=1).std().fillna(0)
        ax.plot(batch_step.values, smoothed.values,
                label=ALGO_DISPLAY[algo], color=ALGO_COLORS[algo])
        ax.fill_between(batch_step.values,
                        (smoothed - std).values, (smoothed + std).values,
                        alpha=0.15, color=ALGO_COLORS[algo])
    ax.axhline(y=950, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Batch Step")
    ax.set_ylabel("Episode Return (rolling mean +/- std)")
    ax.set_title("Learning Efficiency: Return vs Batch Step", fontweight="bold")
    ax.set_xlim(0, total_steps)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_learning_curves.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_learning_curves.png")


# --------------------------------------------------------------------------- #
# Fig 2: Sample Efficiency -- Return vs Total Env Steps
# --------------------------------------------------------------------------- #
def make_fig2(out):
    fig, ax = plt.subplots(figsize=(14, 5))
    for algo in ALGOS:
        df = load_csv(algo)
        if df is None or "global_step" not in df.columns:
            continue
        y = df["ep_return"]
        w = adaptive_window(len(y))
        smoothed = y.rolling(w, min_periods=1).mean()
        ax.plot(df["global_step"].values, smoothed.values,
                label=ALGO_DISPLAY[algo], color=ALGO_COLORS[algo])
    ax.axhline(y=950, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Total Environment Steps")
    ax.set_ylabel("Episode Return (rolling mean)")
    ax.set_title("Sample Efficiency: Return vs Environment Steps", fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_sample_efficiency.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_sample_efficiency.png")


# --------------------------------------------------------------------------- #
# Fig 3: Deployment -- (a) bar chart + (b) per-episode scatter
# --------------------------------------------------------------------------- #
def make_fig3(out):
    deploy_data = {}
    for algo in ALGOS:
        df = load_deploy(algo)
        if df is not None and "ep_return" in df.columns:
            deploy_data[algo] = df

    if not deploy_data:
        print("  [fig3] No deployment data, skipping.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    gridspec_kw={"width_ratios": [1, 1.3]})

    names, means, stds, colors = [], [], [], []
    for algo in ALGOS:
        if algo not in deploy_data:
            continue
        df = deploy_data[algo]
        names.append(ALGO_DISPLAY[algo])
        means.append(df["ep_return"].mean())
        stds.append(df["ep_return"].std())
        colors.append(ALGO_COLORS[algo])

    x = np.arange(len(names))
    ax1.bar(x, means, yerr=stds, color=colors, capsize=6,
            alpha=0.85, edgecolor="black", linewidth=0.8, width=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9, rotation=15)
    ax1.set_ylabel("Episode Return (mean +/- std)")
    ax1.set_title("(a) Deployment Performance", fontweight="bold", fontsize=12)
    ax1.set_ylim(bottom=0)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.text(i, m + s + max(10, ax1.get_ylim()[1] * 0.02),
                 f"{m:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    for algo in ALGOS:
        if algo not in deploy_data:
            continue
        df = deploy_data[algo]
        eps = df["episode"].values if "episode" in df.columns else np.arange(len(df))
        ax2.plot(eps, df["ep_return"].values, marker="o", markersize=6,
                 linewidth=1.5, label=ALGO_DISPLAY[algo], color=ALGO_COLORS[algo],
                 alpha=0.8)
    ax2.axhline(y=950, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Evaluation Episode")
    ax2.set_ylabel("Episode Return")
    ax2.set_title("(b) Per-Episode Consistency", fontweight="bold", fontsize=12)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=9)

    fig.suptitle("Deployment Performance (Deterministic Policy, 10 Episodes)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_deployment.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_deployment.png")


# --------------------------------------------------------------------------- #
# Fig 4: Convergence Speed -- Steps to reach threshold
# --------------------------------------------------------------------------- #
def make_fig4(out, threshold=200):
    results = {}
    for algo in ALGOS:
        df = load_csv(algo)
        if df is None or "global_step" not in df.columns:
            continue
        w = adaptive_window(len(df))
        smoothed = df["ep_return"].rolling(w, min_periods=1).mean()
        reached = smoothed[smoothed >= threshold]
        if len(reached) > 0:
            results[algo] = df["global_step"].iloc[reached.index[0]]
        else:
            results[algo] = None

    if not results:
        print("  [fig4] No data, skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    names, steps, colors_list = [], [], []
    for algo in reversed(ALGOS):
        if algo not in results:
            continue
        names.append(ALGO_DISPLAY[algo])
        steps.append(results[algo] if results[algo] is not None else 0)
        colors_list.append(ALGO_COLORS[algo])

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, steps, color=colors_list, alpha=0.85,
                   edgecolor="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Total Environment Steps to Reach Threshold")
    ax.set_title(f"Convergence Speed (threshold = {threshold} return)",
                 fontweight="bold")
    for i, (bar, s) in enumerate(zip(bars, steps)):
        if s > 0:
            ax.text(s + ax.get_xlim()[1] * 0.01, i, f"{s:,}",
                    va="center", fontsize=10)
        else:
            ax.text(ax.get_xlim()[1] * 0.5, i, "Did Not Reach",
                    va="center", fontsize=10, color="red", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_convergence_speed.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_convergence_speed.png")


# --------------------------------------------------------------------------- #
# Fig 5: Reward Efficiency -- Return per Step
# --------------------------------------------------------------------------- #
def make_fig5(out):
    fig, ax = plt.subplots(figsize=(14, 5))
    cfg = _load_cfg()
    num_envs = cfg.get("shared", {}).get("num_envs", 256)
    total_steps = cfg.get("shared", {}).get("total_steps", 20000)
    for algo in ALGOS:
        df = load_csv(algo)
        if df is None or "ep_return" not in df.columns or "ep_length" not in df.columns:
            continue
        batch_step = df["global_step"] / num_envs
        rps = df["ep_return"] / df["ep_length"].clip(lower=1)
        w = adaptive_window(len(rps))
        smoothed = rps.rolling(w, min_periods=1).mean()
        ax.plot(batch_step.values, smoothed.values, label=ALGO_DISPLAY[algo],
                color=ALGO_COLORS[algo])
    ax.set_xlabel("Batch Step")
    ax.set_ylabel("Return per Step (rolling mean)")
    ax.set_title("Reward Efficiency: Average Reward per Timestep", fontweight="bold")
    ax.set_xlim(0, total_steps)
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_reward_per_step.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig5_reward_per_step.png")


# --------------------------------------------------------------------------- #
# Fig 6: Epsilon Decay (value-based only)
# --------------------------------------------------------------------------- #
def make_fig6(out):
    fig, ax = plt.subplots(figsize=(14, 5))
    cfg = _load_cfg()
    num_envs = cfg.get("shared", {}).get("num_envs", 256)
    total_steps = cfg.get("shared", {}).get("total_steps", 20000)
    plotted = False
    for algo in ["Linear_Q", "DQN"]:
        df = load_csv(algo)
        if df is None or "epsilon" not in df.columns:
            continue
        batch_step = df["global_step"] / num_envs
        ax.plot(batch_step.values, df["epsilon"].values,
                label=ALGO_DISPLAY[algo], color=ALGO_COLORS[algo])
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Batch Step")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate Decay (Value-Based Algorithms)", fontweight="bold")
    ax.set_xlim(0, total_steps)
    ax.set_ylim(bottom=0, top=1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig6_epsilon_decay.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig6_epsilon_decay.png")


# --------------------------------------------------------------------------- #
# Fig 7: Episode Length over Training
# --------------------------------------------------------------------------- #
def make_fig7(out):
    fig, ax = plt.subplots(figsize=(14, 5))
    cfg = _load_cfg()
    num_envs = cfg.get("shared", {}).get("num_envs", 256)
    total_steps = cfg.get("shared", {}).get("total_steps", 20000)
    for algo in ALGOS:
        df = load_csv(algo)
        if df is None or "ep_length" not in df.columns or "global_step" not in df.columns:
            continue
        batch_step = df["global_step"] / num_envs
        y = df["ep_length"]
        w = adaptive_window(len(y))
        smoothed = y.rolling(w, min_periods=1).mean()
        ax.plot(batch_step.values, smoothed.values,
                label=ALGO_DISPLAY[algo], color=ALGO_COLORS[algo])
    ax.axhline(y=1000, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Batch Step")
    ax.set_ylabel("Episode Length (steps, rolling mean)")
    ax.set_title("Survival Time: Episode Length vs Batch Step", fontweight="bold")
    ax.set_xlim(0, total_steps)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig7_episode_length.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig7_episode_length.png")


# --------------------------------------------------------------------------- #
# Fig 8: Deployment Boxplot
# --------------------------------------------------------------------------- #
def make_fig8(out):
    data_list, labels, colors = [], [], []
    for algo in ALGOS:
        df = load_deploy(algo)
        if df is None or "ep_return" not in df.columns:
            continue
        data_list.append(df["ep_return"].values)
        labels.append(ALGO_DISPLAY[algo])
        colors.append(ALGO_COLORS[algo])

    if not data_list:
        print("  [fig8] No deployment data, skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data_list, tick_labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markersize=6))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("Episode Return")
    ax.set_title("Deployment Performance Distribution (10 Episodes)", fontweight="bold")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig8_deployment_boxplot.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig8_deployment_boxplot.png")


# --------------------------------------------------------------------------- #
# Fig 9 & 10: Policy & Value Surface
# --------------------------------------------------------------------------- #
def make_fig9(out):
    """Plot policy action and value function heatmaps for actor-critic algos."""
    import sys
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "RL_Algorithm"))

    import torch

    ac_algos = {"AC": "AC", "A2C": "A2C", "PPO": "PPO"}
    model_dir = os.path.join(ROOT, "model", "Stabilize")

    cfg_path = os.path.join(ROOT, "scripts", "Function_based", "configs", "rl_config.json")
    if not os.path.isfile(cfg_path):
        print("  [fig9] No config file, skipping.")
        return
    with open(cfg_path) as _f:
        _cfg = json.load(_f)
    _shared = _cfg["shared"]

    loaded = {}
    for algo, label in ac_algos.items():
        model_path = os.path.join(model_dir, algo, f"{algo}_final.pth")
        if not os.path.isfile(model_path):
            continue
        try:
            _ac = _cfg["algorithms"].get(algo, {})
            if algo == "PPO":
                from RL_Algorithm.Function_based.PPO import PPO as PPO_cls
                agent = PPO_cls(
                    device=torch.device("cpu"),
                    num_of_action=_ac.get("num_of_action", 1),
                    action_range=_shared["action_range"],
                    n_observations=_shared["n_observations"],
                    hidden_dims=_ac["hidden_dims"],
                    activation=_ac.get("activation", "elu"),
                    action_type=_ac.get("action_type", "continuous"),
                    init_noise_std=_ac.get("init_noise_std", 1.0),
                    num_learning_epochs=_ac["num_learning_epochs"],
                    num_mini_batches=_ac["num_mini_batches"],
                    clip_param=_ac["clip_param"],
                    gamma=_shared["discount_factor"],
                    lam=_ac["lam"],
                    value_loss_coef=_ac.get("value_loss_coef", 1.0),
                    entropy_coef=_ac.get("entropy_coef", 0.01),
                    learning_rate=_ac["learning_rate"],
                    max_grad_norm=_ac.get("max_grad_norm", 1.0),
                    desired_kl=_ac.get("desired_kl", 0.0),
                )
                agent.load_model(os.path.join(model_dir, algo), f"{algo}_final.pth")
                loaded[label] = agent.policy
            elif algo == "AC":
                from RL_Algorithm.Function_based.AC import AC as AC_cls
                agent = AC_cls(
                    device=torch.device("cpu"),
                    num_of_action=_ac.get("num_of_action", 1),
                    action_range=_shared["action_range"],
                    n_observations=_shared["n_observations"],
                    hidden_dims=_ac["hidden_dims"],
                    activation=_ac.get("activation", "elu"),
                    action_type=_ac.get("action_type", "continuous"),
                    init_noise_std=_ac.get("init_noise_std", 1.0),
                    learning_rate=_ac["learning_rate"],
                    discount_factor=_shared["discount_factor"],
                    value_loss_coef=_ac.get("value_loss_coef", 0.5),
                    entropy_coef=_ac.get("entropy_coef", 0.01),
                    max_grad_norm=_ac.get("max_grad_norm", 0.5),
                )
                agent.load_model(os.path.join(model_dir, algo), f"{algo}_final.pth")
                loaded[label] = agent.policy
            elif algo == "A2C":
                from RL_Algorithm.Function_based.A2C import A2C as A2C_cls
                agent = A2C_cls(
                    device=torch.device("cpu"),
                    num_of_action=_ac.get("num_of_action", 1),
                    action_range=_shared["action_range"],
                    n_observations=_shared["n_observations"],
                    hidden_dims=_ac["hidden_dims"],
                    activation=_ac.get("activation", "elu"),
                    action_type=_ac.get("action_type", "continuous"),
                    init_noise_std=_ac.get("init_noise_std", 1.0),
                    learning_rate=_ac["learning_rate"],
                    discount_factor=_shared["discount_factor"],
                    value_loss_coef=_ac.get("value_loss_coef", 0.5),
                    entropy_coef=_ac.get("entropy_coef", 0.01),
                    max_grad_norm=_ac.get("max_grad_norm", 0.5),
                )
                agent.load_model(os.path.join(model_dir, algo), f"{algo}_final.pth")
                loaded[label] = agent.policy
        except Exception as e:
            print(f"  [fig9] Failed to load {algo}: {e}")

    # Try MC_REINFORCE
    reinforce_net = None
    rf_path = os.path.join(model_dir, "MC_REINFORCE", "MC_REINFORCE_final.pth")
    if os.path.isfile(rf_path):
        try:
            _mc = _cfg["algorithms"]["MC_REINFORCE"]
            from RL_Algorithm.Function_based.MC_REINFORCE import MC_REINFORCE as MC_cls
            agent = MC_cls(
                device=torch.device("cpu"),
                num_of_action=_mc.get("num_of_action", 1),
                action_range=_shared["action_range"],
                n_observations=_shared["n_observations"],
                hidden_dim=_mc["hidden_dim"],
                dropout=_mc.get("dropout", 0.0),
                action_type=_mc.get("action_type", "continuous"),
                learning_rate=_mc["learning_rate"],
                discount_factor=_shared["discount_factor"],
            )
            agent.load_model(os.path.join(model_dir, "MC_REINFORCE"), "MC_REINFORCE_final.pth")
            reinforce_net = agent.policy_net
        except Exception as e:
            print(f"  [fig9] Failed to load MC_REINFORCE: {e}")

    # Try DQN Q-surface
    dqn_model = None
    dqn_path = os.path.join(model_dir, "DQN", "DQN_final.pth")
    if os.path.isfile(dqn_path):
        try:
            _dqn = _cfg["algorithms"]["DQN"]
            from RL_Algorithm.Function_based.DQN import DQN as DQN_cls
            agent = DQN_cls(
                device=torch.device("cpu"),
                num_of_action=_dqn["num_of_action"],
                action_range=_shared["action_range"],
                n_observations=_shared["n_observations"],
                hidden_dim=_dqn["hidden_dim"],
                dropout=_dqn.get("dropout", 0.0),
                learning_rate=_dqn["learning_rate"],
                tau=_dqn["tau"],
                initial_epsilon=0.0, epsilon_decay=0.0, final_epsilon=0.0,
                discount_factor=_shared["discount_factor"],
                buffer_size=1000, batch_size=_dqn["batch_size"],
            )
            agent.load_model(os.path.join(model_dir, "DQN"), "DQN_final.pth")
            dqn_model = agent.policy_net
        except Exception as e:
            print(f"  [fig9] Failed to load DQN: {e}")

    # Try Linear Q
    linear_q_w = None
    lq_path = os.path.join(model_dir, "Linear_Q", "Linear_Q_final.npy")
    if os.path.isfile(lq_path):
        try:
            linear_q_w = np.load(lq_path)
        except Exception as e:
            print(f"  [fig9] Failed to load Linear_Q: {e}")

    n_models = len(loaded) + (1 if reinforce_net else 0) + (1 if dqn_model else 0) + (1 if linear_q_w is not None else 0)
    if n_models == 0:
        print("  [fig9] No models found, skipping.")
        return

    angle_range = np.linspace(-0.25, 0.25, 100)
    angvel_range = np.linspace(-3.0, 3.0, 100)
    AA, VV = np.meshgrid(angle_range, angvel_range)

    obs_grid = np.zeros((100 * 100, 4), dtype=np.float32)
    obs_grid[:, 1] = AA.flatten()
    obs_grid[:, 3] = VV.flatten()
    obs_tensor = torch.tensor(obs_grid)

    surfaces = []

    for label, policy in loaded.items():
        policy.eval()
        with torch.no_grad():
            actions = policy.act_inference(obs_tensor).numpy().reshape(100, 100)
            values = policy.evaluate(obs_tensor).numpy().reshape(100, 100)
        surfaces.append((label, actions, values))

    if reinforce_net is not None:
        reinforce_net.eval()
        with torch.no_grad():
            actions_rf = reinforce_net(obs_tensor).numpy().reshape(100, 100)
        surfaces.append(("MC REINFORCE", actions_rf, None))

    if dqn_model is not None:
        dqn_model.eval()
        with torch.no_grad():
            q_vals = dqn_model(obs_tensor).numpy()
        action_values = np.linspace(-2.5, 2.5, q_vals.shape[1])
        actions_cont = action_values[q_vals.argmax(axis=1)].reshape(100, 100)
        max_q = q_vals.max(axis=1).reshape(100, 100)
        surfaces.append(("DQN", actions_cont, max_q))

    if linear_q_w is not None:
        obs_scale = np.array([2.4, 3.0, 0.21, 3.0])
        obs_norm = np.clip(obs_grid / obs_scale, -1, 1)
        q_all = obs_norm @ linear_q_w
        action_values_lq = np.linspace(-2.5, 2.5, linear_q_w.shape[1])
        actions_lq = action_values_lq[q_all.argmax(axis=1)].reshape(100, 100)
        max_q_lq = q_all.max(axis=1).reshape(100, 100)
        surfaces.append(("Linear Q", actions_lq, max_q_lq))

    n = len(surfaces)
    if n == 0:
        print("  [fig9] No surfaces to plot, skipping.")
        return

    # Fig 9: 3D Policy Surface
    fig_pol = plt.figure(figsize=(6 * min(n, 4), 5 * ((n + 3) // 4)))
    for i, (label, actions_2d, _) in enumerate(surfaces):
        ax = fig_pol.add_subplot((n + 3) // 4, min(n, 4), i + 1, projection="3d")
        ax.plot_surface(AA, VV, actions_2d, cmap="RdBu_r", alpha=0.85,
                        rstride=2, cstride=2, edgecolor="none")
        ax.set_xlabel("Pole Angle", fontsize=9)
        ax.set_ylabel("Ang. Vel.", fontsize=9)
        ax.set_zlabel("Action", fontsize=9)
        ax.set_title(f"{label}", fontweight="bold", fontsize=11)
        ax.view_init(elev=25, azim=-60)
    fig_pol.suptitle("Policy Surface: Action vs (Pole Angle, Angular Velocity)\n[cart_pos=0, cart_vel=0]",
                     fontsize=14, fontweight="bold")
    fig_pol.tight_layout()
    fig_pol.savefig(os.path.join(out, "fig9_policy_surface.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_pol)
    print("  Saved fig9_policy_surface.png")

    # Fig 10: 3D Value Surface
    val_surfaces = [(l, v) for l, _, v in surfaces if v is not None]
    if val_surfaces:
        nv = len(val_surfaces)
        fig_val = plt.figure(figsize=(6 * min(nv, 4), 5 * ((nv + 3) // 4)))
        for i, (label, values_2d) in enumerate(val_surfaces):
            ax = fig_val.add_subplot((nv + 3) // 4, min(nv, 4), i + 1, projection="3d")
            ax.plot_surface(AA, VV, values_2d, cmap="viridis", alpha=0.85,
                            rstride=2, cstride=2, edgecolor="none")
            ax.set_xlabel("Pole Angle", fontsize=9)
            ax.set_ylabel("Ang. Vel.", fontsize=9)
            ax.set_zlabel("V(s) / max Q", fontsize=9)
            ax.set_title(f"{label}", fontweight="bold", fontsize=11)
            ax.view_init(elev=25, azim=-60)
        fig_val.suptitle("Value Surface: V(s) / max Q(s,a) vs (Pole Angle, Angular Velocity)\n[cart_pos=0, cart_vel=0]",
                         fontsize=14, fontweight="bold")
        fig_val.tight_layout()
        fig_val.savefig(os.path.join(out, "fig10_value_surface.png"), dpi=150, bbox_inches="tight")
        plt.close(fig_val)
        print("  Saved fig10_value_surface.png")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(ROOT, "figures"))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Generating figures... (ROOT={ROOT})")
    print(f"  CSVs in: {EXP_DIR}")

    make_fig1(args.output)
    make_fig2(args.output)
    make_fig3(args.output)
    make_fig4(args.output)
    make_fig5(args.output)
    make_fig6(args.output)
    make_fig7(args.output)
    make_fig8(args.output)
    make_fig9(args.output)

    print("Done.")


if __name__ == "__main__":
    main()
