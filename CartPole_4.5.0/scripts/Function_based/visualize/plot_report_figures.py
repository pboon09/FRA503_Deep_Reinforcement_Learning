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


def _resample_to_grid(batch_step, values, total_steps, n_bins=500):
    """Resample irregularly-spaced episode data onto a uniform grid.

    Returns (grid_x, mean_y, std_y) arrays of length n_bins.
    """
    grid = np.linspace(0, total_steps, n_bins + 1)
    mean_y = np.full(n_bins, np.nan)
    std_y = np.full(n_bins, 0.0)
    bs = np.asarray(batch_step)
    vals = np.asarray(values)
    for i in range(n_bins):
        mask = (bs >= grid[i]) & (bs < grid[i + 1])
        if mask.sum() > 0:
            mean_y[i] = vals[mask].mean()
            std_y[i] = vals[mask].std() if mask.sum() > 1 else 0.0
    grid_x = (grid[:-1] + grid[1:]) / 2
    # Forward-fill NaN bins
    valid = ~np.isnan(mean_y)
    if valid.sum() > 0:
        last_val = mean_y[valid][0]
        for i in range(n_bins):
            if np.isnan(mean_y[i]):
                mean_y[i] = last_val
            else:
                last_val = mean_y[i]
    # Smooth with small window
    w = max(1, n_bins // 20)
    mean_s = pd.Series(mean_y).rolling(w, min_periods=1).mean().values
    std_s = pd.Series(std_y).rolling(w, min_periods=1).mean().values
    return grid_x, mean_s, std_s


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
        batch_step = df["global_step"] / num_envs
        gx, gy, gs = _resample_to_grid(batch_step, df["ep_return"], total_steps)
        ax.plot(gx, gy, label=ALGO_DISPLAY[algo], color=ALGO_COLORS[algo])
        ax.fill_between(gx, gy - gs, gy + gs, alpha=0.15, color=ALGO_COLORS[algo])
    ax.axhline(y=950, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Batch Step")
    ax.set_ylabel("Episode Return (mean +/- std)")
    ax.set_title("Training Return vs Batch Step", fontweight="bold")
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
    cfg = _load_cfg()
    num_envs = cfg.get("shared", {}).get("num_envs", 256)
    total_steps = cfg.get("shared", {}).get("total_steps", 20000)
    max_env_steps = total_steps * num_envs
    for algo in ALGOS:
        df = load_csv(algo)
        if df is None or "global_step" not in df.columns:
            continue
        gx, gy, _ = _resample_to_grid(df["global_step"], df["ep_return"], max_env_steps)
        ax.plot(gx, gy, label=ALGO_DISPLAY[algo], color=ALGO_COLORS[algo])
    ax.axhline(y=950, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Total Environment Steps")
    ax.set_ylabel("Episode Return (mean)")
    ax.set_title("Return vs Env Steps", fontweight="bold")
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

    # (a) Bar chart with mean +/- std
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
    ax1.set_title("(a) Mean Return", fontweight="bold", fontsize=12)
    ax1.set_ylim(bottom=0)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.text(i, m + s + max(10, ax1.get_ylim()[1] * 0.02),
                 f"{m:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # (b) Violin + strip plot (replaces hard-to-read per-episode scatter)
    violin_data, violin_labels, violin_colors = [], [], []
    for algo in ALGOS:
        if algo not in deploy_data:
            continue
        violin_data.append(deploy_data[algo]["ep_return"].values)
        violin_labels.append(ALGO_DISPLAY[algo])
        violin_colors.append(ALGO_COLORS[algo])

    positions = np.arange(len(violin_data))
    parts = ax2.violinplot(violin_data, positions=positions, showmeans=True,
                            showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(violin_colors[i])
        pc.set_alpha(0.5)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("red")
    parts["cmedians"].set_linestyle("--")
    # Overlay individual points with jitter
    for i, data in enumerate(violin_data):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(data))
        ax2.scatter(positions[i] + jitter, data, s=8, alpha=0.4,
                    color=violin_colors[i], edgecolors="none", zorder=3)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(violin_labels, fontsize=9, rotation=15)
    ax2.set_ylabel("Episode Return")
    ax2.set_title("(b) Return Distribution", fontweight="bold", fontsize=12)
    ax2.set_ylim(bottom=0)

    fig.suptitle("Deployment Performance", fontsize=14, fontweight="bold")
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
        gx, gy, _ = _resample_to_grid(batch_step, rps, total_steps)
        ax.plot(gx, gy, label=ALGO_DISPLAY[algo], color=ALGO_COLORS[algo])
    ax.set_xlabel("Batch Step")
    ax.set_ylabel("Return per Step (mean)")
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
    ax.set_title("Exploration Decay", fontweight="bold")
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
        gx, gy, _ = _resample_to_grid(batch_step, df["ep_length"], total_steps)
        ax.plot(gx, gy, label=ALGO_DISPLAY[algo], color=ALGO_COLORS[algo])
    ax.axhline(y=1000, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Batch Step")
    ax.set_ylabel("Episode Length (steps, mean)")
    ax.set_title("Episode Length vs Batch Step", fontweight="bold")
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

    fig, ax = plt.subplots(figsize=(12, 6))
    positions = np.arange(len(data_list))
    # Violin
    parts = ax.violinplot(data_list, positions=positions, showmeans=False,
                           showmedians=False, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.35)
    # Boxplot overlay
    bp = ax.boxplot(data_list, positions=positions, patch_artist=True,
                    showmeans=True, widths=0.2,
                    meanprops=dict(marker="D", markerfacecolor="white", markersize=5),
                    medianprops=dict(color="black", linewidth=1.5),
                    boxprops=dict(linewidth=0.8))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    # Strip (jittered points)
    rng = np.random.default_rng(42)
    for i, data in enumerate(data_list):
        jitter = rng.uniform(-0.08, 0.08, len(data))
        ax.scatter(positions[i] + jitter, data, s=10, alpha=0.35,
                   color=colors[i], edgecolors="none", zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Episode Return")
    ax.set_title("Deployment Return Distribution", fontweight="bold")
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
    fig_pol.suptitle("Learned Policy Surface",
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
        fig_val.suptitle("Learned Value Surface",
                         fontsize=14, fontweight="bold")
        fig_val.tight_layout()
        fig_val.savefig(os.path.join(out, "fig10_value_surface.png"), dpi=150, bbox_inches="tight")
        plt.close(fig_val)
        print("  Saved fig10_value_surface.png")


# --------------------------------------------------------------------------- #
# Fig 11: Focused 2x2 Policy/Value Contrast (PPO vs TD3)
# --------------------------------------------------------------------------- #
def make_fig11_contrast(out, agent_a="PPO", agent_b="TD3"):
    """Generate a 2x2 grid: (policy, value) x (robust agent, brittle agent)."""
    import sys
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "RL_Algorithm"))
    import torch

    cfg_path = os.path.join(ROOT, "scripts", "Function_based", "configs", "rl_config.json")
    if not os.path.isfile(cfg_path):
        print("  [fig11] No config file, skipping.")
        return
    with open(cfg_path) as _f:
        _cfg = json.load(_f)
    _shared = _cfg["shared"]
    model_dir = os.path.join(ROOT, "model", "Stabilize")

    # --- Observation grid ---
    angle_range = np.linspace(-0.25, 0.25, 100)
    angvel_range = np.linspace(-3.0, 3.0, 100)
    AA, VV = np.meshgrid(angle_range, angvel_range)
    obs_grid = np.zeros((100 * 100, 4), dtype=np.float32)
    obs_grid[:, 1] = AA.flatten()
    obs_grid[:, 3] = VV.flatten()
    obs_tensor = torch.tensor(obs_grid)

    def _load_ac(algo):
        _ac = _cfg["algorithms"].get(algo, {})
        if algo == "PPO":
            from RL_Algorithm.Function_based.PPO import PPO as Cls
            agent = Cls(
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
        elif algo in ("AC", "A2C"):
            if algo == "AC":
                from RL_Algorithm.Function_based.AC import AC as Cls
            else:
                from RL_Algorithm.Function_based.A2C import A2C as Cls
            agent = Cls(
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
        else:
            return None
        agent.load_model(os.path.join(model_dir, algo), f"{algo}_final.pth")
        return agent.policy

    def _load_td3():
        _td3 = _cfg["algorithms"]["TD3"]
        from RL_Algorithm.Function_based.TD3 import TD3 as Cls
        agent = Cls(
            device=torch.device("cpu"),
            num_of_action=_td3.get("num_of_action", 1),
            action_range=_shared["action_range"],
            n_observations=_shared["n_observations"],
            hidden_dim=_td3["hidden_dim"],
            learning_rate=_td3["learning_rate"],
            tau=_td3["tau"],
            discount_factor=_shared["discount_factor"],
            buffer_size=1000, batch_size=_td3["batch_size"],
            exploration_noise=_td3["exploration_noise"],
            target_noise=_td3["target_noise"],
            target_noise_clip=_td3["target_noise_clip"],
            policy_update_freq=_td3["policy_update_freq"],
        )
        agent.load_model(os.path.join(model_dir, "TD3"), "TD3_final.pth")
        return agent

    def _load_dqn():
        _dqn = _cfg["algorithms"]["DQN"]
        from RL_Algorithm.Function_based.DQN import DQN as Cls
        agent = Cls(
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
        return agent

    # --- Load models and compute surfaces ---
    results = {}  # algo -> (policy_surface, value_surface)
    for algo in [agent_a, agent_b]:
        try:
            if algo in ("PPO", "AC", "A2C"):
                policy = _load_ac(algo)
                policy.eval()
                with torch.no_grad():
                    act = policy.act_inference(obs_tensor).numpy().reshape(100, 100)
                    val = policy.evaluate(obs_tensor).numpy().reshape(100, 100)
                results[algo] = (act, val)
            elif algo == "TD3":
                agent = _load_td3()
                agent.actor.eval()
                agent.critic.eval()
                with torch.no_grad():
                    raw_act = agent.actor(obs_tensor)
                    act_scaled = (raw_act.numpy() * (_shared["action_range"][1])).reshape(100, 100)
                    q1, _ = agent.critic(obs_tensor, raw_act)
                    val = q1.numpy().reshape(100, 100)
                results[algo] = (act_scaled, val)
            elif algo == "DQN":
                agent = _load_dqn()
                agent.policy_net.eval()
                with torch.no_grad():
                    q_vals = agent.policy_net(obs_tensor).numpy()
                action_values = np.linspace(*_shared["action_range"], q_vals.shape[1])
                act = action_values[q_vals.argmax(axis=1)].reshape(100, 100)
                val = q_vals.max(axis=1).reshape(100, 100)
                results[algo] = (act, val)
        except Exception as e:
            print(f"  [fig11] Failed to load {algo}: {e}")

    if len(results) < 2:
        print(f"  [fig11] Need 2 models, got {list(results.keys())}. Skipping.")
        return

    # --- Plot 2x2 grid ---
    fig = plt.figure(figsize=(12, 10))
    titles = [
        (agent_a, "Policy Surface", 0),
        (agent_a, "Value Surface", 1),
        (agent_b, "Policy Surface", 0),
        (agent_b, "Value Surface", 1),
    ]
    labels = [
        f"{ALGO_DISPLAY.get(agent_a, agent_a)} (Robust)",
        f"{ALGO_DISPLAY.get(agent_a, agent_a)} (Robust)",
        f"{ALGO_DISPLAY.get(agent_b, agent_b)} (Brittle)",
        f"{ALGO_DISPLAY.get(agent_b, agent_b)} (Brittle)",
    ]
    cmaps = ["RdBu_r", "viridis", "RdBu_r", "viridis"]
    zlabels = ["Action (N)", "V(s)", "Action (N)", "Q(s,a)"]

    for idx, (algo, surf_type, surf_idx) in enumerate(titles):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        surface_data = results[algo][surf_idx]
        ax.plot_surface(AA, VV, surface_data, cmap=cmaps[idx], alpha=0.85,
                        rstride=2, cstride=2, edgecolor="none")
        ax.set_xlabel("Pole Angle", fontsize=10)
        ax.set_ylabel("Ang. Velocity", fontsize=10)
        ax.set_zlabel(zlabels[idx], fontsize=10)
        ax.set_title(f"{labels[idx]}: {surf_type}", fontweight="bold", fontsize=11)
        ax.view_init(elev=25, azim=-60)

    fig.suptitle(
        f"Policy and Value Surfaces: {ALGO_DISPLAY.get(agent_a, agent_a)} vs "
        f"{ALGO_DISPLAY.get(agent_b, agent_b)}\n"
        r"[cart_pos=0, cart_vel=0, varying $\theta$ and $\dot{\theta}$]",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fname = "fig3_policy_value_surfaces.png"
    fig.savefig(os.path.join(out, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# --------------------------------------------------------------------------- #
# Data loaders for new CSV files
# --------------------------------------------------------------------------- #
def load_losses(algo):
    p = os.path.join(EXP_DIR, f"{algo}_losses.csv")
    return pd.read_csv(p) if os.path.isfile(p) else None


def load_trajectory(algo):
    p = os.path.join(EXP_DIR, f"{algo}_deploy_trajectory.csv")
    return pd.read_csv(p) if os.path.isfile(p) else None


# --------------------------------------------------------------------------- #
# Phase Portrait: pole_angle vs pole_angular_velocity during deployment
# --------------------------------------------------------------------------- #
def make_phase_portrait(out):
    """Phase portrait (pole angle vs angular velocity) from deployment trajectories."""
    n_algos = sum(1 for a in ALGOS if load_trajectory(a) is not None)
    if n_algos == 0:
        print("  [phase_portrait] No trajectory data, skipping.")
        return

    cols = min(n_algos, 4)
    rows = (n_algos + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if n_algos == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    idx = 0
    for algo in ALGOS:
        traj = load_trajectory(algo)
        if traj is None or len(traj) == 0:
            continue
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        # Plot a few episodes as trajectories
        episodes = sorted(traj["episode"].unique())
        n_show = min(5, len(episodes))
        sample_eps = [episodes[int(i * len(episodes) / n_show)] for i in range(n_show)]

        for ep in sample_eps:
            ep_data = traj[traj["episode"] == ep]
            ax.plot(ep_data["pole_angle"], ep_data["pole_ang_vel"],
                    alpha=0.5, linewidth=0.6, color=ALGO_COLORS[algo])
        # Mark the origin (upright equilibrium)
        ax.scatter([0], [0], marker="x", s=80, color="red", zorder=5, linewidths=2)
        ax.set_xlabel("Pole Angle (rad)")
        ax.set_ylabel("Angular Velocity")
        ax.set_title(ALGO_DISPLAY[algo], fontweight="bold")
        idx += 1

    # Hide unused axes
    for i in range(idx, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)

    fig.suptitle("Phase Portrait (Deployment)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_phase_portrait.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_phase_portrait.png")


# --------------------------------------------------------------------------- #
# Train vs Deploy Gap: compare last training return with deployment return
# --------------------------------------------------------------------------- #
def make_train_deploy_gap(out):
    """Bar chart comparing final training return vs deployment return."""
    names, train_means, deploy_means, colors = [], [], [], []
    for algo in ALGOS:
        df_train = load_csv(algo)
        df_deploy = load_deploy(algo)
        if df_train is None or df_deploy is None:
            continue
        # Use last 5% of training episodes as "final training return"
        n_tail = max(1, len(df_train) // 20)
        train_mean = df_train["ep_return"].tail(n_tail).mean()
        deploy_mean = df_deploy["ep_return"].mean()
        names.append(ALGO_DISPLAY[algo])
        train_means.append(train_mean)
        deploy_means.append(deploy_mean)
        colors.append(ALGO_COLORS[algo])

    if not names:
        print("  [train_deploy_gap] No data, skipping.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, train_means, w, label="Training (last 5%)",
           color=colors, alpha=0.6, edgecolor="black", linewidth=0.8)
    ax.bar(x + w / 2, deploy_means, w, label="Deployment",
           color=colors, alpha=1.0, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, rotation=15)
    ax.set_ylabel("Mean Episode Return")
    ax.set_title("Training vs Deployment Return", fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_train_deploy_gap.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_train_deploy_gap.png")


# --------------------------------------------------------------------------- #
# Action Histogram: distribution of actions during deployment
# --------------------------------------------------------------------------- #
def make_action_histogram(out):
    """Histogram of actions taken during deployment for each algorithm."""
    n_algos = sum(1 for a in ALGOS if load_trajectory(a) is not None)
    if n_algos == 0:
        print("  [action_histogram] No trajectory data, skipping.")
        return

    cols = min(n_algos, 4)
    rows = (n_algos + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_algos == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    idx = 0
    for algo in ALGOS:
        traj = load_trajectory(algo)
        if traj is None or "action" not in traj.columns or len(traj) == 0:
            continue
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.hist(traj["action"].values, bins=50, color=ALGO_COLORS[algo],
                alpha=0.8, edgecolor="none", density=True)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Action (Force)")
        ax.set_ylabel("Density")
        ax.set_title(ALGO_DISPLAY[algo], fontweight="bold")
        idx += 1

    for i in range(idx, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)

    fig.suptitle("Action Distribution (Deployment)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_action_histogram.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_action_histogram.png")


# --------------------------------------------------------------------------- #
# Per-algorithm plots (friend's style)
# --------------------------------------------------------------------------- #
def make_per_algo_plots(out):
    """Generate per-algorithm plots: learning curve, reward w/ std,
    steps vs reward, episode length, actor loss, critic loss, entropy."""
    smooth_w = 50
    smooth_w_std = 100

    for algo in ALGOS:
        algo_dir = os.path.join(out, algo)
        os.makedirs(algo_dir, exist_ok=True)

        df = load_csv(algo)
        loss_df = load_losses(algo)
        color = ALGO_COLORS[algo]
        name = ALGO_DISPLAY[algo]

        if df is None:
            continue

        # 1. Learning curve (smoothed reward vs episode)
        fig, ax = plt.subplots(figsize=(10, 5))
        smoothed = df["ep_return"].rolling(smooth_w, min_periods=1).mean()
        ax.plot(df["episode"], smoothed, color=color, label=f"Smoothed (w={smooth_w})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title(f"{name} \u2014 Training Return", fontweight="bold")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(algo_dir, "learning_curve.png"), bbox_inches="tight")
        plt.close(fig)

        # 2. Reward with std (mean +/- 1 std band)
        fig, ax = plt.subplots(figsize=(10, 5))
        mean_r = df["ep_return"].rolling(smooth_w_std, min_periods=1).mean()
        std_r = df["ep_return"].rolling(smooth_w_std, min_periods=1).std().fillna(0)
        ax.plot(df["episode"], mean_r, color=color, label=f"Mean (w={smooth_w_std})")
        ax.fill_between(df["episode"], mean_r - std_r, mean_r + std_r,
                        alpha=0.2, color=color, label="\u00b11 std")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title(f"{name} \u2014 Return Stability", fontweight="bold")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(algo_dir, "reward_with_std.png"), bbox_inches="tight")
        plt.close(fig)

        # 3. Steps vs reward (reward vs total env steps)
        fig, ax = plt.subplots(figsize=(10, 5))
        smoothed = df["ep_return"].rolling(smooth_w, min_periods=1).mean()
        steps_k = df["global_step"] / 1000
        ax.plot(steps_k, smoothed, color=color, label="Smoothed")
        ax.set_xlabel("Total Env Steps")
        ax.set_ylabel("Cumulative Reward")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}k"))
        ax.set_title(f"{name} \u2014 Return vs Env Steps", fontweight="bold")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(algo_dir, "steps_vs_reward.png"), bbox_inches="tight")
        plt.close(fig)

        # 4. Episode length curve
        fig, ax = plt.subplots(figsize=(10, 5))
        smoothed_len = df["ep_length"].rolling(smooth_w, min_periods=1).mean()
        ax.plot(df["episode"], smoothed_len, color=color, label=f"Smoothed (w={smooth_w})")
        ax.axhline(y=1000, color="gray", linestyle="--", alpha=0.5, label="Max steps (1000)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps Survived")
        ax.set_title(f"{name} \u2014 Episode Length", fontweight="bold")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(algo_dir, "episode_length_curve.png"), bbox_inches="tight")
        plt.close(fig)

        # 5. Epsilon curve (value-based only)
        if algo in ("Linear_Q", "DQN") and "epsilon" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df["episode"], df["epsilon"], color=color, label="Epsilon")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Epsilon")
            ax.set_title(f"{name} \u2014 Epsilon Decay", fontweight="bold")
            ax.set_ylim(bottom=0, top=1.05)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(algo_dir, "epsilon_curve.png"), bbox_inches="tight")
            plt.close(fig)

        # Loss plots (from losses CSV)
        if loss_df is not None and len(loss_df) > 0:
            loss_smooth = max(1, len(loss_df) // 50)

            # 6. Actor loss curve
            if "actor_loss" in loss_df.columns:
                al = pd.to_numeric(loss_df["actor_loss"], errors="coerce")
                valid = al.dropna()
                if len(valid) > 0:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    smoothed_al = valid.rolling(loss_smooth, min_periods=1).mean()
                    ax.plot(valid.index, smoothed_al, color=color, label="Smoothed")
                    ax.set_xlabel("Update step")
                    ax.set_ylabel("Actor Loss")
                    ax.set_title(f"{name} \u2014 Actor Loss", fontweight="bold")
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(os.path.join(algo_dir, "actor_loss_curve.png"), bbox_inches="tight")
                    plt.close(fig)

            # 7. Critic loss curve
            if "critic_loss" in loss_df.columns:
                cl = pd.to_numeric(loss_df["critic_loss"], errors="coerce")
                valid = cl.dropna()
                if len(valid) > 0:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    smoothed_cl = valid.rolling(loss_smooth, min_periods=1).mean()
                    ax.plot(valid.index, smoothed_cl, color="#FF8C00", label="Smoothed")
                    ax.set_xlabel("Update step")
                    ax.set_ylabel("Critic Loss")
                    ax.set_title(f"{name} \u2014 Critic / Value Loss", fontweight="bold")
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(os.path.join(algo_dir, "critic_loss_curve.png"), bbox_inches="tight")
                    plt.close(fig)

            # 8. Entropy curve
            if "entropy" in loss_df.columns:
                ent = pd.to_numeric(loss_df["entropy"], errors="coerce")
                valid = ent.dropna()
                if len(valid) > 0:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    smoothed_ent = valid.rolling(loss_smooth, min_periods=1).mean()
                    ax.plot(valid.index, smoothed_ent, color=color, label="Smoothed")
                    ax.set_xlabel("Update step")
                    ax.set_ylabel("Entropy")
                    ax.set_title(f"{name} \u2014 Policy Entropy", fontweight="bold")
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(os.path.join(algo_dir, "entropy_curve.png"), bbox_inches="tight")
                    plt.close(fig)

        print(f"  Saved per-algorithm plots for {name}")


# --------------------------------------------------------------------------- #
# Comparison: Actor Loss, Critic Loss, Entropy across algorithms
# --------------------------------------------------------------------------- #
def make_comparison_losses(out):
    """Comparison plots for actor loss, critic loss, and entropy."""
    comp_dir = os.path.join(out, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)

    # Actor loss comparison (actor-critic algorithms only)
    fig_al, ax_al = plt.subplots(figsize=(14, 5))
    fig_cl, ax_cl = plt.subplots(figsize=(14, 5))
    fig_ent, ax_ent = plt.subplots(figsize=(14, 5))
    plotted_al, plotted_cl, plotted_ent = False, False, False

    for algo in ALGOS:
        loss_df = load_losses(algo)
        if loss_df is None or len(loss_df) == 0:
            continue
        color = ALGO_COLORS[algo]
        name = ALGO_DISPLAY[algo]
        smooth = max(1, len(loss_df) // 50)

        if "actor_loss" in loss_df.columns:
            al = pd.to_numeric(loss_df["actor_loss"], errors="coerce").dropna()
            if len(al) > 0:
                smoothed = al.rolling(smooth, min_periods=1).mean()
                ax_al.plot(al.index, smoothed, color=color, label=name)
                plotted_al = True

        if "critic_loss" in loss_df.columns:
            cl = pd.to_numeric(loss_df["critic_loss"], errors="coerce").dropna()
            if len(cl) > 0:
                smoothed = cl.rolling(smooth, min_periods=1).mean()
                ax_cl.plot(cl.index, smoothed, color=color, label=name)
                plotted_cl = True

        if "entropy" in loss_df.columns:
            ent = pd.to_numeric(loss_df["entropy"], errors="coerce").dropna()
            if len(ent) > 0:
                smoothed = ent.rolling(smooth, min_periods=1).mean()
                ax_ent.plot(ent.index, smoothed, color=color, label=name)
                plotted_ent = True

    if plotted_al:
        ax_al.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax_al.set_xlabel("Update step")
        ax_al.set_ylabel("Actor Loss")
        ax_al.set_title("Actor Loss Comparison", fontweight="bold")
        ax_al.legend()
        fig_al.tight_layout()
        fig_al.savefig(os.path.join(comp_dir, "comparison_actor_loss.png"), bbox_inches="tight")
        print("  Saved comparisons/comparison_actor_loss.png")
    plt.close(fig_al)

    if plotted_cl:
        ax_cl.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax_cl.set_xlabel("Update step")
        ax_cl.set_ylabel("Critic Loss")
        ax_cl.set_title("Critic Loss Comparison", fontweight="bold")
        ax_cl.legend()
        fig_cl.tight_layout()
        fig_cl.savefig(os.path.join(comp_dir, "comparison_critic_loss.png"), bbox_inches="tight")
        print("  Saved comparisons/comparison_critic_loss.png")
    plt.close(fig_cl)

    if plotted_ent:
        ax_ent.set_xlabel("Update step")
        ax_ent.set_ylabel("Entropy")
        ax_ent.set_title("Policy Entropy Comparison (Smoothed)", fontweight="bold")
        ax_ent.legend()
        fig_ent.tight_layout()
        fig_ent.savefig(os.path.join(comp_dir, "comparison_entropy.png"), bbox_inches="tight")
        print("  Saved comparisons/comparison_entropy.png")
    plt.close(fig_ent)

    # Also save comparison reward and episode length in comparisons folder
    fig_r, ax_r = plt.subplots(figsize=(14, 5))
    fig_el, ax_el = plt.subplots(figsize=(14, 5))
    for algo in ALGOS:
        df = load_csv(algo)
        if df is None:
            continue
        color = ALGO_COLORS[algo]
        name = ALGO_DISPLAY[algo]
        smooth = max(1, len(df) // 20)
        smoothed_r = df["ep_return"].rolling(smooth, min_periods=1).mean()
        ax_r.plot(df["episode"], smoothed_r, color=color, label=name)
        smoothed_el = df["ep_length"].rolling(smooth, min_periods=1).mean()
        ax_el.plot(df["episode"], smoothed_el, color=color, label=name)

    ax_r.set_xlabel("Episode")
    ax_r.set_ylabel("Cumulative Reward")
    ax_r.set_title("Training Return Comparison", fontweight="bold")
    ax_r.legend()
    fig_r.tight_layout()
    fig_r.savefig(os.path.join(comp_dir, "comparison_reward.png"), bbox_inches="tight")
    plt.close(fig_r)
    print("  Saved comparisons/comparison_reward.png")

    ax_el.set_xlabel("Episode")
    ax_el.set_ylabel("Episode Length")
    ax_el.set_title("Episode Length Comparison", fontweight="bold")
    ax_el.legend()
    fig_el.tight_layout()
    fig_el.savefig(os.path.join(comp_dir, "comparison_ep_length.png"), bbox_inches="tight")
    plt.close(fig_el)
    print("  Saved comparisons/comparison_ep_length.png")


# --------------------------------------------------------------------------- #
# Deployment plots (friend's style)
# --------------------------------------------------------------------------- #
def make_deployment_plots(out):
    """Per-algorithm and comparison deployment plots."""
    deploy_dir = os.path.join(out, "deployment")
    os.makedirs(deploy_dir, exist_ok=True)

    deploy_data = {}
    for algo in ALGOS:
        df = load_deploy(algo)
        if df is not None and "ep_return" in df.columns:
            deploy_data[algo] = df

    if not deploy_data:
        print("  [deployment] No data, skipping.")
        return

    # --- Deployment avg reward bar chart ---
    fig, ax = plt.subplots(figsize=(12, 6))
    names, means, colors_list = [], [], []
    for algo in ALGOS:
        if algo not in deploy_data:
            continue
        df = deploy_data[algo]
        names.append(ALGO_DISPLAY[algo])
        means.append(df["ep_return"].mean())
        colors_list.append(ALGO_COLORS[algo])
    x = np.arange(len(names))
    bars = ax.bar(x, means, color=colors_list, alpha=0.85,
                  edgecolor="black", linewidth=0.8, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, rotation=25)
    ax.set_ylabel("Avg Reward")
    ax.set_title("Deployment Mean Return", fontweight="bold")
    ax.set_ylim(bottom=0)
    for i, m in enumerate(means):
        ax.text(i, m + ax.get_ylim()[1] * 0.01, f"{m:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=colors_list[i])
    fig.tight_layout()
    fig.savefig(os.path.join(deploy_dir, "deployment_reward.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved deployment/deployment_reward.png")

    # --- Per-episode reward scatter ---
    fig, ax = plt.subplots(figsize=(14, 5))
    for algo in ALGOS:
        if algo not in deploy_data:
            continue
        df = deploy_data[algo]
        ax.plot(df["episode"], df["ep_return"], marker=".", markersize=3,
                linewidth=0.8, label=ALGO_DISPLAY[algo], color=ALGO_COLORS[algo],
                alpha=0.8)
    ax.set_xlabel("Deployment Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Deployment Per-Episode Return", fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(deploy_dir, "deployment_reward_per_ep.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved deployment/deployment_reward_per_ep.png")

    # --- Avg episode length bar ---
    fig, ax = plt.subplots(figsize=(12, 6))
    names2, lens2, colors2 = [], [], []
    for algo in ALGOS:
        if algo not in deploy_data:
            continue
        df = deploy_data[algo]
        names2.append(ALGO_DISPLAY[algo])
        lens2.append(df["ep_length"].mean())
        colors2.append(ALGO_COLORS[algo])
    x2 = np.arange(len(names2))
    ax.bar(x2, lens2, color=colors2, alpha=0.85, edgecolor="black", linewidth=0.8, width=0.6)
    ax.set_xticks(x2)
    ax.set_xticklabels(names2, fontsize=10, rotation=25)
    ax.set_ylabel("Avg Episode Length")
    ax.set_title("Deployment Episode Length", fontweight="bold")
    ax.set_ylim(bottom=0)
    for i, l in enumerate(lens2):
        ax.text(i, l + ax.get_ylim()[1] * 0.01, f"{l:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=colors2[i])
    fig.tight_layout()
    fig.savefig(os.path.join(deploy_dir, "deployment_ep_length.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved deployment/deployment_ep_length.png")

    # --- Episode length histogram ---
    fig, ax = plt.subplots(figsize=(14, 5))
    max_len = max(df["ep_length"].max() for df in deploy_data.values())
    for algo in ALGOS:
        if algo not in deploy_data:
            continue
        df = deploy_data[algo]
        ax.hist(df["ep_length"], bins=30, alpha=0.5, label=ALGO_DISPLAY[algo],
                color=ALGO_COLORS[algo], edgecolor="none")
    ax.axvline(x=1000, color="black", linestyle="--", linewidth=1.5, label="Max steps (1000)")
    ax.set_xlabel("Episode Length (steps)")
    ax.set_ylabel("Count")
    ax.set_title("Episode Length Distribution", fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(deploy_dir, "deployment_length_hist.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved deployment/deployment_length_hist.png")



# --------------------------------------------------------------------------- #
# Trajectory plots: state variables during deployment
# --------------------------------------------------------------------------- #
def make_trajectory_plots(out):
    """Plot cart pos, pole angle, cart vel, pole ang vel, and action
    during deployment episodes for each algorithm."""
    traj_dir = os.path.join(out, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    state_vars = [
        ("cart_pos", "Cart Position"),
        ("pole_angle", "Pole Angle (rad)"),
        ("cart_vel", "Cart Velocity"),
        ("pole_ang_vel", "Pole Angular Velocity"),
        ("action", "Action (Force)"),
    ]

    for algo in ALGOS:
        traj_df = load_trajectory(algo)
        if traj_df is None or len(traj_df) == 0:
            continue

        color = ALGO_COLORS[algo]
        name = ALGO_DISPLAY[algo]

        # Pick a few representative episodes (first, middle, last)
        episodes = sorted(traj_df["episode"].unique())
        if len(episodes) == 0:
            continue
        sample_eps = []
        for idx in [0, len(episodes) // 2, -1]:
            if episodes[idx] not in sample_eps:
                sample_eps.append(episodes[idx])

        fig, axes = plt.subplots(len(state_vars), 1, figsize=(12, 3 * len(state_vars)),
                                 sharex=True)
        for ax_i, (col, label) in enumerate(state_vars):
            if col not in traj_df.columns:
                continue
            for ep in sample_eps:
                ep_data = traj_df[traj_df["episode"] == ep]
                axes[ax_i].plot(ep_data["step"], ep_data[col],
                                alpha=0.7, linewidth=0.8, label=f"Ep {ep}")
            axes[ax_i].set_ylabel(label)
            if ax_i == 0:
                axes[ax_i].legend(fontsize=8, ncol=len(sample_eps))
        axes[-1].set_xlabel("Step")
        fig.suptitle(f"{name} \u2014 State Trajectories (Deployment)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(traj_dir, f"{algo}_trajectory.png"), bbox_inches="tight")
        plt.close(fig)

    # Comparison: overlay all algorithms for episode 0
    fig, axes = plt.subplots(len(state_vars), 1, figsize=(14, 3 * len(state_vars)),
                             sharex=True)
    plotted_any = False
    for algo in ALGOS:
        traj_df = load_trajectory(algo)
        if traj_df is None or len(traj_df) == 0:
            continue
        ep0 = traj_df[traj_df["episode"] == 0]
        if len(ep0) == 0:
            continue
        plotted_any = True
        for ax_i, (col, label) in enumerate(state_vars):
            if col not in ep0.columns:
                continue
            axes[ax_i].plot(ep0["step"], ep0[col], color=ALGO_COLORS[algo],
                            label=ALGO_DISPLAY[algo], linewidth=1.0, alpha=0.8)
            axes[ax_i].set_ylabel(label)
    if plotted_any:
        axes[0].legend(fontsize=8, ncol=4)
        axes[-1].set_xlabel("Step")
        fig.suptitle("All Algorithms \u2014 State Trajectories (Episode 0)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(traj_dir, "comparison_trajectory.png"), bbox_inches="tight")
        print("  Saved trajectory plots")
    plt.close(fig)


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
    make_fig6(args.output)
    make_fig7(args.output)
    make_fig8(args.output)
    make_fig9(args.output)
    make_fig11_contrast(args.output, agent_a="PPO", agent_b="TD3")

    # New analysis plots
    make_phase_portrait(args.output)
    make_train_deploy_gap(args.output)
    make_action_histogram(args.output)

    # Per-algorithm plots
    make_per_algo_plots(args.output)

    # Comparison loss/entropy plots
    make_comparison_losses(args.output)

    # Deployment detailed plots
    make_deployment_plots(args.output)

    # State trajectory plots
    make_trajectory_plots(args.output)

    print("Done.")


if __name__ == "__main__":
    main()
