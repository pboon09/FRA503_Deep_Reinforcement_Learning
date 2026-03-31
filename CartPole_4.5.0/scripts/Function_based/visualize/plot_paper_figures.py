#!/usr/bin/env python3
"""Generate 8 publication figures for HW3 report.

Produces:
    fig1_learning_curves.png
    fig2_convergence_speed.png
    fig3_deployment_boxplot.png
    fig4_policy_value_surfaces.png
    fig5_action_traces.png
    fig6_reinforce_variance.png
    fig7_sac_temperature.png
    fig8_capacity_boundary.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Project paths – allow importing RL algorithm classes for surface plots
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "RL_Algorithm"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALGOS = ["Linear_Q", "DQN", "MC_REINFORCE", "AC", "A2C", "PPO", "SAC", "TD3"]

COLORS = {
    "Linear_Q":      "#0072B2",
    "DQN":           "#E69F00",
    "MC_REINFORCE":  "#D55E00",
    "AC":            "#009E73",
    "A2C":           "#CC79A7",
    "PPO":           "#56B4E9",
    "SAC":           "#F0E442",
    "TD3":           "#999999",
}

BATCH_SIZE = 256
MAX_BATCH_STEP = 20_000
CONVERGENCE_THRESHOLD = 900
ROLLING_THIN = 50
ROLLING_THICK = 200

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def _apply_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "axes.grid": False,
    })


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def _load_csv(path: str) -> pd.DataFrame | None:
    if os.path.isfile(path):
        return pd.read_csv(path)
    return None


def load_training(exp_dir: str) -> dict[str, pd.DataFrame]:
    out = {}
    for algo in ALGOS:
        df = _load_csv(os.path.join(exp_dir, f"{algo}.csv"))
        if df is not None:
            out[algo] = df
    return out


def load_losses(exp_dir: str) -> dict[str, pd.DataFrame]:
    out = {}
    for algo in ALGOS:
        df = _load_csv(os.path.join(exp_dir, f"{algo}_losses.csv"))
        if df is not None:
            out[algo] = df
    return out


def load_deploy(exp_dir: str) -> dict[str, pd.DataFrame]:
    out = {}
    for algo in ALGOS:
        df = _load_csv(os.path.join(exp_dir, f"{algo}_deploy.csv"))
        if df is not None:
            out[algo] = df
    return out


def load_trajectories(exp_dir: str) -> dict[str, pd.DataFrame]:
    out = {}
    for algo in ALGOS:
        df = _load_csv(os.path.join(exp_dir, f"{algo}_deploy_trajectory.csv"))
        if df is not None:
            out[algo] = df
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _first_crossing(series: pd.Series, threshold: float) -> int | None:
    """Return index of first value >= threshold, or None."""
    mask = series >= threshold
    if mask.any():
        return mask.idxmax()
    return None


# ===========================================================================
# Figure 1 — Learning Curves
# ===========================================================================
def fig1_learning_curves(train_data: dict, fig_dir: str):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    crossing_info = []

    for algo in ALGOS:
        if algo not in train_data:
            continue
        df = train_data[algo].copy()
        batch_step = df["global_step"].values / BATCH_SIZE
        ret = df["ep_return"].values

        # Rolling averages by episode index
        roll50 = pd.Series(ret).rolling(ROLLING_THIN, min_periods=1).mean().values
        roll200 = pd.Series(ret).rolling(ROLLING_THICK, min_periods=1).mean().values

        c = COLORS[algo]
        ax.plot(batch_step, roll50, color=c, alpha=0.5, linewidth=0.8)
        ax.plot(batch_step, roll200, color=c, linewidth=1.8, label=algo)

        # Convergence: first rolling-50 >= threshold
        idx = _first_crossing(pd.Series(roll50), CONVERGENCE_THRESHOLD)
        if idx is not None and idx < len(batch_step):
            bs = batch_step[idx]
            if bs <= MAX_BATCH_STEP:
                crossing_info.append((algo, bs))

    # Vertical dashed lines for convergence
    for algo, bs in crossing_info:
        ax.axvline(bs, color=COLORS[algo], linestyle="--", linewidth=0.7, alpha=0.6)
        ax.annotate(
            f"{int(bs)}",
            xy=(bs, ax.get_ylim()[1]),
            xytext=(0, 2),
            textcoords="offset points",
            fontsize=6,
            rotation=45,
            ha="left",
            va="bottom",
            color=COLORS[algo],
        )

    ax.set_xlim(0, MAX_BATCH_STEP)
    ax.set_xlabel("Batch step (global_step / 256)")
    ax.set_ylabel("Episode return")
    ax.set_title("Fig 1 — Learning Curves")
    ax.legend(ncol=2, fontsize=8, loc="lower right")

    path = os.path.join(fig_dir, "fig1_learning_curves.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ===========================================================================
# Figure 2 — Convergence Speed Bar Chart
# ===========================================================================
def fig2_convergence_speed(train_data: dict, fig_dir: str):
    records = []
    for algo in ALGOS:
        if algo not in train_data:
            continue
        df = train_data[algo].copy()
        batch_step = df["global_step"].values / BATCH_SIZE
        ret = df["ep_return"].values
        roll50 = pd.Series(ret).rolling(ROLLING_THIN, min_periods=1).mean().values

        idx = _first_crossing(pd.Series(roll50), CONVERGENCE_THRESHOLD)
        if idx is not None and idx < len(batch_step) and batch_step[idx] <= MAX_BATCH_STEP:
            records.append({"algo": algo, "bs": batch_step[idx], "dnf": False})
        else:
            records.append({"algo": algo, "bs": MAX_BATCH_STEP, "dnf": True})

    records.sort(key=lambda r: r["bs"])

    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    y_pos = np.arange(len(records))
    bars_bs = [r["bs"] for r in records]
    bar_colors = [COLORS[r["algo"]] for r in records]
    hatches = ["//" if r["dnf"] else "" for r in records]

    bars = ax.barh(y_pos, bars_bs, color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["algo"] for r in records])
    ax.set_xlabel("Batch step at first rolling-50 >= 900")
    ax.set_title("Fig 2 — Convergence Speed")

    for i, r in enumerate(records):
        label = "DNF" if r["dnf"] else f"{int(r['bs'])}"
        ax.text(r["bs"] + 100, i, label, va="center", fontsize=8)

    path = os.path.join(fig_dir, "fig2_convergence_speed.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ===========================================================================
# Figure 3 — Deployment Boxplot
# ===========================================================================
def fig3_deployment_boxplot(deploy_data: dict, fig_dir: str):
    # Compute medians and sort descending
    algo_medians = []
    for algo in ALGOS:
        if algo not in deploy_data:
            continue
        med = deploy_data[algo]["ep_return"].median()
        algo_medians.append((algo, med))
    algo_medians.sort(key=lambda x: -x[1])

    ordered_algos = [a for a, _ in algo_medians]
    data_lists = [deploy_data[a]["ep_return"].values for a in ordered_algos]

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    bp = ax.boxplot(
        data_lists,
        vert=True,
        patch_artist=True,
        showfliers=False,
        widths=0.5,
        medianprops=dict(color="black", linewidth=1.5),
    )

    # Color by tier
    for i, (algo, med) in enumerate(algo_medians):
        if med > 990:
            fc = "#a8d5a2"  # green
        elif med >= 300:
            fc = "#fff4a3"  # yellow
        else:
            fc = "#f4a3a3"  # red
        bp["boxes"][i].set_facecolor(fc)
        bp["boxes"][i].set_edgecolor("black")

    # Jittered scatter overlay
    for i, (algo, _) in enumerate(algo_medians):
        vals = deploy_data[algo]["ep_return"].values
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            np.full_like(vals, i + 1, dtype=float) + jitter,
            vals,
            color=COLORS[algo],
            alpha=0.3,
            s=15,
            zorder=3,
        )

    ax.axhline(999, color="gray", linestyle="--", linewidth=0.8, label="near-perfect (999)")
    ax.set_xticks(range(1, len(ordered_algos) + 1))
    ax.set_xticklabels(ordered_algos, rotation=30, ha="right")
    ax.set_ylabel("Episode return")
    ax.set_title("Fig 3 — Deployment Performance")
    ax.legend(fontsize=8)

    path = os.path.join(fig_dir, "fig3_deployment_boxplot.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ===========================================================================
# Figure 4 — Policy & Value Surfaces
# ===========================================================================
def fig4_policy_value_surfaces(model_dir: str, fig_dir: str):
    try:
        import torch
    except ImportError:
        print("  [WARN] torch not available, skipping fig4")
        return None

    device = torch.device("cpu")
    RES = 100
    theta = np.linspace(-0.4, 0.4, RES)
    theta_dot = np.linspace(-3.0, 3.0, RES)
    TH, TD = np.meshgrid(theta, theta_dot)  # shape (RES, RES)

    obs_np = np.zeros((RES * RES, 4), dtype=np.float32)
    obs_np[:, 0] = 0.0       # cart_pos
    obs_np[:, 1] = TH.ravel()  # pole_angle
    obs_np[:, 2] = 0.0       # cart_vel
    obs_np[:, 3] = TD.ravel()  # pole_ang_vel
    obs_t = torch.from_numpy(obs_np).to(device)

    surface_algos = ["PPO", "SAC", "DQN"]
    policy_maps = {}
    value_maps = {}

    # --- PPO ---
    try:
        from RL_Algorithm.Function_based.AC import ActorCritic
        policy = ActorCritic(
            state_dim=4, action_dim=1, hidden_dims=[256, 256],
            activation="elu", action_type="continuous", init_noise_std=1.0,
        ).to(device)
        ppo_path = os.path.join(model_dir, "PPO", "PPO_final.pth")
        policy.load_state_dict(torch.load(ppo_path, map_location=device, weights_only=True))
        policy.eval()
        with torch.no_grad():
            p = policy.actor(obs_t).cpu().numpy().reshape(RES, RES)
            v = policy.critic(obs_t).cpu().numpy().reshape(RES, RES)
        policy_maps["PPO"] = p
        value_maps["PPO"] = v
    except Exception as e:
        print(f"  [WARN] PPO surface failed: {e}")

    # --- SAC ---
    try:
        from RL_Algorithm.Function_based.SAC import SAC_Actor, SAC_Critic
        actor = SAC_Actor(n_observations=4, hidden_dim=256, n_actions=1).to(device)
        critic = SAC_Critic(n_observations=4, n_actions=1, hidden_dim=256).to(device)
        sac_path = os.path.join(model_dir, "SAC", "SAC_final.pth")
        ckpt = torch.load(sac_path, map_location=device, weights_only=True)
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        actor.eval()
        critic.eval()
        with torch.no_grad():
            mean, _ = actor(obs_t)
            action_for_v = torch.tanh(mean)
            q1, q2 = critic(obs_t, action_for_v)
            v = torch.min(q1, q2)
        policy_maps["SAC"] = mean.cpu().numpy().reshape(RES, RES)
        value_maps["SAC"] = v.cpu().numpy().reshape(RES, RES)
    except Exception as e:
        print(f"  [WARN] SAC surface failed: {e}")

    # --- DQN ---
    try:
        from RL_Algorithm.Function_based.DQN import DQN_network
        net = DQN_network(4, 128, 11, 0.0).to(device)
        dqn_path = os.path.join(model_dir, "DQN", "DQN_final.pth")
        net.load_state_dict(torch.load(dqn_path, map_location=device, weights_only=True))
        net.eval()
        # Map 11 discrete indices to action values in [-2.5, 2.5]
        action_values = np.linspace(-2.5, 2.5, 11)
        with torch.no_grad():
            Q = net(obs_t).cpu().numpy()  # (N, 11)
        best_idx = Q.argmax(axis=1)
        policy_maps["DQN"] = action_values[best_idx].reshape(RES, RES)
        value_maps["DQN"] = Q.max(axis=1).reshape(RES, RES)
    except Exception as e:
        print(f"  [WARN] DQN surface failed: {e}")

    # --- Plotting ---
    available = [a for a in surface_algos if a in policy_maps]
    if not available:
        print("  [WARN] No models loaded for fig4, skipping.")
        return None

    ncols = len(available)
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 6), constrained_layout=True)
    if ncols == 1:
        axes = axes.reshape(2, 1)

    # Policy row
    p_vmin = min(policy_maps[a].min() for a in available)
    p_vmax = max(policy_maps[a].max() for a in available)
    p_abs = max(abs(p_vmin), abs(p_vmax))

    v_vmin = min(value_maps[a].min() for a in available)
    v_vmax = max(value_maps[a].max() for a in available)

    for j, algo in enumerate(available):
        im_p = axes[0, j].pcolormesh(
            theta, theta_dot, policy_maps[algo],
            cmap="RdBu", vmin=-p_abs, vmax=p_abs, shading="auto",
        )
        axes[0, j].set_title(f"{algo} policy")
        axes[0, j].set_xlabel(r"$\theta$ (rad)")
        axes[0, j].set_ylabel(r"$\dot{\theta}$ (rad/s)")

        im_v = axes[1, j].pcolormesh(
            theta, theta_dot, value_maps[algo],
            cmap="viridis", vmin=v_vmin, vmax=v_vmax, shading="auto",
        )
        axes[1, j].set_title(f"{algo} value")
        axes[1, j].set_xlabel(r"$\theta$ (rad)")
        axes[1, j].set_ylabel(r"$\dot{\theta}$ (rad/s)")

    fig.colorbar(im_p, ax=axes[0, :].tolist(), label="Action", shrink=0.8)
    fig.colorbar(im_v, ax=axes[1, :].tolist(), label="Value", shrink=0.8)

    fig.suptitle("Fig 4 — Policy & Value Surfaces", fontsize=13)
    path = os.path.join(fig_dir, "fig4_policy_value_surfaces.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ===========================================================================
# Figure 5 — Action Trace Comparison
# ===========================================================================
def fig5_action_traces(traj_data: dict, fig_dir: str):
    if "DQN" not in traj_data or "PPO" not in traj_data:
        print("  [WARN] Missing DQN or PPO trajectory data, skipping fig5")
        return None

    # Pick episode: for PPO pick first episode with length 1000, for DQN any
    def _pick_episode(df, prefer_len=None):
        for ep in df["episode"].unique():
            sub = df[df["episode"] == ep]
            if prefer_len is not None and len(sub) >= prefer_len:
                return sub.reset_index(drop=True)
        return df[df["episode"] == df["episode"].iloc[0]].reset_index(drop=True)

    dqn_ep = _pick_episode(traj_data["DQN"])
    ppo_ep = _pick_episode(traj_data["PPO"], prefer_len=1000)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    # Subplot 1 — DQN action
    ax1 = axes[0]
    ax1.plot(dqn_ep["step"], dqn_ep["action"], color=COLORS["DQN"], linewidth=0.8)
    disc_levels = np.linspace(-2.5, 2.5, 11)
    for lv in disc_levels:
        ax1.axhline(lv, color="gray", linestyle=":", linewidth=0.3, alpha=0.5)
    ax1.set_ylabel("Force (N)")
    ax1.set_title("DQN action trace (discrete)")

    # Subplot 2 — PPO action
    ax2 = axes[1]
    ax2.plot(ppo_ep["step"], ppo_ep["action"], color=COLORS["PPO"], linewidth=0.8)
    ax2.set_ylabel("Force (N)")
    ax2.set_title("PPO action trace (continuous)")

    # Subplot 3 — Pole angle overlay
    ax3 = axes[2]
    ax3.plot(dqn_ep["step"], dqn_ep["pole_angle"], color=COLORS["DQN"],
             linestyle="--", linewidth=0.8, label="DQN")
    ax3.plot(ppo_ep["step"], ppo_ep["pole_angle"], color=COLORS["PPO"],
             linestyle="--", linewidth=0.8, label="PPO")

    # RMS of theta for last 500 steps
    def _rms_tail(arr, n=500):
        tail = arr[-n:] if len(arr) >= n else arr
        return np.sqrt(np.mean(tail ** 2))

    rms_dqn = _rms_tail(dqn_ep["pole_angle"].values)
    rms_ppo = _rms_tail(ppo_ep["pole_angle"].values)
    ax3.annotate(f"DQN RMS={rms_dqn:.4f}", xy=(0.02, 0.92), xycoords="axes fraction",
                 fontsize=8, color=COLORS["DQN"])
    ax3.annotate(f"PPO RMS={rms_ppo:.4f}", xy=(0.02, 0.82), xycoords="axes fraction",
                 fontsize=8, color=COLORS["PPO"])
    ax3.set_ylabel(r"$\theta$ (rad)")
    ax3.set_xlabel("Timestep")
    ax3.set_title("Pole angle comparison")
    ax3.legend(fontsize=8)

    fig.suptitle("Fig 5 — Action Trace Comparison", fontsize=13)
    path = os.path.join(fig_dir, "fig5_action_traces.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ===========================================================================
# Figure 6 — REINFORCE Instability Analysis
# ===========================================================================
def fig6_reinforce_variance(train_data: dict, fig_dir: str):
    if "MC_REINFORCE" not in train_data:
        print("  [WARN] MC_REINFORCE data missing, skipping fig6")
        return None

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)

    df_r = train_data["MC_REINFORCE"].copy()
    bs_r = df_r["global_step"].values / BATCH_SIZE
    ret_r = df_r["ep_return"].values

    # Scatter of raw returns
    ax.scatter(bs_r, ret_r, color=COLORS["MC_REINFORCE"], alpha=0.05, s=3, rasterized=True)

    roll50_r = pd.Series(ret_r).rolling(ROLLING_THIN, min_periods=1)
    mean_r = roll50_r.mean().values
    std_r = roll50_r.std().values
    std_r = np.nan_to_num(std_r, nan=0.0)

    ax.plot(bs_r, mean_r, color=COLORS["MC_REINFORCE"], linewidth=1.5, label="REINFORCE rolling-50")
    ax.fill_between(bs_r, mean_r - std_r, mean_r + std_r,
                    color=COLORS["MC_REINFORCE"], alpha=0.15)

    # PPO reference
    if "PPO" in train_data:
        df_p = train_data["PPO"].copy()
        bs_p = df_p["global_step"].values / BATCH_SIZE
        ret_p = df_p["ep_return"].values
        roll50_p = pd.Series(ret_p).rolling(ROLLING_THIN, min_periods=1)
        mean_p = roll50_p.mean().values
        std_p = roll50_p.std().values
        std_p = np.nan_to_num(std_p, nan=0.0)

        ax.plot(bs_p, mean_p, color=COLORS["PPO"], linewidth=1.0, linestyle="--", label="PPO rolling-50")

        # Std ratio at midpoint
        mid_r_idx = len(ret_r) // 2
        mid_p_idx = len(ret_p) // 2
        sigma_r = std_r[mid_r_idx] if mid_r_idx < len(std_r) else np.nan
        sigma_p = std_p[mid_p_idx] if mid_p_idx < len(std_p) else np.nan
        if sigma_p > 0:
            ratio = sigma_r / sigma_p
            mid_bs = bs_r[mid_r_idx]
            ax.annotate(
                f"$\\sigma_{{RE}}/\\sigma_{{PPO}}$ = {ratio:.1f}",
                xy=(mid_bs, mean_r[mid_r_idx]),
                xytext=(mid_bs + 500, mean_r[mid_r_idx] + 100),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"),
            )

    ax.set_xlim(0, MAX_BATCH_STEP)
    ax.set_xlabel("Batch step")
    ax.set_ylabel("Episode return")
    ax.set_title("Fig 6 — REINFORCE Instability Analysis")
    ax.legend(fontsize=8)

    path = os.path.join(fig_dir, "fig6_reinforce_variance.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ===========================================================================
# Figure 7 — SAC Temperature Dynamics
# ===========================================================================
def fig7_sac_temperature(loss_data: dict, fig_dir: str):
    if "SAC" not in loss_data:
        print("  [WARN] SAC loss data missing, skipping fig7")
        return None

    df = loss_data["SAC"].copy()

    if "alpha" not in df.columns:
        print("  [WARN] SAC losses CSV has no 'alpha' column, skipping fig7")
        return None

    # Subsample for readability
    step = max(1, len(df) // 2000)
    df_sub = df.iloc[::step].copy()

    fig, ax1 = plt.subplots(figsize=(8, 3.5), constrained_layout=True)

    color_alpha = "#9467BD"   # purple
    color_entropy = "#17BECF" # teal

    ax1.plot(df_sub["update_step"], df_sub["alpha"], color=color_alpha,
             linewidth=1.0, label=r"SAC $\alpha$")
    ax1.set_xlabel("Update step")
    ax1.set_ylabel(r"$\alpha$ (temperature)", color=color_alpha)
    ax1.tick_params(axis="y", labelcolor=color_alpha)

    # TD3 fixed sigma reference
    ax1.axhline(0.1, color=COLORS["TD3"], linestyle=":", linewidth=1.0, label=r"TD3 $\sigma$=0.1")

    ax2 = ax1.twinx()
    if "entropy" in df_sub.columns:
        ax2.plot(df_sub["update_step"], df_sub["entropy"], color=color_entropy,
                 linewidth=1.0, linestyle="--", label="Policy entropy")
    ax2.set_ylabel("Entropy", color=color_entropy)
    ax2.tick_params(axis="y", labelcolor=color_entropy)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    ax1.set_title("Fig 7 — SAC Temperature Dynamics")

    path = os.path.join(fig_dir, "fig7_sac_temperature.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ===========================================================================
# Figure 8 — Capacity Boundary Bar Chart
# ===========================================================================
def fig8_capacity_boundary(deploy_data: dict, fig_dir: str):
    # Linear
    linear_vals = deploy_data.get("Linear_Q")
    dqn_vals = deploy_data.get("DQN")

    if linear_vals is None or dqn_vals is None:
        print("  [WARN] Missing Linear_Q or DQN deploy data, skipping fig8")
        return None

    linear_mean = linear_vals["ep_return"].mean()
    linear_std = linear_vals["ep_return"].std()

    dqn_mean = dqn_vals["ep_return"].mean()
    dqn_std = dqn_vals["ep_return"].std()

    # Deep + Continuous: mean of algorithm means
    cont_algos = ["SAC", "PPO", "AC", "A2C", "TD3"]
    cont_means = []
    for a in cont_algos:
        if a in deploy_data:
            cont_means.append(deploy_data[a]["ep_return"].mean())
    if not cont_means:
        print("  [WARN] No continuous algo deploy data, skipping fig8")
        return None

    cont_group_mean = np.mean(cont_means)
    cont_group_std = np.std(cont_means)

    labels = ["Linear", "Deep + Discrete", "Deep + Continuous"]
    means = [linear_mean, dqn_mean, cont_group_mean]
    stds = [linear_std, dqn_std, cont_group_std]
    bar_colors = [COLORS["Linear_Q"], COLORS["DQN"], "#56B4E9"]

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    # Use log scale if ratio is large
    use_log = (max(means) / max(min(means), 1e-3)) > 50

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors,
                  edgecolor="black", linewidth=0.5)

    if use_log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean deployment return")
    ax.set_title("Fig 8 — Capacity Boundary")

    # Annotate multiplication factors
    for i in range(len(means) - 1):
        if means[i] > 0:
            factor = means[i + 1] / means[i]
            mid_x = (x[i] + x[i + 1]) / 2
            mid_y = max(means[i], means[i + 1]) * (1.15 if not use_log else 1.5)
            ax.annotate(
                f"x{factor:.0f}" if factor >= 1 else f"x{factor:.1f}",
                xy=(mid_x, mid_y),
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

    path = os.path.join(fig_dir, "fig8_capacity_boundary.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ===========================================================================
# Main
# ===========================================================================
def main():
    _apply_style()

    default_root = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
    default_exp = os.path.join(default_root, "experiments", "suite_1_baseline")
    default_fig = os.path.join(default_root, "figures")
    default_model = os.path.join(default_root, "model", "Stabilize")

    parser = argparse.ArgumentParser(description="Generate 8 publication figures for HW3.")
    parser.add_argument("--exp-dir", default=default_exp, help="Path to experiment CSVs")
    parser.add_argument("--fig-dir", default=default_fig, help="Output directory for figures")
    parser.add_argument("--model-dir", default=default_model, help="Path to saved models")
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    print(f"Experiment dir : {args.exp_dir}")
    print(f"Figure dir     : {args.fig_dir}")
    print(f"Model dir      : {args.model_dir}")
    print()

    # Load all data once
    train_data = load_training(args.exp_dir)
    loss_data = load_losses(args.exp_dir)
    deploy_data = load_deploy(args.exp_dir)
    traj_data = load_trajectories(args.exp_dir)

    print(f"Loaded training data for: {list(train_data.keys())}")
    print(f"Loaded deploy data for  : {list(deploy_data.keys())}")
    print()

    generated = []

    # Fig 1
    print("Generating fig1_learning_curves ...")
    p = fig1_learning_curves(train_data, args.fig_dir)
    if p:
        generated.append(p)

    # Fig 2
    print("Generating fig2_convergence_speed ...")
    p = fig2_convergence_speed(train_data, args.fig_dir)
    if p:
        generated.append(p)

    # Fig 3
    print("Generating fig3_deployment_boxplot ...")
    p = fig3_deployment_boxplot(deploy_data, args.fig_dir)
    if p:
        generated.append(p)

    # Fig 4
    print("Generating fig4_policy_value_surfaces ...")
    p = fig4_policy_value_surfaces(args.model_dir, args.fig_dir)
    if p:
        generated.append(p)

    # Fig 5
    print("Generating fig5_action_traces ...")
    p = fig5_action_traces(traj_data, args.fig_dir)
    if p:
        generated.append(p)

    # Fig 6
    print("Generating fig6_reinforce_variance ...")
    p = fig6_reinforce_variance(train_data, args.fig_dir)
    if p:
        generated.append(p)

    # Fig 7
    print("Generating fig7_sac_temperature ...")
    p = fig7_sac_temperature(loss_data, args.fig_dir)
    if p:
        generated.append(p)

    # Fig 8
    print("Generating fig8_capacity_boundary ...")
    p = fig8_capacity_boundary(deploy_data, args.fig_dir)
    if p:
        generated.append(p)

    # Summary
    print()
    print("=" * 60)
    print(f"Generated {len(generated)} / 8 figures:")
    for p in generated:
        print(f"  {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
