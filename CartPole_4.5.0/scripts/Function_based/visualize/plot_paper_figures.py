#!/usr/bin/env python3
"""Generate publication figures for HW3 report.

Produces:
    fig1_learning_curves.png   — 1x2 split: value-based (left) vs actor-critic (right)
    fig2_convergence_speed.png — horizontal bar chart (env steps to 900)
    fig3_deployment_boxplot.png — violin/box deployment returns
    fig4_policy_value_surfaces.png — PPO vs TD3 policy & value (needs torch)
    fig5_action_traces.png     — DQN vs PPO action + pole angle
    fig6_reinforce_variance.png — REINFORCE vs PPO training comparison
    fig7_sac_temperature.png   — SAC alpha over training
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "RL_Algorithm"))

# ---------------------------------------------------------------------------
ALGOS = ["Linear_Q", "DQN", "MC_REINFORCE", "AC", "A2C", "PPO", "SAC", "TD3"]

LABELS = {
    "Linear_Q": "Linear Q", "DQN": "DQN", "MC_REINFORCE": "MC REINFORCE",
    "AC": "Actor-Critic", "A2C": "A2C", "PPO": "PPO", "SAC": "SAC", "TD3": "TD3",
}

# Bold, high-contrast palette for both panels
COLORS = {
    "Linear_Q":     "#1f77b4",   # blue
    "DQN":          "#ff7f0e",   # orange
    "MC_REINFORCE": "#2ca02c",   # green
    "AC":           "#d62728",   # red
    "A2C":          "#e6550d",   # dark orange
    "PPO":          "#1a9850",   # dark green
    "SAC":          "#7570b3",   # indigo
    "TD3":          "#e7298a",   # magenta
}

LEFT_GROUP = ["Linear_Q", "DQN", "MC_REINFORCE", "AC"]
RIGHT_GROUP = ["A2C", "PPO", "SAC", "TD3"]

BATCH_SIZE = 256
CONVERGENCE_THRESHOLD = 900
ROLLING_W = 50


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def _apply_style():
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "figure.dpi": 150, "savefig.dpi": 150, "axes.grid": False,
    })


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def _load_csv(path):
    if os.path.isfile(path):
        return pd.read_csv(path)
    return None


def load_all(exp_dir):
    train, losses, deploy, traj = {}, {}, {}, {}
    for algo in ALGOS:
        for store, suffix in [(train, ""), (losses, "_losses"), (deploy, "_deploy"), (traj, "_deploy_trajectory")]:
            df = _load_csv(os.path.join(exp_dir, f"{algo}{suffix}.csv"))
            if df is not None:
                store[algo] = df
    return train, losses, deploy, traj


def _rolling(ret, w=ROLLING_W):
    s = pd.Series(ret)
    return s.rolling(w, min_periods=1).mean().values, s.rolling(w, min_periods=1).std().fillna(0).values


# ===========================================================================
# Fig 1 — Learning Curves (1x2 split)
# ===========================================================================
def fig1_learning_curves(train_data, fig_dir):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))

    for group, ax, title in [(LEFT_GROUP, ax_l, "Value-Based & Policy Gradient"),
                              (RIGHT_GROUP, ax_r, "Actor-Critic (Advanced)")]:
        for algo in group:
            if algo not in train_data:
                continue
            df = train_data[algo]
            batch_step = df["global_step"].values / BATCH_SIZE
            ret = df["ep_return"].values
            mean, std = _rolling(ret)
            c = COLORS[algo]
            ax.plot(batch_step, mean, color=c, linewidth=1.5, label=LABELS[algo])
            ax.fill_between(batch_step, mean - std, mean + std, alpha=0.15, color=c)

        ax.set_xlabel("Batch Step (x256 envs)")
        ax.set_ylabel("Episode Return")
        ax.set_title(title)
        ax.set_xlim(0, 20000)
        ax.set_ylim(-50, 1100)
        ax.legend(fontsize=9, loc="center right")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    path = os.path.join(fig_dir, "fig1_learning_curves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ===========================================================================
# Fig 2 — Convergence Speed
# ===========================================================================
def fig2_convergence_speed(train_data, fig_dir):
    records = []
    for algo in ALGOS:
        if algo not in train_data:
            continue
        df = train_data[algo]
        ret = df["ep_return"].values
        batch_step = df["global_step"].values / BATCH_SIZE
        mean, _ = _rolling(ret)
        found = None
        for i in range(len(mean)):
            if mean[i] >= CONVERGENCE_THRESHOLD:
                found = int(batch_step[i])
                break
        records.append({"algo": algo, "step": found})

    converged = sorted([r for r in records if r["step"] is not None], key=lambda r: r["step"])
    dnf = [r for r in records if r["step"] is None]
    ordered = converged + dnf
    max_step = 20000

    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    y_pos = range(len(ordered))
    vals = [r["step"] if r["step"] else max_step for r in ordered]
    bar_colors = [COLORS[r["algo"]] for r in ordered]

    ax.barh(y_pos, vals, color=bar_colors, height=0.6)
    for i, r in enumerate(ordered):
        if r["step"]:
            ax.text(r["step"] + max_step * 0.02, i, f'{r["step"]:,}', va="center", fontsize=8)
        else:
            ax.text(max_step * 0.5, i, "DNF", va="center", ha="center",
                    fontsize=9, fontweight="bold", color="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([LABELS[r["algo"]] for r in ordered], fontsize=9)
    ax.set_xlabel("Batch Step to Reach 900 (rolling-50)")
    ax.set_title("Convergence Speed")
    ax.set_xlim(0, max_step)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis="x")

    path = os.path.join(fig_dir, "fig2_convergence_speed.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ===========================================================================
# Fig 3 — Deployment Boxplot (compact height)
# ===========================================================================
def fig3_deployment_boxplot(deploy_data, fig_dir):
    algo_medians = []
    for algo in ALGOS:
        if algo in deploy_data:
            med = deploy_data[algo]["ep_return"].median()
            algo_medians.append((algo, med))
    algo_medians.sort(key=lambda x: -x[1])

    ordered = [a for a, _ in algo_medians]
    data_lists = [deploy_data[a]["ep_return"].values for a in ordered]

    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    bp = ax.boxplot(data_lists, vert=True, patch_artist=True, showfliers=False, widths=0.5,
                    medianprops=dict(color="black", linewidth=1.5))

    for i, (algo, med) in enumerate(algo_medians):
        fc = "#a8d5a2" if med > 990 else ("#fff4a3" if med >= 300 else "#f4a3a3")
        bp["boxes"][i].set_facecolor(fc)
        bp["boxes"][i].set_edgecolor("black")

    # Jitter scatter
    rng = np.random.default_rng(42)
    for i, (algo, _) in enumerate(algo_medians):
        vals = deploy_data[algo]["ep_return"].values
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full_like(vals, i + 1, dtype=float) + jitter, vals,
                   color=COLORS[algo], alpha=0.3, s=12, zorder=3)

    ax.set_xticks(range(1, len(ordered) + 1))
    ax.set_xticklabels([LABELS[a] for a in ordered], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Episode Return")
    ax.set_title("Deployment Performance (100 Episodes)")

    path = os.path.join(fig_dir, "fig3_deployment_boxplot.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ===========================================================================
# Fig 4 — Policy & Value Surfaces (PPO vs TD3)
# ===========================================================================
def fig4_policy_value_surfaces(model_dir, fig_dir):
    try:
        import torch
    except ImportError:
        print("  [WARN] torch not available, skipping fig4")
        return None

    device = torch.device("cpu")
    RES = 80
    theta = np.linspace(-0.3, 0.3, RES)
    theta_dot = np.linspace(-3.0, 3.0, RES)
    TH, TD = np.meshgrid(theta, theta_dot)

    obs_np = np.zeros((RES * RES, 4), dtype=np.float32)
    obs_np[:, 1] = TH.ravel()
    obs_np[:, 3] = TD.ravel()
    obs_t = torch.from_numpy(obs_np).to(device)

    surfaces = {}  # algo -> (policy_map, value_map)

    # PPO
    try:
        from RL_Algorithm.Function_based.AC import ActorCritic
        policy = ActorCritic(state_dim=4, action_dim=1, hidden_dims=[256, 256],
                             activation="elu", action_type="continuous", init_noise_std=1.0).to(device)
        policy.load_state_dict(torch.load(
            os.path.join(model_dir, "PPO", "PPO_final.pth"), map_location=device, weights_only=True))
        policy.eval()
        with torch.no_grad():
            p = policy.actor(obs_t).cpu().numpy().reshape(RES, RES)
            v = policy.critic(obs_t).cpu().numpy().reshape(RES, RES)
        surfaces["PPO"] = (p, v)
    except Exception as e:
        print(f"  [WARN] PPO surface failed: {e}")

    # TD3
    try:
        from RL_Algorithm.Function_based.TD3 import TD3_Actor, TD3_Critic
        actor = TD3_Actor(4, 256, 1).to(device)
        critic = TD3_Critic(4, 1, 256).to(device)
        ckpt = torch.load(os.path.join(model_dir, "TD3", "TD3_final.pth"),
                          map_location=device, weights_only=True)
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        actor.eval(); critic.eval()
        with torch.no_grad():
            a_raw = actor(obs_t)
            p = (-2.5 + (a_raw + 1.0) * 0.5 * 5.0).cpu().numpy().reshape(RES, RES)
            q1, q2 = critic(obs_t, a_raw)
            v = torch.min(q1, q2).cpu().numpy().reshape(RES, RES)
        surfaces["TD3"] = (p, v)
    except Exception as e:
        print(f"  [WARN] TD3 surface failed: {e}")

    available = [a for a in ["PPO", "TD3"] if a in surfaces]
    if not available:
        print("  [WARN] No models for fig4")
        return None

    fig, axes = plt.subplots(2, len(available), figsize=(5 * len(available), 8),
                             subplot_kw={"projection": "3d"})
    if len(available) == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(r"Policy and Value Surfaces (cart_pos=0, cart_vel=0, varying $\theta$ and $\dot{\theta}$)",
                 fontsize=13, y=0.98)

    for col, algo in enumerate(available):
        pm, vm = surfaces[algo]
        ax_p = axes[0, col]
        ax_p.plot_surface(TH, TD, pm, cmap="RdBu_r", alpha=0.85, edgecolor="none")
        ax_p.set_xlabel(r"$\theta$", fontsize=9)
        ax_p.set_ylabel(r"$\dot{\theta}$", fontsize=9)
        ax_p.set_zlabel("Action (m/s)", fontsize=9)
        ax_p.set_title(f"{algo}: Policy Surface", fontsize=11)

        ax_v = axes[1, col]
        ax_v.plot_surface(TH, TD, vm, cmap="viridis", alpha=0.85, edgecolor="none")
        ax_v.set_xlabel(r"$\theta$", fontsize=9)
        ax_v.set_ylabel(r"$\dot{\theta}$", fontsize=9)
        ax_v.set_zlabel("V(s)", fontsize=9)
        ax_v.set_title(f"{algo}: Value Surface", fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(fig_dir, "fig4_policy_value_surfaces.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ===========================================================================
# Fig 5 — Action Traces (DQN vs PPO: action + pole angle)
# ===========================================================================
def fig5_action_traces(traj_data, fig_dir):
    if "DQN" not in traj_data or "PPO" not in traj_data:
        print("  [WARN] Missing DQN or PPO trajectory data, skipping fig5")
        return None

    def _pick_ep0(df):
        return df[df["episode"] == df["episode"].unique()[0]].reset_index(drop=True)

    dqn_ep = _pick_ep0(traj_data["DQN"])
    ppo_ep = _pick_ep0(traj_data["PPO"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)

    # Top: Action trace comparison (both on same plot)
    ax1.plot(dqn_ep["step"], dqn_ep["action"], color=COLORS["DQN"],
             linewidth=0.6, alpha=0.8, label="DQN (discrete)")
    ax1.plot(ppo_ep["step"], ppo_ep["action"], color=COLORS["PPO"],
             linewidth=1.2, label="PPO (continuous)")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.set_title("Action Trace Comparison")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bottom: Pole angle comparison
    def _rms(arr, n=500):
        tail = arr[-n:] if len(arr) >= n else arr
        return np.sqrt(np.mean(tail ** 2))

    rms_dqn = _rms(dqn_ep["pole_angle"].values)
    rms_ppo = _rms(ppo_ep["pole_angle"].values)

    ax2.plot(dqn_ep["step"], dqn_ep["pole_angle"], color=COLORS["DQN"],
             linewidth=1.2, label=f"DQN (RMS={rms_dqn:.4f})")
    ax2.plot(ppo_ep["step"], ppo_ep["pole_angle"], color=COLORS["PPO"],
             linewidth=1.2, linestyle="--", label=f"PPO (RMS={rms_ppo:.4f})")
    ax2.set_ylabel(r"$\theta$ (rad)")
    ax2.set_xlabel("Timestep")
    ax2.set_title("Pole Angle Comparison")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    path = os.path.join(fig_dir, "fig5_action_traces.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ===========================================================================
# Fig 6 — REINFORCE vs PPO
# ===========================================================================
def fig6_reinforce_variance(train_data, fig_dir):
    if "MC_REINFORCE" not in train_data:
        print("  [WARN] MC_REINFORCE data missing, skipping fig6")
        return None

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)

    for algo, color in [("MC_REINFORCE", COLORS["MC_REINFORCE"]), ("PPO", COLORS["PPO"])]:
        if algo not in train_data:
            continue
        df = train_data[algo]
        batch_step = df["global_step"].values / BATCH_SIZE
        ret = df["ep_return"].values
        mean, std = _rolling(ret)
        ax.plot(batch_step, mean, color=color, linewidth=2, label=LABELS[algo])
        ax.fill_between(batch_step, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Batch Step (x256 envs)")
    ax.set_ylabel("Episode Return")
    ax.set_title("REINFORCE vs PPO: Training Instability")
    ax.set_xlim(0, 20000)
    ax.set_ylim(-50, 1100)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    path = os.path.join(fig_dir, "fig6_reinforce_variance.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ===========================================================================
# Fig 7 — SAC Temperature
# ===========================================================================
def fig7_sac_temperature(loss_data, fig_dir):
    if "SAC" not in loss_data:
        print("  [WARN] SAC loss data missing, skipping fig7")
        return None
    df = loss_data["SAC"]
    if "alpha" not in df.columns:
        print("  [WARN] SAC losses has no 'alpha' column, skipping fig7")
        return None

    df = df.dropna(subset=["alpha"])

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(df["update_step"], df["alpha"], color="teal", linewidth=1.2)
    ax.set_xlabel("Environment Step")
    ax.set_ylabel(r"Temperature $\alpha$")
    ax.set_title(r"SAC Entropy Temperature $\alpha$ Over Training")
    ax.grid(True, alpha=0.3)

    path = os.path.join(fig_dir, "fig7_sac_temperature.png")
    fig.savefig(path, bbox_inches="tight")
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

    parser = argparse.ArgumentParser(description="Generate publication figures for HW3.")
    parser.add_argument("--exp-dir", default=default_exp)
    parser.add_argument("--fig-dir", default=default_fig)
    parser.add_argument("--model-dir", default=default_model)
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    train_data, loss_data, deploy_data, traj_data = load_all(args.exp_dir)
    print(f"Training : {list(train_data.keys())}")
    print(f"Deploy   : {list(deploy_data.keys())}")
    print()

    generated = []
    for name, func, data in [
        ("fig1_learning_curves",       fig1_learning_curves,       train_data),
        ("fig2_convergence_speed",     fig2_convergence_speed,     train_data),
        ("fig3_deployment_boxplot",    fig3_deployment_boxplot,    deploy_data),
        ("fig5_action_traces",         fig5_action_traces,         traj_data),
        ("fig6_reinforce_variance",    fig6_reinforce_variance,    train_data),
        ("fig7_sac_temperature",       fig7_sac_temperature,       loss_data),
    ]:
        print(f"Generating {name} ...")
        p = func(data, args.fig_dir)
        if p:
            generated.append(p)

    # Fig 4 needs model_dir, not data dict
    print("Generating fig4_policy_value_surfaces ...")
    p = fig4_policy_value_surfaces(args.model_dir, args.fig_dir)
    if p:
        generated.append(p)

    print(f"\nGenerated {len(generated)} / 7 figures:")
    for p in generated:
        print(f"  {p}")


if __name__ == "__main__":
    main()
