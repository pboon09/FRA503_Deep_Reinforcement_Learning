#!/usr/bin/env python3
"""Deployment trajectory analysis: phase portraits, state trajectories, action profiles.

Reads per-step trajectory CSVs produced by play.py (--trajectory_dir) and generates
advanced control-theory-inspired visualizations.

Usage:
    python scripts/RL_Algorithm/visualize/plot_analysis.py \
        --trajectories experiments/suite_4_deployment/trajectories/MC_trajectory.csv \
                       experiments/suite_4_deployment/trajectories/SARSA_trajectory.csv \
                       experiments/suite_4_deployment/trajectories/Q_Learning_trajectory.csv \
                       experiments/suite_4_deployment/trajectories/Double_Q_Learning_trajectory.csv \
        --output figures/suite_4_deployment/
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _color(i: int) -> str:
    return COLORS[i % len(COLORS)]


def label_from_path(path: str) -> str:
    """Extract algorithm name: 'MC_trajectory.csv' -> 'MC'."""
    stem = Path(path).stem
    return stem.replace("_trajectory", "")


# ─────────────────────────────────────────────────────────────────────────────
# Per-algorithm plots
# ─────────────────────────────────────────────────────────────────────────────

def save_phase_portrait(out_dir: str, label: str, df: pd.DataFrame):
    """Phase portrait: pole_angle vs pole_vel, one line per episode, colored by time.

    A stable controller produces trajectories spiraling inward to the origin.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    episodes = df["episode"].unique()
    cmap = cm.get_cmap("tab10")

    for i, ep in enumerate(episodes):
        ep_df = df[df["episode"] == ep]
        color = cmap(i % 10)
        ax.plot(ep_df["pole_angle"], ep_df["pole_vel"],
                linewidth=1.2, alpha=0.7, color=color, label=f"Ep {ep}")
        # Mark start and end
        ax.plot(ep_df["pole_angle"].iloc[0], ep_df["pole_vel"].iloc[0],
                "o", color=color, markersize=5)
        ax.plot(ep_df["pole_angle"].iloc[-1], ep_df["pole_vel"].iloc[-1],
                "x", color=color, markersize=7, markeredgewidth=2)

    ax.axhline(0, color="k", linewidth=0.5, alpha=0.4)
    ax.axvline(0, color="k", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("Pole Angle (rad)")
    ax.set_ylabel("Pole Angular Velocity (rad/s)")
    ax.set_title(f"Phase Portrait — {label}\n(o = start, x = end)")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "phase_portrait.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_state_trajectory(out_dir: str, label: str, df: pd.DataFrame):
    """Time-series of all 4 state variables across episodes."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Deployment State Trajectory — {label}", fontsize=13, fontweight="bold")

    state_vars = [
        ("cart_pos", "Cart Position (m)", "tab:blue"),
        ("pole_angle", "Pole Angle (rad)", "tab:orange"),
        ("cart_vel", "Cart Velocity (m/s)", "tab:green"),
        ("pole_vel", "Pole Ang. Velocity (rad/s)", "tab:red"),
    ]

    # Create a continuous step index across episodes
    df = df.copy()
    df["global_step"] = range(len(df))

    for ax, (col, ylabel, color) in zip(axes, state_vars):
        ax.plot(df["global_step"], df[col], linewidth=0.8, color=color)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
        # Mark episode boundaries
        for ep in df["episode"].unique()[1:]:
            idx = df[df["episode"] == ep].index[0]
            ax.axvline(df.loc[idx, "global_step"], color="gray",
                       linewidth=0.5, linestyle=":", alpha=0.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    path = os.path.join(out_dir, "state_trajectory.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_action_profile(out_dir: str, label: str, df: pd.DataFrame):
    """Action force over time — reveals control strategy (bang-bang vs smooth)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    df = df.copy()
    df["global_step"] = range(len(df))

    ax.plot(df["global_step"], df["action_val"], linewidth=0.8, color="tab:purple")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
    # Episode boundaries
    for ep in df["episode"].unique()[1:]:
        idx = df[df["episode"] == ep].index[0]
        ax.axvline(df.loc[idx, "global_step"], color="gray",
                   linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Force (N)")
    ax.set_title(f"Action Profile — {label}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "action_profile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison plots (all algorithms)
# ─────────────────────────────────────────────────────────────────────────────

def save_phase_comparison(out_dir: str, datasets: list):
    """2x2 subplot: phase portrait for each algorithm (best episode)."""
    n = len(datasets)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    fig.suptitle("Phase Portrait Comparison (Best Episode)",
                 fontsize=14, fontweight="bold")

    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (label, df) in enumerate(datasets):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        # Pick the episode with highest total reward
        ep_rewards = df.groupby("episode")["reward"].sum()
        best_ep = ep_rewards.idxmax()
        ep_df = df[df["episode"] == best_ep]

        # Color by time step within episode
        steps = np.arange(len(ep_df))
        sc = ax.scatter(ep_df["pole_angle"], ep_df["pole_vel"],
                        c=steps, cmap="viridis", s=8, alpha=0.8)
        ax.plot(ep_df["pole_angle"], ep_df["pole_vel"],
                linewidth=0.5, alpha=0.3, color="gray")
        fig.colorbar(sc, ax=ax, label="Time step")

        ax.axhline(0, color="k", linewidth=0.5, alpha=0.4)
        ax.axvline(0, color="k", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("Pole Angle (rad)")
        ax.set_ylabel("Pole Ang. Velocity (rad/s)")
        ax.set_title(f"{label} (ep {best_ep}, R={ep_rewards[best_ep]:.1f})")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, "phase_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_reward_per_episode(out_dir: str, datasets: list):
    """Bar chart: per-episode reward for each algorithm side by side."""
    fig, ax = plt.subplots(figsize=(12, 6))

    n_algos = len(datasets)
    all_episodes = set()
    for _, df in datasets:
        all_episodes.update(df["episode"].unique())
    episodes = sorted(all_episodes)
    n_eps = len(episodes)

    width = 0.8 / n_algos
    x = np.arange(n_eps)

    for i, (label, df) in enumerate(datasets):
        ep_rewards = df.groupby("episode")["reward"].sum()
        rewards = [ep_rewards.get(ep, 0) for ep in episodes]
        ax.bar(x + i * width, rewards, width, label=label, color=_color(i))

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Deployment: Per-Episode Reward")
    ax.set_xticks(x + width * (n_algos - 1) / 2)
    ax.set_xticklabels([str(ep) for ep in episodes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(out_dir, "reward_per_episode.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_stability_comparison(out_dir: str, datasets: list):
    """Compare control quality: pole angle std and cart position std per algorithm."""
    labels = []
    angle_stds = []
    pos_stds = []

    for label, df in datasets:
        labels.append(label)
        angle_stds.append(df["pole_angle"].std())
        pos_stds.append(df["cart_pos"].std())

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, angle_stds, width, label="Pole Angle Std (rad)", color="tab:orange")
    ax.bar(x + width / 2, pos_stds, width, label="Cart Position Std (m)", color="tab:blue")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Standard Deviation")
    ax.set_title("Deployment Stability: Lower = More Stable Control")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, (a, p) in enumerate(zip(angle_stds, pos_stds)):
        ax.text(i - width / 2, a + 0.005, f"{a:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, p + 0.005, f"{p:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = os.path.join(out_dir, "stability_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_report_deployment(out_dir: str, datasets: list):
    """Combined 2x2 report figure for deployment analysis."""
    n = len(datasets)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Deployment Analysis Report", fontsize=14, fontweight="bold")

    # (a) Phase portrait overlay (all algos, best episode each)
    ax = axes[0, 0]
    for i, (label, df) in enumerate(datasets):
        ep_rewards = df.groupby("episode")["reward"].sum()
        best_ep = ep_rewards.idxmax()
        ep_df = df[df["episode"] == best_ep]
        ax.plot(ep_df["pole_angle"], ep_df["pole_vel"],
                linewidth=1.2, alpha=0.7, color=_color(i), label=label)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.4)
    ax.axvline(0, color="k", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("Pole Angle (rad)")
    ax.set_ylabel("Pole Ang. Velocity (rad/s)")
    ax.set_title("(a) Phase Portrait (Best Episode)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Pole angle over time (best episode each)
    ax = axes[0, 1]
    for i, (label, df) in enumerate(datasets):
        ep_rewards = df.groupby("episode")["reward"].sum()
        best_ep = ep_rewards.idxmax()
        ep_df = df[df["episode"] == best_ep].reset_index(drop=True)
        ax.plot(ep_df.index, ep_df["pole_angle"],
                linewidth=1.0, alpha=0.8, color=_color(i), label=label)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Pole Angle (rad)")
    ax.set_title("(b) Pole Angle Over Time (Best Episode)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Stability bar chart
    ax = axes[1, 0]
    labels_list = [l for l, _ in datasets]
    angle_stds = [df["pole_angle"].std() for _, df in datasets]
    x = np.arange(n)
    bars = ax.bar(x, angle_stds, 0.5, color=[_color(i) for i in range(n)])
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Pole Angle Std (rad)")
    ax.set_title("(c) Control Stability (Lower = Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, rotation=15)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, angle_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # (d) Action distribution
    ax = axes[1, 1]
    for i, (label, df) in enumerate(datasets):
        ax.hist(df["action_val"], bins=30, alpha=0.5, density=True,
                label=label, color=_color(i))
    ax.set_xlabel("Force (N)")
    ax.set_ylabel("Density")
    ax.set_title("(d) Deployment Action Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "report_deployment.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Deployment trajectory analysis plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--trajectories", nargs="+", required=True, metavar="CSV",
        help="One or more trajectory CSV files from play.py.",
    )
    p.add_argument(
        "--output", default="figures/suite_4_deployment", metavar="DIR",
        help="Output directory for plots.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Expand globs
    resolved = []
    for pattern in args.trajectories:
        matches = sorted(glob.glob(pattern, recursive=True))
        resolved.extend(matches if matches else [pattern])

    # Load
    datasets = []
    for path in resolved:
        if not os.path.isfile(path):
            print(f"WARNING: Not found, skipping: {path}", file=sys.stderr)
            continue
        print(f"Loading: {path}")
        df = pd.read_csv(path)
        label = label_from_path(path)
        datasets.append((label, df))

    if not datasets:
        print("ERROR: No valid trajectory files loaded.", file=sys.stderr)
        sys.exit(1)

    root = args.output

    # Per-algorithm plots
    for label, df in datasets:
        algo_dir = os.path.join(root, label)
        os.makedirs(algo_dir, exist_ok=True)
        print(f"\n[{label}] → {algo_dir}/")
        save_phase_portrait(algo_dir, label, df)
        save_state_trajectory(algo_dir, label, df)
        save_action_profile(algo_dir, label, df)

    # Comparison plots
    if len(datasets) > 1:
        cmp_dir = os.path.join(root, "comparison")
        os.makedirs(cmp_dir, exist_ok=True)
        print(f"\n[comparison] → {cmp_dir}/")
        save_phase_comparison(cmp_dir, datasets)
        save_reward_per_episode(cmp_dir, datasets)
        save_stability_comparison(cmp_dir, datasets)
        save_report_deployment(cmp_dir, datasets)

    print("\nDone.")


if __name__ == "__main__":
    main()
