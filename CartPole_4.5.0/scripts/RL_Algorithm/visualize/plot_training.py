#!/usr/bin/env python3
"""Visualize CartPole tabular RL training logs.

Generates plots from one or more CSV log files produced by train.py.
Saves per-algorithm figures to  figures/{algorithm}/
Saves comparison figures to     figures/comparison/   (or figures/{algorithm}/ if only one file)

Usage:
    # Single run
    python scripts/RL_Algorithm/visualize/plot_training.py \\
        --logs logs/Stabilize/Q_Learning/training_log_2024-01-01_12-00-00.csv

    # Compare algorithms (shell globs accepted)
    python scripts/RL_Algorithm/visualize/plot_training.py \\
        --logs "logs/Stabilize/MC/training_log_*.csv" \\
               "logs/Stabilize/SARSA/training_log_*.csv" \\
               "logs/Stabilize/Q_Learning/training_log_*.csv" \\
               "logs/Stabilize/Double_Q_Learning/training_log_*.csv" \\
        --output figures/ --show
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
import seaborn as sns

GAMMA = 0.99   # discount factor used for TD-error approximation

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def label_from_path(path: str) -> str:
    """Extract algorithm name from the CSV path.

    Prefers the filename stem (e.g. 'MC' from 'MC.csv').
    Falls back to parent directory if the stem starts with 'training_log'.
    """
    stem = Path(path).stem
    if not stem.startswith("training_log"):
        return stem
    parts = Path(path).parts
    if len(parts) >= 3:
        return parts[-2]
    return stem


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["_q_cols"] = None   # placeholder; use get_q_cols() instead
    return df


def get_q_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("q_")]


def rolling_mean(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).mean()


def progress_pct(df: pd.DataFrame) -> np.ndarray:
    """Return 0–100 % array based on row position (normalises different-length runs)."""
    n = len(df)
    if n <= 1:
        return np.zeros(n)
    return np.linspace(0, 100, n)


def episode_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-episode data.

    Returns DataFrame with columns: [episode, steps, sum_reward, var_pole_angle].
    Supports both formats:
      - New: one row per episode with 'ep_length' and 'reward' = total episode reward
      - Old: multiple rows per episode, grouped by episode number
    """
    if "ep_length" in df.columns:
        # New format: one row per completed episode
        return pd.DataFrame({
            "episode": df["episode"].values,
            "steps": df["ep_length"].values,
            "sum_reward": df["reward"].values,
            "var_pole_angle": df["pole_angle"].values,
        }).reset_index(drop=True)
    else:
        # Old format: multiple rows per episode block
        block_id = (df["episode"] != df["episode"].shift()).cumsum()
        grp = df.groupby(block_id, sort=False)
        result = grp.agg(
            episode=("episode", "first"),
            steps=("step", "count"),
            sum_reward=("reward", "sum"),
            var_pole_angle=("pole_angle", "var"),
        ).reset_index(drop=True)
        return result[result["steps"] > 0].reset_index(drop=True)


def compute_td_error(df: pd.DataFrame) -> pd.Series:
    """Approximate TD error for each step.

    td_error[t] ≈ | reward[t] + γ * max_Q(s')[t+1] - Q(s,a)[t] |

    Q(s,a)[t] = q_{action_idx}[t]
    max_Q(s')[t+1] is taken from the next row's q-value columns.
    The last row gets NaN.
    """
    q_cols = get_q_cols(df)
    if not q_cols:
        return pd.Series(np.nan, index=df.index)

    # Q value of the action taken at each step
    q_taken = np.array([
        df[f"q_{int(a)}"].iloc[i] if f"q_{int(a)}" in df.columns else np.nan
        for i, a in enumerate(df["action_idx"])
    ])

    max_q_next = df[q_cols].max(axis=1).shift(-1).values   # max Q of next state

    td_err = np.abs(df["reward"].values + GAMMA * max_q_next - q_taken)
    return pd.Series(td_err, index=df.index)


def action_entropy_per_block(df: pd.DataFrame) -> pd.Series:
    """Entropy of action_idx distribution within each episode block."""
    block_id = (df["episode"] != df["episode"].shift()).cumsum()
    def entropy(group):
        counts = group["action_idx"].value_counts(normalize=True)
        return -(counts * np.log(counts + 1e-12)).sum()
    return df.groupby(block_id, sort=False).apply(entropy).reset_index(drop=True)


def latest_q_per_state(df: pd.DataFrame) -> pd.DataFrame:
    """For each unique (pole_angle_dis, cart_pos_dis), keep the last row."""
    q_cols = get_q_cols(df)
    keep = ["pole_angle_dis", "cart_pos_dis", "action_idx"] + q_cols
    sub = df[keep].dropna()
    # last visit to each state
    return sub.groupby(["pole_angle_dis", "cart_pos_dis"], sort=False).last().reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Per-algorithm (individual) plots  →  figures/{algorithm}/
# ─────────────────────────────────────────────────────────────────────────────

def save_state_heatmap(out_dir: str, label: str, df: pd.DataFrame):
    """Visit-count heatmap: pole_angle_dis × cart_pos_dis."""
    fig, ax = plt.subplots(figsize=(8, 6))
    counts = (
        df.groupby(["pole_angle_dis", "cart_pos_dis"]).size()
          .reset_index(name="count")
    )
    if counts.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        pivot = counts.pivot(
            index="pole_angle_dis", columns="cart_pos_dis", values="count"
        ).fillna(0)
        sns.heatmap(
            np.log1p(pivot), ax=ax, cmap="YlOrRd",
            cbar_kws={"label": "log(1+visits)"},
            xticklabels=False, yticklabels=False,
        )
    ax.set_title(f"State Visitation Heatmap — {label}")
    ax.set_xlabel("cart_pos_dis →")
    ax.set_ylabel("pole_angle_dis ↑")
    fig.tight_layout()
    path = os.path.join(out_dir, "state_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_policy_heatmap(out_dir: str, label: str, df: pd.DataFrame,
                        action_range: tuple[float, float] = (-10.0, 10.0)):
    """Learned policy heatmap: for each visited state, show best force (N).

    Unvisited states (all Q == 0) are masked and shown in gray.
    Color encodes the mapped physical force, not the raw action index.
    """
    q_cols = get_q_cols(df)
    if not q_cols:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    latest = latest_q_per_state(df)
    if latest.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        n_actions = len(q_cols)
        a_min, a_max = action_range

        # Mask unvisited states (all Q == 0)
        q_vals = latest[q_cols].values
        visited = np.abs(q_vals).sum(axis=1) > 0
        best_idx = q_vals.argmax(axis=1).astype(float)
        best_idx[~visited] = np.nan

        # Map action index → physical force
        latest["best_force"] = a_min + (a_max - a_min) * best_idx / max(n_actions - 1, 1)
        pivot = latest.pivot(
            index="pole_angle_dis", columns="cart_pos_dis", values="best_force"
        )
        ax.set_facecolor("#d9d9d9")      # gray background for unvisited
        cmap = cm.get_cmap("RdYlGn")
        im = ax.pcolormesh(
            pivot.columns, pivot.index, pivot.values,
            cmap=cmap, vmin=a_min, vmax=a_max,
        )
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Best Action Force (N)")
        cb.set_ticks([a_min, 0, a_max])
        cb.set_ticklabels([f"{a_min:.0f}N (Left)", "0N", f"{a_max:.0f}N (Right)"])

    ax.set_title(f"Learned Policy — {label}")
    ax.set_xlabel("cart_pos_dis →")
    ax.set_ylabel("pole_angle_dis ↑")
    fig.tight_layout()
    path = os.path.join(out_dir, "policy_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_state_trajectory(out_dir: str, label: str, df: pd.DataFrame):
    """Raw pole_angle and cart_pos over global steps — visual control quality."""
    # Subsample to keep the plot readable (max 5000 points)
    step = max(1, len(df) // 5000)
    sub = df.iloc[::step]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"State Trajectory — {label}", fontsize=12)

    ax1.plot(sub["step"], sub["pole_angle"], linewidth=0.6, color="tab:orange")
    ax1.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.set_ylabel("Pole Angle (rad)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(sub["step"], sub["cart_pos"], linewidth=0.6, color="tab:blue")
    ax2.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_ylabel("Cart Position (m)")
    ax2.set_xlabel("Global Step")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "state_trajectory.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_phase_portrait(out_dir: str, label: str, df: pd.DataFrame):
    """Phase portrait: pole_angle vs pole_vel, colored by action_val."""
    fig, ax = plt.subplots(figsize=(8, 6))
    # Subsample
    step = max(1, len(df) // 3000)
    sub = df.iloc[::step]

    sc = ax.scatter(
        sub["pole_angle"], sub["pole_vel"],
        c=sub["action_val"], cmap="coolwarm",
        s=2, alpha=0.5,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Action Value (force)")
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.4)
    ax.axvline(0, color="k", linewidth=0.5, alpha=0.4)
    ax.set_title(f"Phase Portrait — {label}")
    ax.set_xlabel("Pole Angle (rad)")
    ax.set_ylabel("Pole Angular Velocity (rad/s)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "phase_portrait.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison plots  →  figures/comparison/
# ─────────────────────────────────────────────────────────────────────────────

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _color(i: int) -> str:
    return COLORS[i % len(COLORS)]


def save_reward_curve(out_dir: str, datasets, window: int):
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, df) in enumerate(datasets):
        s = rolling_mean(df["reward"], window)
        ax.plot(progress_pct(df), s, label=label, linewidth=1.8, color=_color(i))
    ax.set_title("Reward per Step")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "reward_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_episode_reward(out_dir: str, datasets, window: int):
    """Total reward summed per episode block."""
    fig, ax = plt.subplots(figsize=(10, 5))
    smooth_w = max(1, window // 20)
    for i, (label, df) in enumerate(datasets):
        ep = episode_blocks(df)
        s = rolling_mean(ep["sum_reward"], smooth_w)
        ax.plot(ep["episode"], s, label=label, linewidth=1.8, color=_color(i))
    ax.set_title("Episode Total Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "episode_reward.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_episode_length(out_dir: str, datasets, window: int):
    fig, ax = plt.subplots(figsize=(10, 5))
    smooth_w = max(1, window // 20)
    for i, (label, df) in enumerate(datasets):
        ep = episode_blocks(df)
        s = rolling_mean(ep["steps"], smooth_w)
        ax.plot(ep["episode"], s, label=label, linewidth=1.8, color=_color(i))
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode (env 0)")
    ax.set_ylabel("Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "episode_length.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_epsilon(out_dir: str, datasets):
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (label, df) in enumerate(datasets):
        ax.plot(progress_pct(df), df["epsilon"], label=label, linewidth=1.8, color=_color(i))
    ax.set_title("Epsilon Decay")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("Epsilon")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "epsilon_decay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_action_distribution(out_dir: str, datasets):
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, df) in enumerate(datasets):
        ax.hist(df["action_val"], bins=60, alpha=0.5, density=True,
                label=label, color=_color(i))
    ax.set_title("Action Value Distribution")
    ax.set_xlabel("Action Value (continuous force)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "action_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_max_q(out_dir: str, datasets, window: int):
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, df) in enumerate(datasets):
        q_cols = get_q_cols(df)
        if not q_cols:
            continue
        max_q = df[q_cols].max(axis=1)
        ax.plot(progress_pct(df), rolling_mean(max_q, window),
                label=label, linewidth=1.8, color=_color(i))
    ax.set_title("Max Q-value")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("Max Q-value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "max_q_value.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_q_spread(out_dir: str, datasets, window: int):
    """max(Q) - min(Q) per step — shows how differentiated the value estimates are."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, df) in enumerate(datasets):
        q_cols = get_q_cols(df)
        if not q_cols:
            continue
        spread = df[q_cols].max(axis=1) - df[q_cols].min(axis=1)
        ax.plot(progress_pct(df), rolling_mean(spread, window),
                label=label, linewidth=1.8, color=_color(i))
    ax.set_title("Q-value Spread (max − min)")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("Q-value Spread")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "q_spread.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_td_error(out_dir: str, datasets, window: int):
    """Approximate TD error: |r + γ·maxQ(s') − Q(s,a)|. Proxy for learning convergence."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, df) in enumerate(datasets):
        td = compute_td_error(df).dropna()
        if td.empty:
            continue
        pct_idx = progress_pct(df)[td.index]
        ax.plot(pct_idx, rolling_mean(td, window),
                label=label, linewidth=1.8, color=_color(i))
    ax.set_title("TD Error")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("TD Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "td_error.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_pole_variance(out_dir: str, datasets, window: int):
    """Variance of pole_angle per episode block — direct control quality metric."""
    fig, ax = plt.subplots(figsize=(10, 5))
    smooth_w = max(1, window // 20)
    for i, (label, df) in enumerate(datasets):
        ep = episode_blocks(df)
        s = rolling_mean(ep["var_pole_angle"].fillna(0), smooth_w)
        ax.plot(ep["episode"], s, label=label, linewidth=1.8, color=_color(i))
    ax.set_title("Pole Angle Variance per Episode")
    ax.set_xlabel("Episode (env 0)")
    ax.set_ylabel("Variance of Pole Angle (rad²)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "pole_angle_variance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_action_entropy(out_dir: str, datasets):
    """Shannon entropy of action distribution per episode block."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, df) in enumerate(datasets):
        ent = action_entropy_per_block(df)
        ep = episode_blocks(df)
        ax.plot(ep["episode"], rolling_mean(ent, max(1, len(ent) // 100)),
                label=label, linewidth=1.8, color=_color(i))
    ax.set_title("Action Entropy per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Entropy (nats)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "action_entropy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_origin_q_convergence(out_dir: str, datasets):
    """Track max Q-value at the origin state (0,0,0,0) over episodes.

    The origin = pole upright, center, zero velocity.  Monotonic increase
    toward a plateau indicates convergence of the value estimate.
    """
    state_cols = ["cart_pos_dis", "pole_angle_dis", "cart_vel_dis", "pole_vel_dis"]
    fig, ax = plt.subplots(figsize=(10, 5))
    any_data = False
    for i, (label, df) in enumerate(datasets):
        if not all(c in df.columns for c in state_cols):
            continue
        q_cols = get_q_cols(df)
        if not q_cols:
            continue
        # Filter rows at origin state
        mask = (
            (df["cart_pos_dis"] == 0) &
            (df["pole_angle_dis"] == 0) &
            (df["cart_vel_dis"] == 0) &
            (df["pole_vel_dis"] == 0)
        )
        origin = df.loc[mask].copy()
        if origin.empty:
            continue
        any_data = True
        max_q = origin[q_cols].max(axis=1)
        ax.plot(origin["episode"].values, max_q.values,
                label=label, linewidth=1.8, color=_color(i), alpha=0.8)

    if not any_data:
        plt.close(fig)
        return

    ax.set_title("Q-Value Convergence at Origin State (0,0,0,0)")
    ax.set_xlabel("Episode (env 0)")
    ax.set_ylabel("max Q(s_origin, a)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "origin_q_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_report_training(out_dir: str, datasets, window: int):
    """Combined 2x2 subplot: reward, episode length, epsilon, state coverage."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Performance Comparison", fontsize=14, fontweight="bold")

    # (a) Reward curve
    ax = axes[0, 0]
    for i, (label, df) in enumerate(datasets):
        ax.plot(progress_pct(df), rolling_mean(df["reward"], window),
                label=label, linewidth=1.5, color=_color(i))
    ax.set_title("(a) Reward per Step")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Episode length
    ax = axes[0, 1]
    smooth_w = max(1, window // 20)
    for i, (label, df) in enumerate(datasets):
        ep = episode_blocks(df)
        ax.plot(ep["episode"], rolling_mean(ep["steps"], smooth_w),
                label=label, linewidth=1.5, color=_color(i))
    ax.set_title("(b) Episode Length")
    ax.set_xlabel("Episode (env 0)")
    ax.set_ylabel("Steps")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Epsilon decay
    ax = axes[1, 0]
    for i, (label, df) in enumerate(datasets):
        ax.plot(progress_pct(df), df["epsilon"],
                label=label, linewidth=1.5, color=_color(i))
    ax.set_title("(c) Epsilon Decay")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("Epsilon")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) State coverage
    ax = axes[1, 1]
    state_cols = ["cart_pos_dis", "pole_angle_dis", "cart_vel_dis", "pole_vel_dis"]
    for i, (label, df) in enumerate(datasets):
        if not all(c in df.columns for c in state_cols):
            continue
        tuples = list(df[state_cols].itertuples(index=False, name=None))
        seen: set = set()
        cumulative = []
        for t in tuples:
            seen.add(t)
            cumulative.append(len(seen))
        ax.plot(progress_pct(df), cumulative,
                label=label, linewidth=1.5, color=_color(i))
    ax.set_title("(d) Cumulative Unique States")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("Unique States")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "report_training.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_report_qvalue(out_dir: str, datasets, window: int):
    """Combined 1x2 subplot: max Q-value + origin Q convergence."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Q-Value Analysis", fontsize=14, fontweight="bold")

    # (a) Max Q-value
    ax = axes[0]
    for i, (label, df) in enumerate(datasets):
        q_cols = get_q_cols(df)
        if not q_cols:
            continue
        max_q = df[q_cols].max(axis=1)
        ax.plot(progress_pct(df), rolling_mean(max_q, window),
                label=label, linewidth=1.5, color=_color(i))
    ax.set_title("(a) Max Q-value")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("Max Q-value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Origin state convergence
    ax = axes[1]
    state_cols = ["cart_pos_dis", "pole_angle_dis", "cart_vel_dis", "pole_vel_dis"]
    for i, (label, df) in enumerate(datasets):
        if not all(c in df.columns for c in state_cols):
            continue
        q_cols = get_q_cols(df)
        if not q_cols:
            continue
        mask = (
            (df["cart_pos_dis"] == 0) & (df["pole_angle_dis"] == 0) &
            (df["cart_vel_dis"] == 0) & (df["pole_vel_dis"] == 0)
        )
        origin = df.loc[mask]
        if origin.empty:
            continue
        ax.plot(origin["episode"].values, origin[q_cols].max(axis=1).values,
                label=label, linewidth=1.5, color=_color(i), alpha=0.8)
    ax.set_title("(b) Q-Value at Origin (0,0,0,0)")
    ax.set_xlabel("Episode (env 0)")
    ax.set_ylabel("max Q(s_origin, a)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "report_qvalue.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_state_coverage(out_dir: str, datasets):
    state_cols = ["cart_pos_dis", "pole_angle_dis", "cart_vel_dis", "pole_vel_dis"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, df) in enumerate(datasets):
        if not all(c in df.columns for c in state_cols):
            continue
        tuples = list(df[state_cols].itertuples(index=False, name=None))
        seen: set = set()
        cumulative = []
        for t in tuples:
            seen.add(t)
            cumulative.append(len(seen))
        ax.plot(progress_pct(df), cumulative,
                label=label, linewidth=1.8, color=_color(i))
    ax.set_title("Cumulative Unique States Visited")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("Unique States")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "state_coverage.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing & main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize CartPole tabular RL training logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--logs", nargs="+", required=True, metavar="CSV",
        help="One or more CSV log files (shell glob patterns accepted).",
    )
    p.add_argument(
        "--output", default="figures", metavar="DIR",
        help="Root output directory (default: figures/).",
    )
    p.add_argument(
        "--window", type=int, default=200, metavar="N",
        help="Rolling window size for smoothing (default: 200).",
    )
    p.add_argument(
        "--gamma", type=float, default=GAMMA, metavar="G",
        help=f"Discount factor for TD-error approximation (default: {GAMMA}).",
    )
    p.add_argument(
        "--show", action="store_true", default=False,
        help="Show interactive matplotlib window after saving.",
    )
    return p.parse_args()


def make_unique_labels(raw: list[tuple[str, pd.DataFrame]]) -> list[tuple[str, pd.DataFrame]]:
    seen: dict[str, int] = {}
    result = []
    for label, df in raw:
        if label in seen:
            seen[label] += 1
            label = f"{label}_{seen[label]}"
        else:
            seen[label] = 0
        result.append((label, df))
    return result


def main():
    args = parse_args()
    global GAMMA
    GAMMA = args.gamma

    # Expand globs
    resolved: list[str] = []
    for pattern in args.logs:
        matches = glob.glob(pattern, recursive=True)
        resolved.extend(matches if matches else [pattern])

    # Load
    raw: list[tuple[str, pd.DataFrame]] = []
    for path in resolved:
        if not os.path.isfile(path):
            print(f"WARNING: Not found, skipping: {path}", file=sys.stderr)
            continue
        print(f"Loading: {path}")
        df = pd.read_csv(path)
        raw.append((label_from_path(path), df))

    if not raw:
        print("ERROR: No valid CSV files loaded.", file=sys.stderr)
        sys.exit(1)

    datasets = make_unique_labels(raw)
    root = args.output
    w = args.window

    # Decide comparison directory
    if len(datasets) == 1:
        cmp_dir = os.path.join(root, datasets[0][0])
    else:
        cmp_dir = os.path.join(root, "comparison")
    os.makedirs(cmp_dir, exist_ok=True)

    # ── Per-algorithm individual plots ────────────────────────────────────
    for label, df in datasets:
        algo_dir = os.path.join(root, label)
        os.makedirs(algo_dir, exist_ok=True)
        print(f"\n[{label}] per-algorithm plots → {algo_dir}/")
        save_state_heatmap(algo_dir, label, df)
        save_policy_heatmap(algo_dir, label, df)
        save_state_trajectory(algo_dir, label, df)
        save_phase_portrait(algo_dir, label, df)

    # ── Comparison / aggregate plots ─────────────────────────────────────
    print(f"\n[comparison] plots → {cmp_dir}/")
    save_reward_curve(cmp_dir, datasets, w)
    save_episode_reward(cmp_dir, datasets, w)
    save_episode_length(cmp_dir, datasets, w)
    save_epsilon(cmp_dir, datasets)
    save_action_distribution(cmp_dir, datasets)
    save_max_q(cmp_dir, datasets, w)
    save_q_spread(cmp_dir, datasets, w)
    save_td_error(cmp_dir, datasets, w)
    save_pole_variance(cmp_dir, datasets, w)
    save_action_entropy(cmp_dir, datasets)
    save_state_coverage(cmp_dir, datasets)
    save_origin_q_convergence(cmp_dir, datasets)

    # ── Combined report figures (compact subplots) ───────────────────────
    print(f"\n[report] combined figures → {cmp_dir}/")
    save_report_training(cmp_dir, datasets, w)
    save_report_qvalue(cmp_dir, datasets, w)

    print("\nDone.")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
