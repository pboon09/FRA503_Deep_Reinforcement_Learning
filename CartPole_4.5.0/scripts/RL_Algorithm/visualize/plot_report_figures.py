#!/usr/bin/env python3
"""Generate exactly 6 report figures for hw2.tex (Gemini's architecture).

Figures:
  1. fig1_feedback_loop.png    — 1×2: Total Reward + State Coverage
  2. fig2_credit_assignment.png — 1×2: Max Q-value + TD Error
  3. fig3_representation.png   — 2×2: Value heatmaps (top) + Policy heatmaps (bottom)
  4. fig4_action_sweep.png     — 2×2: Action resolution per algorithm
  5. fig5_state_sweep.png      — 2×2: State resolution per algorithm
  6. fig6_deployment.png       — 2×2: Phase portraits (deployment, best episode)

Data sources:
  - Suite 1: experiments/suite_1_baseline/{algo}.csv + {algo}.json
  - Suite 2: experiments/suite_2_action/{algo}_act_{n}.csv
  - Suite 3: experiments/suite_3_state/{algo}_{label}.csv
  - Suite 4: experiments/suite_4_deployment/trajectories/{algo}_trajectory.csv

Usage:
    python scripts/RL_Algorithm/visualize/plot_report_figures.py
    python scripts/RL_Algorithm/visualize/plot_report_figures.py --output figures/report
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ALGOS = ["MC", "SARSA", "Q_Learning", "Double_Q_Learning"]
ALGO_DISPLAY = {
    "MC": "MC", "SARSA": "SARSA",
    "Q_Learning": "Q-Learning", "Double_Q_Learning": "Double Q-Learning",
}
ALGO_COLORS = {
    "MC": "#1f77b4", "SARSA": "#ff7f0e",
    "Q_Learning": "#2ca02c", "Double_Q_Learning": "#d62728",
}

GAMMA = 0.99

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_suite1_csv(algo: str) -> pd.DataFrame | None:
    path = os.path.join(ROOT, "experiments", "suite_1_baseline", f"{algo}.csv")
    return pd.read_csv(path) if os.path.isfile(path) else None


def load_qtable(path: str) -> dict[tuple, list[float]]:
    with open(path, "r") as f:
        data = json.load(f)
    result = {}
    for key_str, vals in data["q_values"].items():
        key_str = key_str.replace("(", "").replace(")", "")
        state = tuple(float(x) for x in key_str.split(", "))
        result[state] = vals
    return result


def episode_blocks(df: pd.DataFrame) -> pd.DataFrame:
    if "ep_length" in df.columns:
        return pd.DataFrame({
            "episode": df["episode"].values,
            "steps": df["ep_length"].values,
            "sum_reward": df["reward"].values,
        }).reset_index(drop=True)
    block_id = (df["episode"] != df["episode"].shift()).cumsum()
    grp = df.groupby(block_id, sort=False)
    result = grp.agg(
        episode=("episode", "first"),
        steps=("step", "count"),
        sum_reward=("reward", "sum"),
    ).reset_index(drop=True)
    return result[result["steps"] > 0].reset_index(drop=True)


def rolling_mean(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).mean()


def get_q_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("q_")]


def progress_pct(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    return np.linspace(0, 100, n) if n > 1 else np.zeros(n)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Exploration Feedback Loop (1×2)
#   Left:  Total Reward (rolling mean ± std)
#   Right: Cumulative Unique States Visited
# ─────────────────────────────────────────────────────────────────────────────

def make_fig1(output_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("The Exploration Feedback Loop", fontsize=14, fontweight="bold")

    state_cols = ["cart_pos_dis", "pole_angle_dis", "cart_vel_dis", "pole_vel_dis"]

    for algo in ALGOS:
        df = load_suite1_csv(algo)
        if df is None:
            continue
        color = ALGO_COLORS[algo]
        label = ALGO_DISPLAY[algo]

        # Left: Total Reward
        ep = episode_blocks(df)
        w = max(1, len(ep) // 15)
        smooth = rolling_mean(ep["sum_reward"], w)
        std = ep["sum_reward"].rolling(w, min_periods=1).std().fillna(0)
        ax1.plot(ep["episode"], smooth, label=label, linewidth=1.8, color=color)
        ax1.fill_between(ep["episode"], smooth - std, smooth + std,
                         alpha=0.15, color=color)

        # Right: State Coverage
        if all(c in df.columns for c in state_cols):
            tuples = list(df[state_cols].itertuples(index=False, name=None))
            seen = set()
            cumulative = []
            for t in tuples:
                seen.add(t)
                cumulative.append(len(seen))
            ax2.plot(progress_pct(df), cumulative,
                     label=label, linewidth=1.8, color=color)

    ax1.set_title("(a) Episode Total Reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.set_title("(b) Cumulative Unique States Visited")
    ax2.set_xlabel("Training Progress (%)")
    ax2.set_ylabel("Unique States")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig1_feedback_loop.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Credit Assignment Bottleneck (1×2)
#   Left:  Max Q-value convergence (rolling mean ± std)
#   Right: TD Error magnitude over training
# ─────────────────────────────────────────────────────────────────────────────

def make_fig2(output_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("The Credit Assignment Bottleneck",
                 fontsize=14, fontweight="bold")

    window = 200

    for algo in ALGOS:
        df = load_suite1_csv(algo)
        if df is None:
            continue
        color = ALGO_COLORS[algo]
        label = ALGO_DISPLAY[algo]
        pct = progress_pct(df)

        # Left: Max Q-value
        q_cols = get_q_cols(df)
        if q_cols:
            max_q = df[q_cols].max(axis=1)
            smooth = rolling_mean(max_q, window)
            ax1.plot(pct, smooth, label=label, linewidth=1.8, color=color)

        # Right: TD Error = |r + γ·maxQ(s') - Q(s,a)|
        if q_cols:
            q_taken = np.array([
                df[f"q_{int(a)}"].iloc[i] if f"q_{int(a)}" in df.columns else np.nan
                for i, a in enumerate(df["action_idx"])
            ])
            max_q_next = df[q_cols].max(axis=1).shift(-1).values
            td_err = np.abs(df["reward"].values + GAMMA * max_q_next - q_taken)
            td_series = pd.Series(td_err, index=df.index).dropna()
            if not td_series.empty:
                pct_td = pct[td_series.index]
                ax2.plot(pct_td, rolling_mean(td_series, window),
                         label=label, linewidth=1.8, color=color)

    ax1.set_title("(a) Max Q-value per Step")
    ax1.set_xlabel("Training Progress (%)")
    ax1.set_ylabel("Max Q-value")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("(b) TD Error (Bellman Residual)")
    ax2.set_xlabel("Training Progress (%)")
    ax2.set_ylabel("|r + γ·max Q(s') − Q(s,a)|")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig2_credit_assignment.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Representation Structure (2×2)
#   Top row:    V(s) heatmap for each algorithm
#   Bottom row: Policy heatmap for each algorithm
#   (Uses 2×4 layout: top 4 = value, bottom 4 = policy)
#   Actually Gemini said 2×2 with MC vs QL only. But 4-panel is more complete.
#   We'll do 2 rows × 4 cols for all 4 algos.
# ─────────────────────────────────────────────────────────────────────────────

DIM_POLE_ANG = 1
DIM_POLE_VEL = 3


def aggregate_v_surface(qtable, dim_x, dim_y):
    v_map = {}
    for state, q_vals in qtable.items():
        key = (state[dim_x], state[dim_y])
        v_map.setdefault(key, []).append(max(q_vals))
    points = {k: np.mean(vs) for k, vs in v_map.items()}
    if not points:
        return np.array([]), np.array([]), np.array([[]])
    xs = sorted(set(k[0] for k in points))
    ys = sorted(set(k[1] for k in points))
    v_grid = np.full((len(ys), len(xs)), np.nan)
    for (x, y), v in points.items():
        v_grid[ys.index(y), xs.index(x)] = v
    return np.array(xs), np.array(ys), v_grid


def aggregate_policy(qtable, dim_x, dim_y, action_range=(-10.0, 10.0)):
    from collections import Counter
    action_map = {}
    n_actions = 0
    for state, q_vals in qtable.items():
        n_actions = max(n_actions, len(q_vals))
        if sum(abs(v) for v in q_vals) == 0:
            continue
        key = (state[dim_x], state[dim_y])
        action_map.setdefault(key, []).append(int(np.argmax(q_vals)))
    policy = {}
    for key, actions in action_map.items():
        policy[key] = Counter(actions).most_common(1)[0][0]
    if not policy:
        return np.array([]), np.array([]), np.array([[]]), action_range
    xs = sorted(set(k[0] for k in policy))
    ys = sorted(set(k[1] for k in policy))
    a_min, a_max = action_range
    force_grid = np.full((len(ys), len(xs)), np.nan)
    for (x, y), a in policy.items():
        force = a_min + (a_max - a_min) * a / max(n_actions - 1, 1)
        force_grid[ys.index(y), xs.index(x)] = force
    return np.array(xs), np.array(ys), force_grid, action_range


def make_fig3(output_dir: str):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle("Representation Structure: Value Functions (top) and Policies (bottom)",
                 fontsize=14, fontweight="bold")

    for idx, algo in enumerate(ALGOS):
        qtable_path = os.path.join(
            ROOT, "experiments", "suite_1_baseline", f"{algo}.json")
        if not os.path.isfile(qtable_path):
            print(f"  WARNING: Missing {qtable_path}")
            continue
        qtable = load_qtable(qtable_path)

        # Top row: V(s) heatmap
        ax = axes[0, idx]
        xs, ys, v_grid = aggregate_v_surface(qtable, DIM_POLE_ANG, DIM_POLE_VEL)
        if xs.size > 0:
            v_filled = np.nan_to_num(v_grid, nan=0.0)
            im = ax.pcolormesh(xs, ys, v_filled, cmap="viridis")
            fig.colorbar(im, ax=ax, label="V(s)", shrink=0.8)
        ax.set_xlabel("Pole Angle")
        ax.set_ylabel("Pole Angular Velocity")
        ax.set_title(f"{ALGO_DISPLAY[algo]}")

        # Bottom row: Policy heatmap
        ax = axes[1, idx]
        ax.set_facecolor("#d9d9d9")
        xs, ys, force_grid, a_range = aggregate_policy(
            qtable, DIM_POLE_ANG, DIM_POLE_VEL)
        if xs.size > 0:
            cmap_obj = cm.get_cmap("RdYlGn")
            im = ax.pcolormesh(xs, ys, force_grid, cmap=cmap_obj,
                               vmin=a_range[0], vmax=a_range[1])
            fig.colorbar(im, ax=ax, label="Force (N)", shrink=0.8)
        ax.set_xlabel("Pole Angle")
        ax.set_ylabel("Pole Angular Velocity")
        ax.set_title(f"{ALGO_DISPLAY[algo]}")

    # Row labels
    axes[0, 0].annotate("V(s) = max Q(s,a)", xy=(-0.3, 0.5),
                         xycoords="axes fraction", fontsize=12,
                         fontweight="bold", rotation=90, va="center")
    axes[1, 0].annotate("π(s) = argmax Q(s,a)", xy=(-0.3, 0.5),
                         xycoords="axes fraction", fontsize=12,
                         fontweight="bold", rotation=90, va="center")

    fig.tight_layout()
    path = os.path.join(output_dir, "fig3_representation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Action Resolution Sweep (2×2)
# ─────────────────────────────────────────────────────────────────────────────

def make_fig4(output_dir: str):
    suite2_dir = os.path.join(ROOT, "experiments", "suite_2_action")
    actions = [3, 5, 11, 21]
    action_colors = {3: "#1f77b4", 5: "#ff7f0e", 11: "#2ca02c", 21: "#d62728"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Action Resolution Sensitivity (Episode Total Reward)",
                 fontsize=14, fontweight="bold")

    for idx, algo in enumerate(ALGOS):
        r, c = divmod(idx, 2)
        ax = axes[r, c]
        for n_act in actions:
            csv_path = os.path.join(suite2_dir, f"{algo}_act_{n_act}.csv")
            if not os.path.isfile(csv_path):
                print(f"  WARNING: Missing {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            ep = episode_blocks(df)
            w = max(1, len(ep) // 15)
            smooth = rolling_mean(ep["sum_reward"], w)
            std = ep["sum_reward"].rolling(w, min_periods=1).std().fillna(0)
            ax.plot(ep["episode"], smooth, label=f"$N_a$={n_act}",
                    linewidth=1.8, color=action_colors[n_act])
            ax.fill_between(ep["episode"], smooth - std, smooth + std,
                            alpha=0.15, color=action_colors[n_act])
        ax.set_title(ALGO_DISPLAY[algo], fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig4_action_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: State Resolution Sweep (2×2)
# ─────────────────────────────────────────────────────────────────────────────

def make_fig5(output_dir: str):
    suite3_dir = os.path.join(ROOT, "experiments", "suite_3_state")
    configs = [
        ("low_1_4_1_4", "Low [1,4,1,4]", "#1f77b4"),
        ("mid_1_8_1_8", "Mid [1,8,1,8]", "#ff7f0e"),
        ("high_2_16_2_16", "High [2,16,2,16]", "#d62728"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("State Resolution Sensitivity (Episode Total Reward)",
                 fontsize=14, fontweight="bold")

    for idx, algo in enumerate(ALGOS):
        r, c = divmod(idx, 2)
        ax = axes[r, c]
        for cfg_label, display, color in configs:
            csv_path = os.path.join(suite3_dir, f"{algo}_{cfg_label}.csv")
            if not os.path.isfile(csv_path):
                print(f"  WARNING: Missing {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            ep = episode_blocks(df)
            w = max(1, len(ep) // 15)
            smooth = rolling_mean(ep["sum_reward"], w)
            std = ep["sum_reward"].rolling(w, min_periods=1).std().fillna(0)
            ax.plot(ep["episode"], smooth, label=display,
                    linewidth=1.8, color=color)
            ax.fill_between(ep["episode"], smooth - std, smooth + std,
                            alpha=0.15, color=color)
        ax.set_title(ALGO_DISPLAY[algo], fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig5_state_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Deployment Stability (2×2 phase portraits)
# ─────────────────────────────────────────────────────────────────────────────

def make_fig6(output_dir: str):
    traj_dir = os.path.join(ROOT, "experiments", "suite_4_deployment", "trajectories")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Deployment Phase Portraits (Best Episode, ε=0)",
                 fontsize=14, fontweight="bold")

    for idx, algo in enumerate(ALGOS):
        r, c = divmod(idx, 2)
        ax = axes[r, c]

        csv_path = os.path.join(traj_dir, f"{algo}_trajectory.csv")
        if not os.path.isfile(csv_path):
            print(f"  WARNING: Missing {csv_path}")
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(ALGO_DISPLAY[algo])
            continue

        df = pd.read_csv(csv_path)
        ep_rewards = df.groupby("episode")["reward"].sum()
        best_ep = ep_rewards.idxmax()
        ep_df = df[df["episode"] == best_ep]

        # Scatter colored by time step
        steps = np.arange(len(ep_df))
        sc = ax.scatter(ep_df["pole_angle"], ep_df["pole_vel"],
                        c=steps, cmap="viridis", s=10, alpha=0.8)
        ax.plot(ep_df["pole_angle"], ep_df["pole_vel"],
                linewidth=0.5, alpha=0.3, color="gray")
        fig.colorbar(sc, ax=ax, label="Time step", shrink=0.8)

        ax.axhline(0, color="k", linewidth=0.5, alpha=0.4)
        ax.axvline(0, color="k", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("Pole Angle (rad)")
        ax.set_ylabel("Pole Ang. Velocity (rad/s)")
        ax.set_title(f"{ALGO_DISPLAY[algo]} (ep {best_ep}, "
                     f"R={ep_rewards[best_ep]:.1f})")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig6_deployment.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate exactly 6 report figures for hw2.tex.")
    p.add_argument(
        "--output", default=os.path.join(ROOT, "figures"),
        help="Output directory (default: figures/).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"Generating 6 report figures → {args.output}/\n")

    print("[1/6] Figure 1: Exploration Feedback Loop")
    make_fig1(args.output)

    print("[2/6] Figure 2: Credit Assignment Bottleneck")
    make_fig2(args.output)

    print("[3/6] Figure 3: Representation Structure")
    make_fig3(args.output)

    print("[4/6] Figure 4: Action Resolution Sweep")
    make_fig4(args.output)

    print("[5/6] Figure 5: State Resolution Sweep")
    make_fig5(args.output)

    print("[6/6] Figure 6: Deployment Phase Portraits")
    make_fig6(args.output)

    print(f"\nDone. 6 figures saved to: {args.output}/")


if __name__ == "__main__":
    main()
