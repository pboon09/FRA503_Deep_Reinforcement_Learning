#!/usr/bin/env python3
"""Visualize Q-table from saved JSON files.

Generates:
  1. 3D surface plot: V(s) = max_a Q(s,a)  over  pole_angle x pole_velocity
  2. 2D policy heatmap: argmax_a Q(s,a)  over  pole_angle x cart_position
  3. 2D policy heatmap: argmax_a Q(s,a)  over  pole_angle x pole_velocity

Usage:
    # Single Q-table
    python scripts/RL_Algorithm/visualize/plot_q_surface.py \\
        --qtable q_value/Stabilize/Q_Learning/Q_Learning_10000_2_1.0_10_10.json \\
        --output figures/Q_Learning/

    # Compare multiple algorithms (pass one JSON per algorithm)
    python scripts/RL_Algorithm/visualize/plot_q_surface.py \\
        --qtable q_value/Stabilize/MC/MC_10000_*.json \\
                 q_value/Stabilize/SARSA/SARSA_10000_*.json \\
        --output figures/comparison/ --show
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Q-table loading
# ─────────────────────────────────────────────────────────────────────────────

def load_qtable(path: str) -> dict[tuple, list[float]]:
    """Load a Q-table JSON and return {(c,p,cv,pv): [q0,q1,...]} dict."""
    with open(path, "r") as f:
        data = json.load(f)

    q_values = data["q_values"]
    result: dict[tuple, list[float]] = {}
    for key_str, vals in q_values.items():
        key_str = key_str.replace("(", "").replace(")", "")
        state = tuple(float(x) for x in key_str.split(", "))
        result[state] = vals
    return result


def label_from_path(path: str) -> str:
    """Extract algorithm name from Q-table path.

    Prefers the filename stem (e.g. 'MC' from 'MC.json').
    Falls back to parent directory if the stem contains underscores with digits
    (like the auto-generated Q-table filenames).
    """
    stem = Path(path).stem
    parts = Path(path).parts
    # Auto-generated names look like "MC_10000_100_10.0_1_8" — use parent dir
    # Manually named files like "MC.json" — use the stem
    if len(parts) >= 3 and sum(c.isdigit() for c in stem) > len(stem) // 2:
        return parts[-2]
    return stem


# ─────────────────────────────────────────────────────────────────────────────
# Data aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_v_surface(qtable: dict, dim_x: int, dim_y: int) -> tuple:
    """Aggregate V(s)=max_a Q(s,a) over two chosen state dimensions.

    For each unique (dim_x, dim_y) pair, averages V across all other dimensions.

    Args:
        qtable: {state_tuple: [q_values]}
        dim_x: state dimension index for x-axis
        dim_y: state dimension index for y-axis

    Returns:
        (x_vals, y_vals, v_grid) as numpy arrays for surface plotting.
    """
    # Collect V for each (x, y) pair
    v_map: dict[tuple, list[float]] = {}
    for state, q_vals in qtable.items():
        key = (state[dim_x], state[dim_y])
        v = max(q_vals)
        v_map.setdefault(key, []).append(v)

    # Average across other dimensions
    points = {}
    for (x, y), vs in v_map.items():
        points[(x, y)] = np.mean(vs)

    if not points:
        return np.array([]), np.array([]), np.array([[]])

    xs = sorted(set(k[0] for k in points))
    ys = sorted(set(k[1] for k in points))

    v_grid = np.full((len(ys), len(xs)), np.nan)
    for (x, y), v in points.items():
        xi = xs.index(x)
        yi = ys.index(y)
        v_grid[yi, xi] = v

    return np.array(xs), np.array(ys), v_grid


def aggregate_policy(qtable: dict, dim_x: int, dim_y: int,
                     action_range: tuple[float, float] = (-10.0, 10.0)) -> tuple:
    """Aggregate best action (argmax Q) over two chosen state dimensions.

    For each (dim_x, dim_y), picks the most common best action across
    all other state dimensions (majority vote).  Unvisited states
    (sum |Q(s,a)| == 0) are masked with NaN.

    The returned action grid contains mapped **force values** (not raw
    indices), so the colorbar shows physical units (Newtons).

    Returns:
        (x_vals, y_vals, force_grid, action_range)
    """
    from collections import Counter

    action_map: dict[tuple, list[int]] = {}
    n_actions = 0
    for state, q_vals in qtable.items():
        n_actions = max(n_actions, len(q_vals))
        key = (state[dim_x], state[dim_y])
        # Skip unvisited states: all Q-values are zero
        if sum(abs(v) for v in q_vals) == 0:
            continue
        best_a = int(np.argmax(q_vals))
        action_map.setdefault(key, []).append(best_a)

    # Majority vote
    policy = {}
    for key, actions in action_map.items():
        c = Counter(actions)
        policy[key] = c.most_common(1)[0][0]

    if not policy:
        return np.array([]), np.array([]), np.array([[]]), action_range

    xs = sorted(set(k[0] for k in policy))
    ys = sorted(set(k[1] for k in policy))

    # Map action index → physical force
    a_min, a_max = action_range
    force_grid = np.full((len(ys), len(xs)), np.nan)
    for (x, y), a in policy.items():
        xi = xs.index(x)
        yi = ys.index(y)
        force = a_min + (a_max - a_min) * a / max(n_actions - 1, 1)
        force_grid[yi, xi] = force

    return np.array(xs), np.array(ys), force_grid, action_range


# ─────────────────────────────────────────────────────────────────────────────
# State dimension indices
# ─────────────────────────────────────────────────────────────────────────────
# State tuple: (cart_pos_dis, pole_angle_dis, cart_vel_dis, pole_vel_dis)
DIM_CART_POS  = 0
DIM_POLE_ANG  = 1
DIM_CART_VEL  = 2
DIM_POLE_VEL  = 3

DIM_NAMES = {
    0: "Cart Position",
    1: "Pole Angle",
    2: "Cart Velocity",
    3: "Pole Angular Velocity",
}


# ─────────────────────────────────────────────────────────────────────────────
# Plot functions
# ─────────────────────────────────────────────────────────────────────────────

def save_3d_surface(out_dir: str, label: str, qtable: dict):
    """3D surface: V(s) = max_a Q(s,a) over pole_angle x pole_velocity."""
    xs, ys, v_grid = aggregate_v_surface(qtable, DIM_POLE_ANG, DIM_POLE_VEL)
    if xs.size == 0:
        print(f"  WARNING: No data for 3D surface ({label})")
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    X, Y = np.meshgrid(xs, ys)
    # Fill NaN with nearest for smoother surface
    v_filled = np.nan_to_num(v_grid, nan=0.0)

    ax.plot_surface(X, Y, v_filled, cmap="viridis", alpha=0.85, edgecolor="none")
    ax.set_xlabel(DIM_NAMES[DIM_POLE_ANG])
    ax.set_ylabel(DIM_NAMES[DIM_POLE_VEL])
    ax.set_zlabel("V(s) = max Q(s,a)")
    ax.set_title(f"Value Function Surface — {label}")
    ax.view_init(elev=30, azim=-120)

    fig.tight_layout()
    path = os.path.join(out_dir, "q_surface_3d.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_policy_heatmap_angle_pos(out_dir: str, label: str, qtable: dict):
    """Policy heatmap: argmax Q over pole_angle x cart_position.

    Unvisited states are shown in gray; color encodes force in Newtons.
    """
    xs, ys, force_grid, a_range = aggregate_policy(
        qtable, DIM_POLE_ANG, DIM_CART_POS)
    if xs.size == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("#d9d9d9")          # gray background for unvisited
    cmap = cm.get_cmap("RdYlGn")
    im = ax.pcolormesh(xs, ys, force_grid, cmap=cmap,
                       vmin=a_range[0], vmax=a_range[1])
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Best Action Force (N)")
    cb.set_ticks([a_range[0], 0, a_range[1]])
    cb.set_ticklabels([f"{a_range[0]:.0f}N (Left)", "0N", f"{a_range[1]:.0f}N (Right)"])
    ax.set_xlabel(DIM_NAMES[DIM_POLE_ANG])
    ax.set_ylabel(DIM_NAMES[DIM_CART_POS])
    ax.set_title(f"Learned Policy — {label}")
    fig.tight_layout()
    path = os.path.join(out_dir, "policy_angle_pos.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_policy_heatmap_angle_vel(out_dir: str, label: str, qtable: dict):
    """Policy heatmap: argmax Q over pole_angle x pole_velocity.

    Unvisited states are shown in gray; color encodes force in Newtons.
    """
    xs, ys, force_grid, a_range = aggregate_policy(
        qtable, DIM_POLE_ANG, DIM_POLE_VEL)
    if xs.size == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("#d9d9d9")          # gray background for unvisited
    cmap = cm.get_cmap("RdYlGn")
    im = ax.pcolormesh(xs, ys, force_grid, cmap=cmap,
                       vmin=a_range[0], vmax=a_range[1])
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Best Action Force (N)")
    cb.set_ticks([a_range[0], 0, a_range[1]])
    cb.set_ticklabels([f"{a_range[0]:.0f}N (Left)", "0N", f"{a_range[1]:.0f}N (Right)"])
    ax.set_xlabel(DIM_NAMES[DIM_POLE_ANG])
    ax.set_ylabel(DIM_NAMES[DIM_POLE_VEL])
    ax.set_title(f"Learned Policy — {label}")
    fig.tight_layout()
    path = os.path.join(out_dir, "policy_angle_vel.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_v_heatmap(out_dir: str, label: str, qtable: dict):
    """2D heatmap of V(s) = max_a Q(s,a) over pole_angle x pole_velocity."""
    xs, ys, v_grid = aggregate_v_surface(qtable, DIM_POLE_ANG, DIM_POLE_VEL)
    if xs.size == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    import seaborn as sns
    v_filled = np.nan_to_num(v_grid, nan=0.0)
    sns.heatmap(v_filled, ax=ax, cmap="viridis",
                xticklabels=[f"{x:.0f}" for x in xs] if len(xs) <= 30 else False,
                yticklabels=[f"{y:.0f}" for y in ys] if len(ys) <= 30 else False,
                cbar_kws={"label": "V(s)"})
    ax.set_xlabel(DIM_NAMES[DIM_POLE_ANG])
    ax.set_ylabel(DIM_NAMES[DIM_POLE_VEL])
    ax.set_title(f"Value Function Heatmap — {label}")
    fig.tight_layout()
    path = os.path.join(out_dir, "v_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_report_policy_comparison(out_dir: str, datasets: list[tuple[str, dict]]):
    """2x2 subplot: policy heatmap (angle × velocity) for each algorithm."""
    n = len(datasets)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    fig.suptitle("Learned Policy Comparison (Pole Angle × Angular Velocity)",
                 fontsize=14, fontweight="bold")

    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (label, qtable) in enumerate(datasets):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        ax.set_facecolor("#d9d9d9")
        xs, ys, force_grid, a_range = aggregate_policy(
            qtable, DIM_POLE_ANG, DIM_POLE_VEL)
        if xs.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            cmap_obj = cm.get_cmap("RdYlGn")
            im = ax.pcolormesh(xs, ys, force_grid, cmap=cmap_obj,
                               vmin=a_range[0], vmax=a_range[1])
            fig.colorbar(im, ax=ax, label="Force (N)")
        ax.set_xlabel("Pole Angle")
        ax.set_ylabel("Pole Angular Velocity")
        ax.set_title(label)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, "report_policy_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_report_value_comparison(out_dir: str, datasets: list[tuple[str, dict]]):
    """2x2 subplot: V(s) heatmap (angle × velocity) for each algorithm."""
    import seaborn as sns
    n = len(datasets)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    fig.suptitle("Value Function Comparison V(s) = max Q(s,a)",
                 fontsize=14, fontweight="bold")

    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (label, qtable) in enumerate(datasets):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        xs, ys, v_grid = aggregate_v_surface(qtable, DIM_POLE_ANG, DIM_POLE_VEL)
        if xs.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            v_filled = np.nan_to_num(v_grid, nan=0.0)
            im = ax.pcolormesh(xs, ys, v_filled, cmap="viridis")
            fig.colorbar(im, ax=ax, label="V(s)")
        ax.set_xlabel("Pole Angle")
        ax.set_ylabel("Pole Angular Velocity")
        ax.set_title(label)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, "report_value_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_comparison_surface(out_dir: str, datasets: list[tuple[str, dict]]):
    """Side-by-side 3D surfaces for multiple algorithms."""
    n = len(datasets)
    if n < 2:
        return

    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(8 * ncols, 6 * nrows))
    fig.suptitle("Value Function Comparison", fontsize=14)

    for idx, (label, qtable) in enumerate(datasets):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        xs, ys, v_grid = aggregate_v_surface(qtable, DIM_POLE_ANG, DIM_POLE_VEL)
        if xs.size == 0:
            continue
        X, Y = np.meshgrid(xs, ys)
        v_filled = np.nan_to_num(v_grid, nan=0.0)
        ax.plot_surface(X, Y, v_filled, cmap="viridis", alpha=0.85, edgecolor="none")
        ax.set_xlabel("Pole Angle")
        ax.set_ylabel("Pole Ang. Vel.")
        ax.set_zlabel("V(s)")
        ax.set_title(label)
        ax.view_init(elev=30, azim=-120)

    fig.tight_layout()
    path = os.path.join(out_dir, "q_surface_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize Q-table from saved JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--qtable", nargs="+", required=True, metavar="JSON",
        help="One or more Q-table JSON files (glob patterns accepted).",
    )
    p.add_argument(
        "--output", default="figures", metavar="DIR",
        help="Output directory for PNG figures (default: figures/).",
    )
    p.add_argument(
        "--show", action="store_true", default=False,
        help="Show interactive matplotlib window.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Expand globs
    resolved: list[str] = []
    for pattern in args.qtable:
        matches = sorted(glob.glob(pattern, recursive=True))
        resolved.extend(matches if matches else [pattern])

    # Load Q-tables
    datasets: list[tuple[str, dict]] = []
    for path in resolved:
        if not os.path.isfile(path):
            print(f"WARNING: Not found, skipping: {path}", file=sys.stderr)
            continue
        print(f"Loading: {path}")
        qtable = load_qtable(path)
        label = label_from_path(path)
        print(f"  States: {len(qtable)}, Actions: {len(next(iter(qtable.values())))}")
        datasets.append((label, qtable))

    if not datasets:
        print("ERROR: No valid Q-table files loaded.", file=sys.stderr)
        sys.exit(1)

    # Make labels unique
    seen: dict[str, int] = {}
    unique_datasets: list[tuple[str, dict]] = []
    for label, qt in datasets:
        if label in seen:
            seen[label] += 1
            label = f"{label}_{seen[label]}"
        else:
            seen[label] = 0
        unique_datasets.append((label, qt))
    datasets = unique_datasets

    # Per-algorithm plots
    for label, qtable in datasets:
        algo_dir = os.path.join(args.output, label)
        os.makedirs(algo_dir, exist_ok=True)
        print(f"\n[{label}] → {algo_dir}/")
        save_3d_surface(algo_dir, label, qtable)
        save_v_heatmap(algo_dir, label, qtable)
        save_policy_heatmap_angle_vel(algo_dir, label, qtable)

    # Comparison plots (if multiple)
    if len(datasets) > 1:
        cmp_dir = os.path.join(args.output, "comparison")
        os.makedirs(cmp_dir, exist_ok=True)
        print(f"\n[comparison] → {cmp_dir}/")
        save_comparison_surface(cmp_dir, datasets)
        save_report_policy_comparison(cmp_dir, datasets)
        save_report_value_comparison(cmp_dir, datasets)

    print("\nDone.")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
