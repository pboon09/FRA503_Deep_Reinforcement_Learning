#!/usr/bin/env python3
"""Generate area-style plots for CartPole tabular RL training logs.

Two styles:
  A) Rolling mean ± std shaded band
  B) Line + fill-to-zero area

Metrics (all with episode on x-axis):
  1. Episode total reward
  2. Episode length (steps)
  3. Discounted return per episode
  4. Pole angle variance per episode
  5. Max Q-value per episode

Usage:
    python plot_area.py --logs exp1.csv exp2.csv --output figures/area_plots/
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_GAMMA = 0.99

def set_gamma(g):
    global _GAMMA
    _GAMMA = g

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def label_from_path(path: str) -> str:
    stem = Path(path).stem
    if not stem.startswith("training_log"):
        return stem
    parts = Path(path).parts
    return parts[-2] if len(parts) >= 3 else stem


def episode_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-episode statistics from step-level CSV."""
    q_cols = [c for c in df.columns if c.startswith("q_")]
    block_id = (df["episode"] != df["episode"].shift()).cumsum()

    def agg_fn(g):
        total_reward = g["reward"].sum()
        steps = len(g)
        # Discounted return
        rewards = g["reward"].values
        discounts = _GAMMA ** np.arange(len(rewards))
        disc_return = (rewards * discounts).sum()
        # Pole angle variance
        pole_var = g["pole_angle"].var() if "pole_angle" in g.columns else 0.0
        # Max Q-value
        max_q = g[q_cols].max().max() if q_cols else 0.0
        return pd.Series({
            "episode": g["episode"].iloc[0],
            "total_reward": total_reward,
            "steps": steps,
            "discounted_return": disc_return,
            "pole_angle_var": pole_var if not np.isnan(pole_var) else 0.0,
            "max_q": max_q,
        })

    result = df.groupby(block_id, sort=False).apply(agg_fn).reset_index(drop=True)
    return result[result["steps"] > 0].reset_index(drop=True)


def rolling_stats(series: pd.Series, window: int):
    """Return rolling mean and std."""
    mean = series.rolling(window, min_periods=1).mean()
    std = series.rolling(window, min_periods=1).std().fillna(0)
    return mean, std


METRIC_CONFIG = {
    "total_reward": {"title": "Episode Total Reward", "ylabel": "Total Reward"},
    "steps": {"title": "Episode Length", "ylabel": "Steps"},
    "discounted_return": {"title": "Discounted Return per Episode", "ylabel": "Discounted Return"},
    "pole_angle_var": {"title": "Pole Angle Variance per Episode", "ylabel": "Variance (rad²)"},
    "max_q": {"title": "Max Q-value per Episode", "ylabel": "Max Q-value"},
}


def plot_band(ax, episodes, values, label, color, window):
    """Style A: rolling mean ± std band."""
    mean, std = rolling_stats(pd.Series(values), window)
    ax.plot(episodes, mean, label=label, color=color, linewidth=1.8)
    ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.2)


def plot_area(ax, episodes, values, label, color, window):
    """Style B: line + fill to zero."""
    mean = pd.Series(values).rolling(window, min_periods=1).mean()
    ax.plot(episodes, mean, label=label, color=color, linewidth=1.8)
    ax.fill_between(episodes, 0, mean, color=color, alpha=0.25)


def generate_plots(datasets, output_dir, window=5):
    """Generate all metric plots in both styles."""
    os.makedirs(output_dir, exist_ok=True)

    # Compute episode stats for each dataset
    ep_data = []
    for label, df in datasets:
        stats = episode_stats(df)
        ep_data.append((label, stats))

    for metric, cfg in METRIC_CONFIG.items():
        for style_name, plot_fn in [("band", plot_band), ("area", plot_area)]:
            fig, ax = plt.subplots(figsize=(10, 5))
            for i, (label, stats) in enumerate(ep_data):
                color = COLORS[i % len(COLORS)]
                plot_fn(ax, stats["episode"].values, stats[metric].values,
                        label, color, window)
            ax.set_title(cfg["title"])
            ax.set_xlabel("Episode")
            ax.set_ylabel(cfg["ylabel"])
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fname = f"{metric}_{style_name}.png"
            path = os.path.join(output_dir, fname)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate area-style RL plots.")
    parser.add_argument("--logs", nargs="+", required=True, help="CSV log files")
    parser.add_argument("--output", default="figures/area_plots", help="Output directory")
    parser.add_argument("--window", type=int, default=5, help="Rolling window size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    args = parser.parse_args()

    set_gamma(args.gamma)

    # Expand globs
    resolved = []
    for pattern in args.logs:
        matches = glob.glob(pattern, recursive=True)
        resolved.extend(matches if matches else [pattern])

    # Load
    datasets = []
    for path in resolved:
        if not os.path.isfile(path):
            print(f"WARNING: Not found: {path}", file=sys.stderr)
            continue
        print(f"Loading: {path}")
        df = pd.read_csv(path)
        datasets.append((label_from_path(path), df))

    if not datasets:
        print("ERROR: No valid CSV files.", file=sys.stderr)
        sys.exit(1)

    generate_plots(datasets, args.output, args.window)
    print("\nDone.")


if __name__ == "__main__":
    main()
