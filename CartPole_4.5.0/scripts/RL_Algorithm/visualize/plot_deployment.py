#!/usr/bin/env python3
"""Generate deployment evaluation bar charts from evaluation_results.csv.

Usage:
    python scripts/RL_Algorithm/visualize/plot_deployment.py \
        --csv experiments/suite_4_deployment/evaluation_results.csv \
        --output figures/suite_4_deployment/
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Plot deployment evaluation results.")
    p.add_argument("--csv", required=True, help="Path to evaluation_results.csv")
    p.add_argument("--output", default="figures/suite_4_deployment",
                   help="Output directory for plots.")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.csv):
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    os.makedirs(args.output, exist_ok=True)

    algorithms = df["algorithm"].tolist()
    mean_rewards = df["mean_reward"].tolist()
    mean_lengths = df["mean_length"].tolist()

    x = np.arange(len(algorithms))
    width = 0.5
    colors = [plt.cm.Set2(i) for i in range(len(algorithms))]

    # --- Bar chart: Mean Reward ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, mean_rewards, width, color=colors)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Deployment: Mean Reward (10 episodes, epsilon=0)")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=15)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, mean_rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    path = os.path.join(args.output, "deployment_mean_reward.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Bar chart: Mean Length ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, mean_lengths, width, color=colors)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Mean Episode Length")
    ax.set_title("Deployment: Mean Episode Length (10 episodes, epsilon=0)")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=15)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, mean_lengths):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    path = os.path.join(args.output, "deployment_mean_length.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Combined side-by-side ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Deployment Evaluation Summary (10 episodes, epsilon=0)",
                 fontsize=14, fontweight="bold")

    ax1.bar(x, mean_rewards, width, color=colors)
    ax1.set_xlabel("Algorithm")
    ax1.set_ylabel("Mean Reward")
    ax1.set_title("(a) Mean Reward")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, rotation=15)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x, mean_lengths, width, color=colors)
    ax2.set_xlabel("Algorithm")
    ax2.set_ylabel("Mean Episode Length")
    ax2.set_title("(b) Mean Episode Length")
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, rotation=15)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(args.output, "deployment_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
