#!/usr/bin/env python3
"""Generate report figures for HW3.

Figures:
  1. fig1_learning_curves.png   — Episode return vs episode (all algos, rolling mean ± IQR)
  2. fig2_episode_length.png    — Episode length vs episode (all algos, rolling mean)
  3. fig3_deployment_bar.png    — Deployment performance bar chart (mean ± std)
  4. fig4_deployment_boxplot.png — Deployment boxplot per algorithm
  5. fig5_epsilon_decay.png     — Epsilon over episodes (value-based algos only)

Data sources:
  - Training:   experiments/suite_1_baseline/{algo}.csv
  - Deployment: experiments/suite_1_baseline/{algo}_deploy.csv

Usage:
    python scripts/Function_based/visualize/plot_report_figures.py
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ALGOS = ["Linear_Q", "DQN", "MC_REINFORCE", "AC", "PPO"]
ALGO_DISPLAY = {
    "Linear_Q": "Linear Q",
    "DQN": "DQN",
    "MC_REINFORCE": "MC REINFORCE",
    "AC": "Actor-Critic",
    "PPO": "PPO",
}
ALGO_COLORS = {
    "Linear_Q": "#1f77b4",
    "DQN": "#ff7f0e",
    "MC_REINFORCE": "#2ca02c",
    "AC": "#d62728",
    "PPO": "#9467bd",
}

# visualize → Function_based → scripts → CartPole_4.5.0
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

EXP_DIR = os.path.join(ROOT, "experiments", "suite_1_baseline")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_training_csv(algo):
    path = os.path.join(EXP_DIR, f"{algo}.csv")
    if os.path.isfile(path):
        return pd.read_csv(path)
    return None


def load_deploy_csv(algo):
    path = os.path.join(EXP_DIR, f"{algo}_deploy.csv")
    if os.path.isfile(path):
        return pd.read_csv(path)
    return None


def rolling(series, w=100):
    return series.rolling(window=w, min_periods=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Learning Curves — Episode Return vs Episode
# ─────────────────────────────────────────────────────────────────────────────

def make_fig1(output_dir, window=100):
    fig, ax = plt.subplots(figsize=(14, 6))
    plotted = False

    for algo in ALGOS:
        df = load_training_csv(algo)
        if df is None or "ep_return" not in df.columns:
            print(f"  [fig1] No ep_return for {algo}, skipping.")
            continue

        y = df["ep_return"]
        smoothed = rolling(y, window)
        ax.plot(smoothed.values, label=ALGO_DISPLAY[algo],
                color=ALGO_COLORS[algo], alpha=0.9, linewidth=1.5)
        # IQR shading
        q25 = y.rolling(window, min_periods=1).quantile(0.25)
        q75 = y.rolling(window, min_periods=1).quantile(0.75)
        ax.fill_between(range(len(smoothed)), q25.values, q75.values,
                        alpha=0.12, color=ALGO_COLORS[algo])
        plotted = True

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Return (rolling mean)", fontsize=12)
    ax.set_title("Learning Efficiency: Episode Return vs Training Episode", fontsize=14)
    if plotted:
        ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_learning_curves.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig1_learning_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Episode Length vs Episode
# ─────────────────────────────────────────────────────────────────────────────

def make_fig2(output_dir, window=100):
    fig, ax = plt.subplots(figsize=(14, 6))
    plotted = False

    for algo in ALGOS:
        df = load_training_csv(algo)
        if df is None or "ep_length" not in df.columns:
            print(f"  [fig2] No ep_length for {algo}, skipping.")
            continue

        y = df["ep_length"]
        smoothed = rolling(y, window)
        ax.plot(smoothed.values, label=ALGO_DISPLAY[algo],
                color=ALGO_COLORS[algo], alpha=0.9, linewidth=1.5)
        plotted = True

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Length (steps, rolling mean)", fontsize=12)
    ax.set_title("Learning Efficiency: Episode Duration vs Training Episode", fontsize=14)
    if plotted:
        ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_episode_length.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig2_episode_length.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Deployment Performance — Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def make_fig3(output_dir):
    means, stds, names, colors = [], [], [], []

    for algo in ALGOS:
        df = load_deploy_csv(algo)
        if df is None or "ep_return" not in df.columns:
            continue
        means.append(df["ep_return"].mean())
        stds.append(df["ep_return"].std())
        names.append(ALGO_DISPLAY[algo])
        colors.append(ALGO_COLORS[algo])

    if not means:
        print("  [fig3] No deployment data found, skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, color=colors, capsize=6, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Mean Episode Return", fontsize=12)
    ax.set_title("Deployment Performance (Deterministic Policy)", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 1, f"{m:.1f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_deployment_bar.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig3_deployment_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Deployment Boxplot
# ─────────────────────────────────────────────────────────────────────────────

def make_fig4(output_dir):
    data_list, labels = [], []

    for algo in ALGOS:
        df = load_deploy_csv(algo)
        if df is None or "ep_return" not in df.columns:
            continue
        data_list.append(df["ep_return"].values)
        labels.append(ALGO_DISPLAY[algo])

    if not data_list:
        print("  [fig4] No deployment data found, skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data_list, labels=labels, patch_artist=True, showmeans=True)
    for patch, algo in zip(bp["boxes"], [a for a in ALGOS if load_deploy_csv(a) is not None]):
        patch.set_facecolor(ALGO_COLORS[algo])
        patch.set_alpha(0.7)

    ax.set_ylabel("Episode Return", fontsize=12)
    ax.set_title("Deployment Performance Distribution", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4_deployment_boxplot.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig4_deployment_boxplot.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Epsilon Decay (value-based only)
# ─────────────────────────────────────────────────────────────────────────────

def make_fig5(output_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    plotted = False

    for algo in ["Linear_Q", "DQN"]:
        df = load_training_csv(algo)
        if df is None or "epsilon" not in df.columns:
            continue
        ax.plot(df["epsilon"].values, label=ALGO_DISPLAY[algo],
                color=ALGO_COLORS[algo], alpha=0.9, linewidth=1.5)
        plotted = True

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Epsilon", fontsize=12)
    ax.set_title("Exploration Rate (ε) Decay Over Training", fontsize=14)
    if plotted:
        ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig5_epsilon_decay.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig5_epsilon_decay.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(ROOT, "figures"))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Generating figures... (ROOT={ROOT})")
    print(f"  Looking for CSVs in: {EXP_DIR}")

    make_fig1(args.output)     # Learning curves (return)
    make_fig2(args.output)     # Episode length
    make_fig3(args.output)     # Deployment bar
    make_fig4(args.output)     # Deployment boxplot
    make_fig5(args.output)     # Epsilon decay

    print("Done.")


if __name__ == "__main__":
    main()
