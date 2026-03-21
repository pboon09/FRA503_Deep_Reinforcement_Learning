#!/usr/bin/env python3
"""Generate report figures for HW3.

Figures:
  1. fig1_sample_efficiency.png  — Return vs total env steps (fair x-axis)
  2. fig2_learning_curves.png    — Return vs episode (standard view)
  3. fig3_deployment_bar.png     — Deployment mean ± std bar chart
  4. fig4_convergence_speed.png  — Steps-to-threshold horizontal bar
  5. fig5_reward_per_step.png    — Reward efficiency (return/length) over episodes

Data:
  - Training:   experiments/suite_1_baseline/{algo}.csv
  - Deployment: experiments/suite_1_baseline/{algo}_deploy.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
ALGOS = ["Linear_Q", "DQN", "MC_REINFORCE", "AC", "PPO"]
ALGO_DISPLAY = {
    "Linear_Q": "Linear Q", "DQN": "DQN", "MC_REINFORCE": "MC REINFORCE",
    "AC": "Actor-Critic", "PPO": "PPO",
}
ALGO_COLORS = {
    "Linear_Q": "#1f77b4", "DQN": "#ff7f0e", "MC_REINFORCE": "#2ca02c",
    "AC": "#d62728", "PPO": "#9467bd",
}

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.join(ROOT, "experiments", "suite_1_baseline")


def load_csv(algo):
    p = os.path.join(EXP_DIR, f"{algo}.csv")
    return pd.read_csv(p) if os.path.isfile(p) else None


def load_deploy(algo):
    p = os.path.join(EXP_DIR, f"{algo}_deploy.csv")
    return pd.read_csv(p) if os.path.isfile(p) else None


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Sample Efficiency — Return vs Total Env Steps
# ─────────────────────────────────────────────────────────────────────────────
def make_fig1(out, window=200):
    fig, ax = plt.subplots(figsize=(14, 6))
    for algo in ALGOS:
        df = load_csv(algo)
        if df is None or "global_step" not in df.columns:
            print(f"  [fig1] No global_step for {algo}, skipping.")
            continue
        x = df["global_step"].values
        y = df["ep_return"].rolling(window, min_periods=1).mean().values
        ax.plot(x, y, label=ALGO_DISPLAY[algo], color=ALGO_COLORS[algo], linewidth=1.5, alpha=0.9)
    ax.set_xlabel("Total Environment Steps", fontsize=12)
    ax.set_ylabel("Episode Return (rolling mean)", fontsize=12)
    ax.set_title("Sample Efficiency: Return vs Environment Steps", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_sample_efficiency.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig1_sample_efficiency.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Learning Curves — Return vs Episode
# ─────────────────────────────────────────────────────────────────────────────
def make_fig2(out, window=200):
    fig, ax = plt.subplots(figsize=(14, 6))
    for algo in ALGOS:
        df = load_csv(algo)
        if df is None or "ep_return" not in df.columns:
            continue
        y = df["ep_return"]
        smoothed = y.rolling(window, min_periods=1).mean()
        ax.plot(smoothed.values, label=ALGO_DISPLAY[algo], color=ALGO_COLORS[algo], linewidth=1.5, alpha=0.9)
        q25 = y.rolling(window, min_periods=1).quantile(0.25)
        q75 = y.rolling(window, min_periods=1).quantile(0.75)
        ax.fill_between(range(len(smoothed)), q25.values, q75.values,
                        alpha=0.1, color=ALGO_COLORS[algo])
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Return (rolling mean ± IQR)", fontsize=12)
    ax.set_title("Learning Efficiency: Return vs Training Episode", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_learning_curves.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig2_learning_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Deployment Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def make_fig3(out):
    means, stds, names, colors = [], [], [], []
    for algo in ALGOS:
        df = load_deploy(algo)
        if df is None or "ep_return" not in df.columns:
            continue
        means.append(df["ep_return"].mean())
        stds.append(df["ep_return"].std())
        names.append(ALGO_DISPLAY[algo])
        colors.append(ALGO_COLORS[algo])
    if not means:
        print("  [fig3] No deployment data, skipping.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=6, alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Mean Episode Return", fontsize=12)
    ax.set_title("Deployment Performance (Deterministic Policy)", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 2, f"{m:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_deployment_bar.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig3_deployment_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Convergence Speed — Steps to reach threshold
# ─────────────────────────────────────────────────────────────────────────────
def make_fig4(out, threshold=200, window=100):
    results = {}
    for algo in ALGOS:
        df = load_csv(algo)
        if df is None or "global_step" not in df.columns:
            continue
        smoothed = df["ep_return"].rolling(window, min_periods=1).mean()
        reached = smoothed[smoothed >= threshold]
        if len(reached) > 0:
            idx = reached.index[0]
            results[algo] = df["global_step"].iloc[idx]
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
    bars = ax.barh(y_pos, steps, color=colors_list, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Total Environment Steps to Reach Threshold", fontsize=12)
    ax.set_title(f"Convergence Speed (threshold = {threshold} return)", fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")

    for i, (bar, s) in enumerate(zip(bars, steps)):
        if s > 0:
            ax.text(s + ax.get_xlim()[1] * 0.01, i, f"{s:,}", va="center", fontsize=10)
        else:
            ax.text(ax.get_xlim()[1] * 0.01, i, "DNF", va="center", fontsize=10, color="red")

    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_convergence_speed.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig4_convergence_speed.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Reward Efficiency — Return per Step
# ─────────────────────────────────────────────────────────────────────────────
def make_fig5(out, window=200):
    fig, ax = plt.subplots(figsize=(14, 6))
    for algo in ALGOS:
        df = load_csv(algo)
        if df is None or "ep_return" not in df.columns or "ep_length" not in df.columns:
            continue
        rps = df["ep_return"] / df["ep_length"].clip(lower=1)
        smoothed = rps.rolling(window, min_periods=1).mean()
        ax.plot(smoothed.values, label=ALGO_DISPLAY[algo], color=ALGO_COLORS[algo], linewidth=1.5, alpha=0.9)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Return per Step (rolling mean)", fontsize=12)
    ax.set_title("Reward Efficiency: Average Reward per Timestep", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_reward_per_step.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig5_reward_per_step.png")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(ROOT, "figures"))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Generating figures... (ROOT={ROOT})")
    print(f"  Looking for CSVs in: {EXP_DIR}")

    make_fig1(args.output)  # Sample efficiency (return vs env steps)
    make_fig2(args.output)  # Learning curves (return vs episode)
    make_fig3(args.output)  # Deployment bar chart
    make_fig4(args.output)  # Convergence speed
    make_fig5(args.output)  # Reward per step

    print("Done.")


if __name__ == "__main__":
    main()
