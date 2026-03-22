#!/usr/bin/env python3
"""Generate report figures for HW3 (HW2-style formatting).

Figures:
  1. fig1_learning_curves.png    — Return vs episode (all algos, mean ± std band)
  2. fig2_sample_efficiency.png  — Return vs total env steps (fair x-axis)
  3. fig3_deployment.png         — (a) bar chart + (b) per-episode scatter
  4. fig4_convergence_speed.png  — Horizontal bar: steps to reach threshold
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
# Style (matches HW2)
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 1.8,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

ALGOS = ["Linear_Q", "DQN", "MC_REINFORCE", "AC", "PPO"]
ALGO_DISPLAY = {
    "Linear_Q": "Linear Q", "DQN": "DQN", "MC_REINFORCE": "MC REINFORCE",
    "AC": "Actor-Critic", "PPO": "PPO",
}
ALGO_COLORS = {
    "Linear_Q": "#0072B2", "DQN": "#E69F00", "MC_REINFORCE": "#009E73",
    "AC": "#D55E00", "PPO": "#CC79A7",
}
ALGO_LINESTYLES = {
    "Linear_Q": "-", "DQN": "-", "MC_REINFORCE": "-", "AC": "-", "PPO": "-",
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


def adaptive_window(n):
    return max(1, n // 15)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Learning Curves — Return vs Episode (HW2 fig1 style)
# ─────────────────────────────────────────────────────────────────────────────
def make_fig1(out):
    fig, ax = plt.subplots(figsize=(14, 5))
    for algo in ALGOS:
        df = load_csv(algo)
        if df is None or "ep_return" not in df.columns:
            continue
        y = df["ep_return"]
        w = adaptive_window(len(y))
        smoothed = y.rolling(w, min_periods=1).mean()
        std = y.rolling(w, min_periods=1).std().fillna(0)
        ax.plot(smoothed.values, label=ALGO_DISPLAY[algo],
                color=ALGO_COLORS[algo], linestyle=ALGO_LINESTYLES[algo])
        ax.fill_between(range(len(smoothed)),
                        (smoothed - std).values, (smoothed + std).values,
                        alpha=0.15, color=ALGO_COLORS[algo])
    ax.axhline(y=950, color="gray", linestyle="--", alpha=0.5, label="Near-optimal (950)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Return (rolling mean ± std)")
    ax.set_title("Learning Efficiency: Return vs Training Episode", fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_learning_curves.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_learning_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Sample Efficiency — Return vs Total Env Steps
# ─────────────────────────────────────────────────────────────────────────────
def make_fig2(out):
    fig, ax = plt.subplots(figsize=(14, 5))
    for algo in ALGOS:
        df = load_csv(algo)
        if df is None or "global_step" not in df.columns:
            continue
        y = df["ep_return"]
        w = adaptive_window(len(y))
        smoothed = y.rolling(w, min_periods=1).mean()
        ax.plot(df["global_step"].values, smoothed.values,
                label=ALGO_DISPLAY[algo], color=ALGO_COLORS[algo],
                linestyle=ALGO_LINESTYLES[algo])
    ax.axhline(y=950, color="gray", linestyle="--", alpha=0.5, label="Near-optimal (950)")
    ax.set_xlabel("Total Environment Steps")
    ax.set_ylabel("Episode Return (rolling mean)")
    ax.set_title("Sample Efficiency: Return vs Environment Steps", fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_sample_efficiency.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_sample_efficiency.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Deployment — (a) bar chart + (b) per-episode scatter (HW2 fig6b style)
# ─────────────────────────────────────────────────────────────────────────────
def make_fig3(out):
    deploy_data = {}
    for algo in ALGOS:
        df = load_deploy(algo)
        if df is not None and "ep_return" in df.columns:
            deploy_data[algo] = df

    if not deploy_data:
        print("  [fig3] No deployment data, skipping.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    gridspec_kw={"width_ratios": [1, 1.3]})

    # Panel (a): Bar chart
    names, means, stds, colors = [], [], [], []
    for algo in ALGOS:
        if algo not in deploy_data:
            continue
        df = deploy_data[algo]
        names.append(ALGO_DISPLAY[algo])
        means.append(df["ep_return"].mean())
        stds.append(df["ep_return"].std())
        colors.append(ALGO_COLORS[algo])

    x = np.arange(len(names))
    ax1.bar(x, means, yerr=stds, color=colors, capsize=6,
            alpha=0.85, edgecolor="black", linewidth=0.8, width=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9, rotation=15)
    ax1.set_ylabel("Episode Return (mean ± std)")
    ax1.set_title("(a) Deployment Performance", fontweight="bold", fontsize=12)
    ax1.set_ylim(bottom=0)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.text(i, m + s + max(10, ax1.get_ylim()[1] * 0.02),
                 f"{m:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel (b): Per-episode scatter
    for algo in ALGOS:
        if algo not in deploy_data:
            continue
        df = deploy_data[algo]
        eps = df["episode"].values if "episode" in df.columns else np.arange(len(df))
        ax2.plot(eps, df["ep_return"].values, marker="o", markersize=6,
                 linewidth=1.5, label=ALGO_DISPLAY[algo], color=ALGO_COLORS[algo],
                 alpha=0.8)
    ax2.axhline(y=950, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Evaluation Episode")
    ax2.set_ylabel("Episode Return")
    ax2.set_title("(b) Per-Episode Consistency", fontweight="bold", fontsize=12)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=9)

    fig.suptitle("Deployment Performance (Deterministic Policy, 10 Episodes)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_deployment.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_deployment.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Convergence Speed — Steps to reach threshold
# ─────────────────────────────────────────────────────────────────────────────
def make_fig4(out, threshold=200):
    results = {}
    for algo in ALGOS:
        df = load_csv(algo)
        if df is None or "global_step" not in df.columns:
            continue
        w = adaptive_window(len(df))
        smoothed = df["ep_return"].rolling(w, min_periods=1).mean()
        reached = smoothed[smoothed >= threshold]
        if len(reached) > 0:
            results[algo] = df["global_step"].iloc[reached.index[0]]
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
    bars = ax.barh(y_pos, steps, color=colors_list, alpha=0.85,
                   edgecolor="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Total Environment Steps to Reach Threshold")
    ax.set_title(f"Convergence Speed (threshold = {threshold} return)",
                 fontweight="bold")
    for i, (bar, s) in enumerate(zip(bars, steps)):
        if s > 0:
            ax.text(s + ax.get_xlim()[1] * 0.01, i, f"{s:,}",
                    va="center", fontsize=10)
        else:
            ax.text(ax.get_xlim()[1] * 0.5, i, "Did Not Reach",
                    va="center", fontsize=10, color="red", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_convergence_speed.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_convergence_speed.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Reward Efficiency — Return per Step
# ─────────────────────────────────────────────────────────────────────────────
def make_fig5(out):
    fig, ax = plt.subplots(figsize=(14, 5))
    for algo in ALGOS:
        df = load_csv(algo)
        if df is None or "ep_return" not in df.columns or "ep_length" not in df.columns:
            continue
        rps = df["ep_return"] / df["ep_length"].clip(lower=1)
        w = adaptive_window(len(rps))
        smoothed = rps.rolling(w, min_periods=1).mean()
        ax.plot(smoothed.values, label=ALGO_DISPLAY[algo],
                color=ALGO_COLORS[algo], linestyle=ALGO_LINESTYLES[algo])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return per Step (rolling mean)")
    ax.set_title("Reward Efficiency: Average Reward per Timestep", fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_reward_per_step.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig5_reward_per_step.png")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(ROOT, "figures"))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Generating figures... (ROOT={ROOT})")
    print(f"  CSVs in: {EXP_DIR}")

    make_fig1(args.output)
    make_fig2(args.output)
    make_fig3(args.output)
    make_fig4(args.output)
    make_fig5(args.output)

    print("Done.")


if __name__ == "__main__":
    main()
