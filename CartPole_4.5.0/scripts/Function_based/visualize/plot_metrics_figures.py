#!/usr/bin/env python3
"""Generate additional report figures from per-update metrics CSVs.

Figures:
  11. fig11_radar_chart.png         — Multi-dimensional algorithm comparison
  12. fig12_loss_curves.png         — Policy/value/TD loss over training
  13. fig13_entropy.png             — Entropy over training
  14. fig14_grad_norm.png           — Gradient norm over training
  15. fig15_explained_variance.png  — Explained variance (critic quality)
  16. fig16_ppo_clip_fraction.png   — PPO clip fraction
  17. fig17_dqn_q_values.png        — DQN mean Q-values over training
  18. fig18_wall_clock.png          — Wall-clock training time comparison
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "lines.linewidth": 1.5, "figure.dpi": 150, "savefig.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.3,
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.join(ROOT, "experiments", "suite_1_baseline")


def load_csv(algo):
    p = os.path.join(EXP_DIR, f"{algo}.csv")
    return pd.read_csv(p) if os.path.isfile(p) else None


def load_metrics(algo):
    p = os.path.join(EXP_DIR, f"{algo}_metrics.csv")
    return pd.read_csv(p) if os.path.isfile(p) else None


def load_deploy(algo):
    p = os.path.join(EXP_DIR, f"{algo}_deploy.csv")
    return pd.read_csv(p) if os.path.isfile(p) else None


def smooth(y, window=50):
    if len(y) < window:
        return y
    return pd.Series(y).rolling(window, min_periods=1, center=True).mean().values


# ─────────────────────────────────────────────────────────────────────────────
# Fig 11: Radar Chart
# ─────────────────────────────────────────────────────────────────────────────
def make_fig11(out):
    # Compute metrics for each algorithm
    metrics = {}
    for algo in ALGOS:
        df = load_csv(algo)
        dep = load_deploy(algo)
        if df is None:
            continue
        rets = df["ep_return"].values
        last2k = rets[-2000:] if len(rets) >= 2000 else rets

        # 1. Final performance (deploy avg / 1000)
        deploy_avg = dep["ep_return"].mean() if dep is not None else 0
        perf = deploy_avg / 1000.0

        # 2. Convergence speed (fraction of training to reach 500 avg)
        roll = pd.Series(rets).rolling(200, min_periods=1).mean()
        reached = np.where(roll > 500)[0]
        if len(reached) > 0:
            conv = 1.0 - reached[0] / len(rets)  # higher = faster
        else:
            conv = 0.0

        # 3. Stability (1 - std/mean of last 2k)
        if np.mean(last2k) > 0:
            stab = max(0, 1.0 - np.std(last2k) / (np.mean(last2k) + 1e-8))
        else:
            stab = 0.0

        # 4. Sample efficiency (deploy / total_steps * 1e7)
        total_steps = df["global_step"].max()
        samp_eff = min(1.0, deploy_avg / (total_steps / 1e6 + 1e-8) / 200)

        # 5. Deployment consistency (1 - std/mean of deploy)
        if dep is not None and len(dep) > 1:
            dep_std = dep["ep_return"].std()
            dep_mean = dep["ep_return"].mean()
            consist = max(0, 1.0 - dep_std / (dep_mean + 1e-8))
        else:
            consist = 0.0

        metrics[algo] = [perf, conv, stab, samp_eff, consist]

    categories = ["Final\nPerformance", "Convergence\nSpeed", "Training\nStability",
                  "Sample\nEfficiency", "Deployment\nConsistency"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for algo in ALGOS:
        if algo not in metrics:
            continue
        values = metrics[algo] + metrics[algo][:1]
        ax.plot(angles, values, 'o-', color=ALGO_COLORS[algo],
                label=ALGO_DISPLAY[algo], linewidth=2, markersize=5)
        ax.fill(angles, values, alpha=0.1, color=ALGO_COLORS[algo])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Multi-Dimensional Algorithm Comparison", size=14, y=1.08)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig11_radar_chart.png"), bbox_inches="tight")
    plt.close(fig)
    print("  fig11_radar_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 12: Loss Curves
# ─────────────────────────────────────────────────────────────────────────────
def make_fig12(out):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Policy/Surrogate loss (PPO, AC)
    ax = axes[0]
    for algo, col_name in [("PPO", "surrogate_loss"), ("AC", "actor_loss")]:
        m = load_metrics(algo)
        if m is None or col_name not in m.columns:
            continue
        y = smooth(m[col_name].values, window=max(1, len(m)//100))
        ax.plot(m["global_step"].values, y, color=ALGO_COLORS[algo],
                label=ALGO_DISPLAY[algo])
    ax.set_xlabel("Global Steps")
    ax.set_ylabel("Policy / Surrogate Loss")
    ax.set_title("Policy Loss")
    ax.legend()

    # Panel 2: Value/Critic loss (PPO, AC)
    ax = axes[1]
    for algo, col_name in [("PPO", "value_loss"), ("AC", "critic_loss")]:
        m = load_metrics(algo)
        if m is None or col_name not in m.columns:
            continue
        y = smooth(m[col_name].values, window=max(1, len(m)//100))
        ax.plot(m["global_step"].values, y, color=ALGO_COLORS[algo],
                label=ALGO_DISPLAY[algo])
    ax.set_xlabel("Global Steps")
    ax.set_ylabel("Value / Critic Loss")
    ax.set_title("Value Loss")
    ax.legend()

    # Panel 3: DQN TD loss
    ax = axes[2]
    m = load_metrics("DQN")
    if m is not None and "td_loss" in m.columns:
        y = smooth(m["td_loss"].values, window=max(1, len(m)//100))
        ax.plot(m["global_step"].values, y, color=ALGO_COLORS["DQN"],
                label="DQN")
    ax.set_xlabel("Global Steps")
    ax.set_ylabel("TD (Huber) Loss")
    ax.set_title("DQN TD Loss")
    ax.legend()

    fig.suptitle("Training Loss Curves", size=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig12_loss_curves.png"), bbox_inches="tight")
    plt.close(fig)
    print("  fig12_loss_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 13: Entropy Over Training
# ─────────────────────────────────────────────────────────────────────────────
def make_fig13(out):
    fig, ax = plt.subplots(figsize=(12, 5))
    for algo in ["PPO", "AC", "MC_REINFORCE"]:
        m = load_metrics(algo)
        if m is None or "entropy" not in m.columns:
            continue
        y = smooth(m["entropy"].values, window=max(1, len(m)//100))
        ax.plot(m["global_step"].values, y, color=ALGO_COLORS[algo],
                label=ALGO_DISPLAY[algo], linewidth=2)
    ax.set_xlabel("Global Steps")
    ax.set_ylabel("Policy Entropy")
    ax.set_title("Entropy Over Training (Exploration → Exploitation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig13_entropy.png"), bbox_inches="tight")
    plt.close(fig)
    print("  fig13_entropy.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 14: Gradient Norm Over Training
# ─────────────────────────────────────────────────────────────────────────────
def make_fig14(out):
    fig, ax = plt.subplots(figsize=(12, 5))
    for algo in ["PPO", "AC", "DQN"]:
        m = load_metrics(algo)
        if m is None or "grad_norm" not in m.columns:
            continue
        y = smooth(m["grad_norm"].values, window=max(1, len(m)//100))
        ax.plot(m["global_step"].values, y, color=ALGO_COLORS[algo],
                label=ALGO_DISPLAY[algo], linewidth=2)
    ax.set_xlabel("Global Steps")
    ax.set_ylabel("Gradient L2 Norm (pre-clip)")
    ax.set_title("Gradient Norm Over Training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig14_grad_norm.png"), bbox_inches="tight")
    plt.close(fig)
    print("  fig14_grad_norm.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 15: Explained Variance
# ─────────────────────────────────────────────────────────────────────────────
def make_fig15(out):
    fig, ax = plt.subplots(figsize=(12, 5))
    for algo in ["PPO", "AC"]:
        m = load_metrics(algo)
        if m is None or "explained_variance" not in m.columns:
            continue
        y = smooth(m["explained_variance"].values, window=max(1, len(m)//100))
        ax.plot(m["global_step"].values, y, color=ALGO_COLORS[algo],
                label=ALGO_DISPLAY[algo], linewidth=2)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect critic")
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5, label="Useless critic")
    ax.set_xlabel("Global Steps")
    ax.set_ylabel("Explained Variance")
    ax.set_title("Explained Variance: How Well Does the Critic Predict Returns?")
    ax.set_ylim(-0.5, 1.1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig15_explained_variance.png"), bbox_inches="tight")
    plt.close(fig)
    print("  fig15_explained_variance.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 16: PPO Clip Fraction
# ─────────────────────────────────────────────────────────────────────────────
def make_fig16(out):
    m = load_metrics("PPO")
    if m is None or "clip_fraction" not in m.columns:
        print("  fig16 SKIPPED (no PPO metrics)")
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    y = smooth(m["clip_fraction"].values, window=max(1, len(m)//50))
    ax.plot(m["global_step"].values, y, color=ALGO_COLORS["PPO"], linewidth=2)
    ax.axhspan(0.1, 0.3, alpha=0.1, color="green", label="Ideal range (0.1–0.3)")
    ax.set_xlabel("Global Steps")
    ax.set_ylabel("Clip Fraction")
    ax.set_title("PPO Clip Fraction (Fraction of Samples Clipped per Update)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig16_ppo_clip_fraction.png"), bbox_inches="tight")
    plt.close(fig)
    print("  fig16_ppo_clip_fraction.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 17: DQN Q-Values
# ─────────────────────────────────────────────────────────────────────────────
def make_fig17(out):
    m = load_metrics("DQN")
    if m is None or "q_mean" not in m.columns:
        print("  fig17 SKIPPED (no DQN metrics)")
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    y = smooth(m["q_mean"].values, window=max(1, len(m)//100))
    ax.plot(m["global_step"].values, y, color=ALGO_COLORS["DQN"], linewidth=2)
    ax.set_xlabel("Global Steps")
    ax.set_ylabel("Mean Q-value (max over actions)")
    ax.set_title("DQN: Average Predicted Q-Values Over Training")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig17_dqn_q_values.png"), bbox_inches="tight")
    plt.close(fig)
    print("  fig17_dqn_q_values.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 18: Wall-Clock Time Comparison
# ─────────────────────────────────────────────────────────────────────────────
def make_fig18(out):
    # Parse timing log
    timing_path = os.path.join(ROOT, "experiments", "timing_log.txt")
    if not os.path.isfile(timing_path):
        print("  fig18 SKIPPED (no timing log)")
        return

    times = {}
    with open(timing_path) as f:
        for line in f:
            if "[TRAIN]" in line and "elapsed" in line:
                parts = line.strip()
                for algo in ALGOS:
                    if algo in parts:
                        # Extract elapsed time
                        idx = parts.index("elapsed")
                        time_str = parts[idx:].split("|")[0].replace("elapsed", "").strip()
                        val = float(time_str.split()[0])
                        unit = time_str.split()[1] if len(time_str.split()) > 1 else "min"
                        if "sec" in unit:
                            val /= 60.0
                        times[algo] = val
                        break

    if not times:
        print("  fig18 SKIPPED (no timing data)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    algos_sorted = sorted(times.keys(), key=lambda a: times[a], reverse=True)
    colors = [ALGO_COLORS[a] for a in algos_sorted]
    bars = ax.barh([ALGO_DISPLAY[a] for a in algos_sorted],
                   [times[a] for a in algos_sorted], color=colors)
    for bar, algo in zip(bars, algos_sorted):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{times[algo]:.1f} min", va="center", fontsize=11)
    ax.set_xlabel("Training Time (minutes)")
    ax.set_title("Wall-Clock Training Time (20,000 episodes)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig18_wall_clock.png"), bbox_inches="tight")
    plt.close(fig)
    print("  fig18_wall_clock.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(ROOT, "figures"))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    print(f"Generating additional figures to {args.output}")

    make_fig11(args.output)
    make_fig12(args.output)
    make_fig13(args.output)
    make_fig14(args.output)
    make_fig15(args.output)
    make_fig16(args.output)
    make_fig17(args.output)
    make_fig18(args.output)

    print("Done!")


if __name__ == "__main__":
    main()
