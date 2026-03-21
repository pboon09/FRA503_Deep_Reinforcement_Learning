#!/usr/bin/env python3
import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ALGOS = ["Linear_Q", "DQN", "MC_REINFORCE", "AC", "PPO"]
ALGO_COLORS = {
    "Linear_Q": "#1f77b4",
    "DQN": "#ff7f0e",
    "MC_REINFORCE": "#2ca02c",
    "AC": "#d62728",
    "PPO": "#9467bd",
}

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_latest_csv(algo, task="Stabilize"):
    # Check experiments/suite_1_baseline/ first (HW2-style)
    exp_path = os.path.join(ROOT, "experiments", "suite_1_baseline", f"{algo}.csv")
    if os.path.isfile(exp_path):
        return exp_path
    # Fallback: logs/{task}/{algo}/
    csv_dir = os.path.join(ROOT, "logs", task, algo)
    matches = sorted(glob.glob(os.path.join(csv_dir, "training_*.csv")), key=os.path.getmtime)
    return matches[-1] if matches else None


def plot_learning_curves(output_dir, task="Stabilize", window=100):
    fig, ax = plt.subplots(figsize=(12, 6))
    for algo in ALGOS:
        csv_path = find_latest_csv(algo, task)
        if csv_path is None:
            print(f"  No CSV for {algo}, skipping.")
            continue
        df = pd.read_csv(csv_path)
        if "avg_episode_duration" in df.columns:
            y = df["avg_episode_duration"]
        elif "ep_length" in df.columns:
            y = df["ep_length"]
        elif "ep_return" in df.columns:
            y = df["ep_return"]
        else:
            continue
        smoothed = y.rolling(window=window, min_periods=1).mean()
        ax.plot(smoothed, label=algo, color=ALGO_COLORS.get(algo), alpha=0.9)
        ax.fill_between(range(len(smoothed)),
                        y.rolling(window=window, min_periods=1).quantile(0.25),
                        y.rolling(window=window, min_periods=1).quantile(0.75),
                        alpha=0.15, color=ALGO_COLORS.get(algo))
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Duration / Return")
    ax.set_title("Learning Efficiency: All Algorithms")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_learning_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig1_learning_curves.png")


def plot_deployment_bar(output_dir, deploy_csv):
    if not os.path.isfile(deploy_csv):
        print(f"  No deployment CSV at {deploy_csv}, skipping.")
        return
    df = pd.read_csv(deploy_csv)
    fig, ax = plt.subplots(figsize=(8, 5))
    means = df.groupby("algorithm")["ep_return"].mean()
    stds = df.groupby("algorithm")["ep_return"].std()
    algos = [a for a in ALGOS if a in means.index]
    colors = [ALGO_COLORS[a] for a in algos]
    ax.bar(algos, [means[a] for a in algos], yerr=[stds[a] for a in algos],
           color=colors, capsize=5, alpha=0.85)
    ax.set_ylabel("Mean Episode Return")
    ax.set_title("Deployment Performance")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_deployment_bar.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig2_deployment_bar.png")


def plot_training_loss(output_dir, task="Stabilize", window=100):
    fig, ax = plt.subplots(figsize=(12, 6))
    for algo in ALGOS:
        csv_path = find_latest_csv(algo, task)
        if csv_path is None:
            continue
        df = pd.read_csv(csv_path)
        if "avg_episode_duration" in df.columns:
            loss = df["avg_episode_duration"]
        elif "loss" in df.columns:
            loss = df["loss"].replace(0, np.nan).dropna()
        else:
            continue
        if len(loss) < 2:
            continue
        smoothed = loss.rolling(window=window, min_periods=1).mean()
        ax.plot(smoothed.values, label=algo, color=ALGO_COLORS.get(algo), alpha=0.9)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_training_loss.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig3_training_loss.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(ROOT, "figures"))
    parser.add_argument("--task", default="Stabilize")
    parser.add_argument("--deploy_csv", default=None)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Generating figures...")
    plot_learning_curves(args.output, args.task)
    plot_training_loss(args.output, args.task)

    if args.deploy_csv:
        plot_deployment_bar(args.output, args.deploy_csv)

    print("Done.")


if __name__ == "__main__":
    main()
