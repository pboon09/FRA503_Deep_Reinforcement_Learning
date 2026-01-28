import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional


def plot_q_comparison(
    true_means: np.ndarray,
    q_estimates: np.ndarray,
    title: str = "Q-Estimates vs True Means",
    save_path: Optional[str] = None,
):
    n_arms = len(true_means)
    x = np.arange(1, n_arms + 1)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width / 2, true_means, width, label="True Mean (q*)", color="tab:red", alpha=0.8)
    ax.bar(x + width / 2, q_estimates, width, label="Estimated Q", color="tab:blue", alpha=0.8)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel("Arm")
    ax.set_ylabel("Value")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")

    y_min = min(min(true_means), min(q_estimates))
    y_max = max(max(true_means), max(q_estimates))
    y_abs = max(abs(y_min), abs(y_max)) * 1.2
    ax.set_ylim(-y_abs, y_abs)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_reward_distribution(
    bandit,
    n_samples: int = 1000,
    title: str = "Reward Distribution per Arm",
    save_path: Optional[str] = None,
):
    rewards = []
    for arm in range(bandit.n_arms):
        arm_rewards = [bandit.pull(arm) for _ in range(n_samples)]
        rewards.append(arm_rewards)

    fig, ax = plt.subplots(figsize=(12, 6))

    positions = range(1, bandit.n_arms + 1)
    ax.violinplot(rewards, positions=positions, showmeans=True)

    for i, mean in enumerate(bandit.true_means):
        if i == 0:
            ax.plot(i + 1, mean, "ro", markersize=10, label="True Mean (q*)")
        else:
            ax.plot(i + 1, mean, "ro", markersize=10)

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1.5, alpha=0.8)

    all_rewards_flat = [r for arm_rewards in rewards for r in arm_rewards]
    y_max = max(all_rewards_flat)
    y_min = min(all_rewards_flat)
    y_abs = max(abs(y_min), abs(y_max)) * 1.1
    ax.set_ylim(-y_abs, y_abs)

    ax.set_xlabel("Arm")
    ax.set_ylabel("Reward")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(positions)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_combined_rewards(
    reward_data: Dict[str, np.ndarray],
    title: str = "Average Reward Comparison",
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, rewards in reward_data.items():
        ax.plot(rewards, label=name)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Average Reward")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_combined_optimal_actions(
    optimal_data: Dict[str, np.ndarray],
    title: str = "Optimal Action Percentage Comparison",
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, optimal_pct in optimal_data.items():
        ax.plot(optimal_pct, label=name)

    ax.set_xlabel("Steps")
    ax.set_ylabel("% Optimal Action")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
