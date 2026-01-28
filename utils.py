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


def plot_action_counts(
    action_counts: np.ndarray,
    optimal_arm: int,
    title: str = "Action Counts per Arm",
    save_path: Optional[str] = None,
):
    n_arms = len(action_counts)
    x = np.arange(1, n_arms + 1)

    colors = ['tab:blue'] * n_arms
    colors[optimal_arm] = 'tab:green'

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, action_counts, color=colors, alpha=0.8)

    ax.set_xlabel("Arm")
    ax.set_ylabel("Times Selected")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, axis="y")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='tab:green', label='Optimal Arm'),
                       Patch(facecolor='tab:blue', label='Other Arms')]
    ax.legend(handles=legend_elements, loc='upper right')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cumulative_regret(
    regret_data: Dict[str, np.ndarray],
    title: str = "Cumulative Regret Over Time",
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, regret in regret_data.items():
        ax.plot(regret, label=name)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_q_error(
    q_error: np.ndarray,
    title: str = "Q Estimation Error Over Time",
    save_path: Optional[str] = None,
):
    """Plot mean absolute error between Q estimates and true means over time."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(q_error, color="tab:purple", linewidth=1.5)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_combined_q_error(
    q_error_data: Dict[str, np.ndarray],
    title: str = "Q Estimation Error Comparison",
    save_path: Optional[str] = None,
):
    """Plot Q estimation error comparison across algorithms."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, q_error in q_error_data.items():
        ax.plot(q_error, label=name)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_grouped_comparison(
    reward_data: Dict[str, np.ndarray],
    optimal_data: Dict[str, np.ndarray],
    group_name: str,
    save_path: Optional[str] = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, rewards in reward_data.items():
        axes[0].plot(rewards, label=name)
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title(f"{group_name}: Average Reward", fontweight="bold")
    axes[0].legend(loc="lower right", framealpha=0.9, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for name, optimal in optimal_data.items():
        axes[1].plot(optimal, label=name)
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("% Optimal Action")
    axes[1].set_title(f"{group_name}: Optimal Action %", fontweight="bold")
    axes[1].legend(loc="lower right", framealpha=0.9, fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_best_overlay(
    reward_data: Dict[str, np.ndarray],
    optimal_data: Dict[str, np.ndarray],
    regret_data: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for i, (name, rewards) in enumerate(reward_data.items()):
        axes[0].plot(rewards, label=name, color=colors[i % len(colors)], linewidth=2)
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title("Best of Each Group: Reward", fontweight="bold")
    axes[0].legend(loc="lower right", framealpha=0.9)
    axes[0].grid(True, alpha=0.3)

    for i, (name, optimal) in enumerate(optimal_data.items()):
        axes[1].plot(optimal, label=name, color=colors[i % len(colors)], linewidth=2)
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("% Optimal Action")
    axes[1].set_title("Best of Each Group: Optimal %", fontweight="bold")
    axes[1].legend(loc="lower right", framealpha=0.9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 100)

    for i, (name, regret) in enumerate(regret_data.items()):
        axes[2].plot(regret, label=name, color=colors[i % len(colors)], linewidth=2)
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("Cumulative Regret")
    axes[2].set_title("Best of Each Group: Regret", fontweight="bold")
    axes[2].legend(loc="upper left", framealpha=0.9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
