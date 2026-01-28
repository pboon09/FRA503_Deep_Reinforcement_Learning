import os
import json
import numpy as np
from typing import Dict, Any

from bandit import MultiArmedBandit
from agent import Agent
from utils import (
    plot_q_comparison,
    plot_reward_distribution,
    plot_combined_rewards,
    plot_combined_optimal_actions,
)

N_ARMS = 10
N_RUNS = 2000
N_STEPS = 1000

EXPERIMENTS = [
    {
        "strategy": "epsilon_greedy",
        "epsilon": 0.0,
    },
    {
        "strategy": "epsilon_greedy",
        "epsilon": 0.01,
    },
    {
        "strategy": "epsilon_greedy",
        "epsilon": 0.1,
    },
    {
        "strategy": "ucb",
        "c": 2.0,
    },
    {
        "strategy": "epsilon_greedy",
        "epsilon": 0.0,
        "initial_q": 5.0,
        "alpha": 0.1,
    },
]


def run_single_experiment(bandit: MultiArmedBandit, agent: Agent, n_steps: int):
    rewards = np.zeros(n_steps)
    actions = np.zeros(n_steps, dtype=int)
    optimal_actions = np.zeros(n_steps, dtype=bool)
    optimal_arm = bandit.get_optimal_arm()

    for step in range(n_steps):
        action = agent.select_action()
        reward = bandit.pull(action)
        agent.update(action, reward)

        rewards[step] = reward
        actions[step] = action
        optimal_actions[step] = action == optimal_arm

    return rewards, actions, optimal_actions


def run_experiment(exp_config: Dict[str, Any]):
    all_rewards = np.zeros((N_RUNS, N_STEPS))
    all_actions = np.zeros((N_RUNS, N_STEPS), dtype=int)
    all_optimal = np.zeros((N_RUNS, N_STEPS))

    last_bandit = None
    last_agent = None

    for run in range(N_RUNS):
        bandit = MultiArmedBandit(n_arms=N_ARMS)
        agent = Agent(
            n_arms=N_ARMS,
            epsilon=exp_config.get("epsilon", 0.1),
            c=exp_config.get("c", 2.0),
            strategy=exp_config.get("strategy", "epsilon_greedy"),
            initial_q=exp_config.get("initial_q", 0.0),
            alpha=exp_config.get("alpha", None),
        )

        rewards, actions, optimal = run_single_experiment(bandit, agent, N_STEPS)
        all_rewards[run] = rewards
        all_actions[run] = actions
        all_optimal[run] = optimal

        last_bandit = bandit
        last_agent = agent

    return {
        "avg_rewards": np.mean(all_rewards, axis=0),
        "avg_optimal": np.mean(all_optimal, axis=0) * 100,
        "all_rewards": all_rewards,
        "all_actions": all_actions,
        "all_optimal": all_optimal,
        "last_bandit": last_bandit,
        "last_agent": last_agent,
    }


def save_log(exp_name: str, exp_config: Dict, results: Dict, output_dir: str, label: str):
    log_data = {
        "experiment": exp_name,
        "label": label,
        "config": exp_config,
        "parameters": {
            "n_arms": N_ARMS,
            "n_runs": N_RUNS,
            "n_steps": N_STEPS,
        },
        "results": {
            "final_avg_reward": float(results["avg_rewards"][-1]),
            "final_optimal_pct": float(results["avg_optimal"][-1]),
        },
        "last_run": {
            "true_means": results["last_bandit"].true_means.tolist(),
            "q_estimates": results["last_agent"].q_estimates.tolist(),
            "optimal_arm": int(results["last_bandit"].get_optimal_arm()),
        },
    }

    log_path = os.path.join(output_dir, f"{exp_name}_log.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)


def generate_label(exp_config: Dict) -> str:
    strategy = exp_config.get("strategy", "epsilon_greedy")
    parts = []

    if strategy == "epsilon_greedy":
        parts.append(f"ε={exp_config.get('epsilon', 0.1)}")
    elif strategy == "ucb":
        parts.append(f"c={exp_config.get('c', 2.0)}")

    if exp_config.get("initial_q", 0.0) != 0.0:
        parts.append(f"Q₀={exp_config.get('initial_q')}")

    if exp_config.get("alpha") is not None:
        parts.append(f"α={exp_config.get('alpha')}")

    return f"{strategy}: {', '.join(parts)}"


def generate_folder_name(exp_config: Dict) -> str:
    strategy = exp_config.get("strategy", "epsilon_greedy")
    parts = [strategy]

    if strategy == "epsilon_greedy":
        parts.append(f"eps{exp_config.get('epsilon', 0.1)}")
    elif strategy == "ucb":
        parts.append(f"c{exp_config.get('c', 2.0)}")

    if exp_config.get("initial_q", 0.0) != 0.0:
        parts.append(f"q{exp_config.get('initial_q')}")

    if exp_config.get("alpha") is not None:
        parts.append(f"alpha{exp_config.get('alpha')}")

    return "_".join(parts)


def run_and_plot_experiment(exp_config: Dict, output_dir: str):
    label = generate_label(exp_config)
    folder_name = generate_folder_name(exp_config)
    print(f"Running {folder_name} ({label})...")
    results = run_experiment(exp_config)
    results["label"] = label
    results["folder_name"] = folder_name

    exp_dir = os.path.join(output_dir, folder_name)
    os.makedirs(exp_dir, exist_ok=True)

    plot_q_comparison(
        results["last_bandit"].true_means,
        results["last_agent"].q_estimates,
        title=f"Q-Estimates vs True Means - {label}",
        save_path=os.path.join(exp_dir, "q_comparison.png"),
    )

    plot_reward_distribution(
        results["last_bandit"],
        title=f"Reward Distribution - {label}",
        save_path=os.path.join(exp_dir, "reward_distribution.png"),
    )

    save_log(folder_name, exp_config, results, exp_dir, label)

    print(f"  Final avg reward: {results['avg_rewards'][-1]:.4f}")
    print(f"  Final optimal %: {results['avg_optimal'][-1]:.2f}%")

    return results


def main():
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Multi-Armed Bandit Experiments")
    print("=" * 60)
    print(f"Arms: {N_ARMS}, Runs: {N_RUNS}, Steps: {N_STEPS}")
    print()

    all_results = []
    for exp_config in EXPERIMENTS:
        results = run_and_plot_experiment(exp_config, output_dir)
        all_results.append(results)

    print()
    print("Generating combined plots...")

    reward_data = {res["label"]: res["avg_rewards"] for res in all_results}
    optimal_data = {res["label"]: res["avg_optimal"] for res in all_results}

    plot_combined_rewards(
        reward_data,
        save_path=os.path.join(output_dir, "combined_rewards.png"),
    )

    plot_combined_optimal_actions(
        optimal_data,
        save_path=os.path.join(output_dir, "combined_optimal.png"),
    )

    print()
    print("=" * 60)
    print("Done! Results saved to figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
