import os
import json
import csv
import numpy as np
from typing import Dict, Any

from bandit import MultiArmedBandit
from agent import Agent
from utils import (
    plot_q_comparison,
    plot_reward_distribution,
    plot_combined_rewards,
    plot_combined_optimal_actions,
    plot_action_counts,
    plot_cumulative_regret,
    plot_grouped_comparison,
    plot_best_overlay,
    plot_q_error,
    plot_combined_q_error,
)

N_ARMS = 10
N_RUNS = 2000
N_STEPS = 1000


SEED = 42


def time_decay(t, r):
    return 1.0 / (1 + 0.01 * t)


EXPERIMENTS = [
    {"strategy": "epsilon_greedy", "epsilon": 0.0, "seed": SEED},
    {"strategy": "epsilon_greedy", "epsilon": 0.01, "seed": SEED},
    {"strategy": "epsilon_greedy", "epsilon": 0.1, "seed": SEED},
    {"strategy": "epsilon_greedy", "epsilon": 0.5, "seed": SEED},
    {"strategy": "epsilon_greedy", "epsilon": time_decay, "seed": SEED},
    
    {"strategy": "ucb", "c": 0.1, "seed": SEED},
    {"strategy": "ucb", "c": 0.5, "seed": SEED},
    {"strategy": "ucb", "c": 1.0, "seed": SEED},
    {"strategy": "ucb", "c": 2.0, "seed": SEED},
    {"strategy": "ucb", "c": 5.0, "seed": SEED},
    
    {"strategy": "epsilon_greedy", "epsilon": 0.0, "initial_q": 5.0, "alpha": 0.1, "seed": SEED},
    {"strategy": "epsilon_greedy", "epsilon": 0.0, "initial_q": 5.0, "alpha": None, "seed": SEED},

    {"strategy": "ucb", "c": 1.0, "initial_q": 5.0, "alpha": 0.1, "seed": SEED},
    {"strategy": "ucb", "c": 1.0, "initial_q": 5.0, "alpha": None, "seed": SEED},
]



def run_single_experiment(bandit: MultiArmedBandit, agent: Agent, n_steps: int):
    rewards = np.zeros(n_steps)
    actions = np.zeros(n_steps, dtype=int)
    optimal_actions = np.zeros(n_steps, dtype=bool)
    q_errors = np.zeros(n_steps)
    optimal_arm = bandit.get_optimal_arm()
    true_means = bandit.true_means

    for step in range(n_steps):
        action = agent.select_action()
        reward = bandit.pull(action)
        agent.update(action, reward)

        rewards[step] = reward
        actions[step] = action
        optimal_actions[step] = action == optimal_arm
        # Mean absolute error between Q estimates and true means
        q_errors[step] = np.mean(np.abs(agent.q_estimates - true_means))

    return rewards, actions, optimal_actions, q_errors


def run_experiment(exp_config: Dict[str, Any]):
    all_rewards = np.zeros((N_RUNS, N_STEPS))
    all_actions = np.zeros((N_RUNS, N_STEPS), dtype=int)
    all_optimal = np.zeros((N_RUNS, N_STEPS))
    all_regret = np.zeros((N_RUNS, N_STEPS))
    all_q_errors = np.zeros((N_RUNS, N_STEPS))

    last_bandit = None
    last_agent = None
    last_actions = None

    base_seed = exp_config.get("seed", None)

    for run in range(N_RUNS):
        if base_seed is not None:
            np.random.seed(base_seed + run)

        bandit = MultiArmedBandit(n_arms=N_ARMS)
        agent = Agent(
            n_arms=N_ARMS,
            epsilon=exp_config.get("epsilon", 0.1),
            c=exp_config.get("c", 2.0),
            strategy=exp_config.get("strategy", "epsilon_greedy"),
            initial_q=exp_config.get("initial_q", 0.0),
            alpha=exp_config.get("alpha", None),
        )

        rewards, actions, optimal, q_errors = run_single_experiment(bandit, agent, N_STEPS)
        all_rewards[run] = rewards
        all_actions[run] = actions
        all_optimal[run] = optimal
        all_q_errors[run] = q_errors

        # Calculate cumulative regret for this run
        optimal_mean = bandit.true_means[bandit.get_optimal_arm()]
        instant_regret = optimal_mean - np.array([bandit.true_means[a] for a in actions])
        all_regret[run] = np.cumsum(instant_regret)

        last_bandit = bandit
        last_agent = agent
        last_actions = actions

    # Compute last run action counts
    last_action_counts = np.zeros(N_ARMS)
    for arm in range(N_ARMS):
        last_action_counts[arm] = np.sum(last_actions == arm)

    # Compute optimal vs suboptimal totals (averaged across runs)
    total_optimal_selections = np.sum(all_optimal)  # Total across all runs and steps
    total_suboptimal_selections = N_RUNS * N_STEPS - total_optimal_selections

    return {
        "avg_rewards": np.mean(all_rewards, axis=0),
        "avg_optimal": np.mean(all_optimal, axis=0) * 100,
        "avg_regret": np.mean(all_regret, axis=0),
        "avg_q_error": np.mean(all_q_errors, axis=0),
        "all_rewards": all_rewards,
        "all_actions": all_actions,
        "all_optimal": all_optimal,
        "last_bandit": last_bandit,
        "last_agent": last_agent,
        "last_action_counts": last_action_counts,
        "total_optimal_selections": int(total_optimal_selections),
        "total_suboptimal_selections": int(total_suboptimal_selections),
    }


def save_log(exp_name: str, exp_config: Dict, results: Dict, output_dir: str, label: str):
    serializable_config = {}
    for key, value in exp_config.items():
        if callable(value):
            func_name = getattr(value, "__name__", "custom_function")
            serializable_config[key] = f"function:{func_name}"
        else:
            serializable_config[key] = value

    log_data = {
        "experiment": exp_name,
        "label": label,
        "config": serializable_config,
        "parameters": {
            "n_arms": N_ARMS,
            "n_runs": N_RUNS,
            "n_steps": N_STEPS,
        },
        "results": {
            "final_avg_reward": float(results["avg_rewards"][-1]),
            "final_optimal_pct": float(results["avg_optimal"][-1]),
            "final_cumulative_regret": float(results["avg_regret"][-1]),
            "final_q_error": float(results["avg_q_error"][-1]),
            "total_optimal_selections": results["total_optimal_selections"],
            "total_suboptimal_selections": results["total_suboptimal_selections"],
        },
        "last_run": {
            "true_means": results["last_bandit"].true_means.tolist(),
            "q_estimates": results["last_agent"].q_estimates.tolist(),
            "optimal_arm": int(results["last_bandit"].get_optimal_arm()),
            "action_counts": results["last_action_counts"].tolist(),
        },
    }

    log_path = os.path.join(output_dir, f"{exp_name}_log.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    csv_path = os.path.join(output_dir, f"{exp_name}_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "avg_reward", "optimal_pct", "cumulative_regret", "q_error"])
        for step in range(N_STEPS):
            writer.writerow([
                step + 1,
                results["avg_rewards"][step],
                results["avg_optimal"][step],
                results["avg_regret"][step],
                results["avg_q_error"][step],
            ])


def generate_label(exp_config: Dict) -> str:
    strategy = exp_config.get("strategy", "epsilon_greedy")
    parts = []

    if strategy == "epsilon_greedy":
        epsilon = exp_config.get("epsilon", 0.1)
        if callable(epsilon):
            func_name = getattr(epsilon, "__name__", "f")
            parts.append(f"ε={func_name}(t,r)")
        else:
            parts.append(f"ε={epsilon}")
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
        epsilon = exp_config.get("epsilon", 0.1)
        if callable(epsilon):
            func_name = getattr(epsilon, "__name__", "decay")
            parts.append(f"eps_{func_name}")
        else:
            parts.append(f"eps{epsilon}")
    elif strategy == "ucb":
        parts.append(f"c{exp_config.get('c', 2.0)}")

    if exp_config.get("initial_q", 0.0) != 0.0:
        parts.append(f"q{exp_config.get('initial_q')}")

    if exp_config.get("alpha") is not None:
        parts.append(f"alpha{exp_config.get('alpha')}")

    return "_".join(parts)


def run_and_plot_experiment(exp_config: Dict, figures_dir: str, logs_dir: str):
    label = generate_label(exp_config)
    folder_name = generate_folder_name(exp_config)
    print(f"Running {folder_name} ({label})...")
    results = run_experiment(exp_config)
    results["label"] = label
    results["folder_name"] = folder_name

    fig_dir = os.path.join(figures_dir, folder_name)
    log_dir = os.path.join(logs_dir, folder_name)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    plot_q_comparison(
        results["last_bandit"].true_means,
        results["last_agent"].q_estimates,
        title=f"Q-Estimates vs True Means - {label}",
        save_path=os.path.join(fig_dir, "q_comparison.png"),
    )

    plot_reward_distribution(
        results["last_bandit"],
        title=f"Reward Distribution - {label}",
        save_path=os.path.join(fig_dir, "reward_distribution.png"),
    )

    plot_action_counts(
        results["last_action_counts"],
        results["last_bandit"].get_optimal_arm(),
        title=f"Action Counts (Last Run) - {label}",
        save_path=os.path.join(fig_dir, "action_counts.png"),
    )

    plot_q_error(
        results["avg_q_error"],
        title=f"Q Estimation Error - {label}",
        save_path=os.path.join(fig_dir, "q_error.png"),
    )

    save_log(folder_name, exp_config, results, log_dir, label)

    print(f"  Final avg reward: {results['avg_rewards'][-1]:.4f}")
    print(f"  Final optimal %: {results['avg_optimal'][-1]:.2f}%")
    print(f"  Final regret: {results['avg_regret'][-1]:.1f}")
    print(f"  Final Q error: {results['avg_q_error'][-1]:.4f}")

    return results


def main():
    figures_dir = "figures"
    logs_dir = "logs"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("=" * 60)
    print("Multi-Armed Bandit Experiments")
    print("=" * 60)
    print(f"Arms: {N_ARMS}, Runs: {N_RUNS}, Steps: {N_STEPS}")
    print()

    all_results = []
    for exp_config in EXPERIMENTS:
        results = run_and_plot_experiment(exp_config, figures_dir, logs_dir)
        all_results.append(results)

    print()
    print("Generating combined plots...")

    reward_data = {res["label"]: res["avg_rewards"] for res in all_results}
    optimal_data = {res["label"]: res["avg_optimal"] for res in all_results}
    regret_data = {res["label"]: res["avg_regret"] for res in all_results}
    q_error_data = {res["label"]: res["avg_q_error"] for res in all_results}

    plot_combined_rewards(
        reward_data,
        save_path=os.path.join(figures_dir, "combined_rewards.png"),
    )

    plot_combined_optimal_actions(
        optimal_data,
        save_path=os.path.join(figures_dir, "combined_optimal.png"),
    )

    plot_cumulative_regret(
        regret_data,
        save_path=os.path.join(figures_dir, "combined_regret.png"),
    )

    plot_combined_q_error(
        q_error_data,
        save_path=os.path.join(figures_dir, "combined_q_error.png"),
    )

    # Group experiments
    print("Generating grouped comparison plots...")

    # Epsilon-Greedy group (indices 0-4)
    eps_results = all_results[0:5]
    eps_reward = {res["label"]: res["avg_rewards"] for res in eps_results}
    eps_optimal = {res["label"]: res["avg_optimal"] for res in eps_results}
    plot_grouped_comparison(
        eps_reward, eps_optimal, "Epsilon-Greedy",
        save_path=os.path.join(figures_dir, "group_epsilon_greedy.png"),
    )

    # UCB group (indices 5-9)
    ucb_results = all_results[5:10]
    ucb_reward = {res["label"]: res["avg_rewards"] for res in ucb_results}
    ucb_optimal = {res["label"]: res["avg_optimal"] for res in ucb_results}
    plot_grouped_comparison(
        ucb_reward, ucb_optimal, "UCB",
        save_path=os.path.join(figures_dir, "group_ucb.png"),
    )

    # Optimistic Greedy group (indices 10-11)
    opt_greedy_results = all_results[10:12]
    opt_greedy_reward = {res["label"]: res["avg_rewards"] for res in opt_greedy_results}
    opt_greedy_optimal = {res["label"]: res["avg_optimal"] for res in opt_greedy_results}
    plot_grouped_comparison(
        opt_greedy_reward, opt_greedy_optimal, "Optimistic Greedy",
        save_path=os.path.join(figures_dir, "group_optimistic_greedy.png"),
    )

    # Optimistic UCB group (indices 12-13)
    opt_ucb_results = all_results[12:14]
    opt_ucb_reward = {res["label"]: res["avg_rewards"] for res in opt_ucb_results}
    opt_ucb_optimal = {res["label"]: res["avg_optimal"] for res in opt_ucb_results}
    plot_grouped_comparison(
        opt_ucb_reward, opt_ucb_optimal, "Optimistic UCB",
        save_path=os.path.join(figures_dir, "group_optimistic_ucb.png"),
    )

    # Find best from main groups (by final reward)
    # Note: Optimistic UCB excluded from overlay - shown in its own group plot
    print("Generating best overlay plot...")

    best_eps = max(eps_results, key=lambda x: x["avg_rewards"][-1])
    best_ucb = max(ucb_results, key=lambda x: x["avg_rewards"][-1])
    best_opt = max(opt_greedy_results, key=lambda x: x["avg_rewards"][-1])

    best_reward = {
        "ε-Greedy": best_eps["avg_rewards"],
        "UCB": best_ucb["avg_rewards"],
        "Optimistic": best_opt["avg_rewards"],
    }
    best_optimal = {
        "ε-Greedy": best_eps["avg_optimal"],
        "UCB": best_ucb["avg_optimal"],
        "Optimistic": best_opt["avg_optimal"],
    }
    best_regret = {
        "ε-Greedy": best_eps["avg_regret"],
        "UCB": best_ucb["avg_regret"],
        "Optimistic": best_opt["avg_regret"],
    }

    plot_best_overlay(
        best_reward, best_optimal, best_regret,
        save_path=os.path.join(figures_dir, "best_overlay.png"),
    )

    print()
    print("=" * 60)
    print("Done! Results saved to figures/ and logs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
