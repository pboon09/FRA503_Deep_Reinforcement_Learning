#!/usr/bin/env python3

import json
import os
import numpy as np
from utils import plot_q_comparison_boxplot

experiments = [
    ("logs/epsilon_greedy_eps0.0", "figures/epsilon_greedy_eps0.0"),
    ("logs/epsilon_greedy_eps0.01", "figures/epsilon_greedy_eps0.01"),
    ("logs/epsilon_greedy_eps0.1", "figures/epsilon_greedy_eps0.1"),
    ("logs/epsilon_greedy_eps0.5", "figures/epsilon_greedy_eps0.5"),
    ("logs/epsilon_greedy_eps_time_decay", "figures/epsilon_greedy_eps_time_decay"),
    ("logs/ucb_c0.1", "figures/ucb_c0.1"),
    ("logs/ucb_c0.5", "figures/ucb_c0.5"),
    ("logs/ucb_c1.0", "figures/ucb_c1.0"),
    ("logs/ucb_c2.0", "figures/ucb_c2.0"),
    ("logs/ucb_c5.0", "figures/ucb_c5.0"),
    ("logs/epsilon_greedy_eps0.0_q5.0", "figures/epsilon_greedy_eps0.0_q5.0"),
    ("logs/epsilon_greedy_eps0.0_q5.0_alpha0.1", "figures/epsilon_greedy_eps0.0_q5.0_alpha0.1"),
    ("logs/ucb_c1.0_q5.0", "figures/ucb_c1.0_q5.0"),
    ("logs/ucb_c1.0_q5.0_alpha0.1", "figures/ucb_c1.0_q5.0_alpha0.1"),
]

np.random.seed(42)

for log_dir, fig_dir in experiments:
    log_files = [f for f in os.listdir(log_dir) if f.endswith('_log.json')]
    if not log_files:
        print(f"No log file found in {log_dir}")
        continue

    log_path = os.path.join(log_dir, log_files[0])

    with open(log_path, 'r') as f:
        data = json.load(f)

    true_means = np.array(data["last_run"]["true_means"])
    q_estimates = np.array(data["last_run"]["q_estimates"])
    label = data.get("label", os.path.basename(log_dir))

    save_path = os.path.join(fig_dir, "q_comparison_boxplot.png")
    plot_q_comparison_boxplot(
        true_means=true_means,
        q_estimates=q_estimates,
        n_samples=500,
        title=f"Reward Distribution with Q-Estimates\n{label} (Last Run)",
        save_path=save_path
    )
    print(f"Generated: {save_path}")

print("\nDone! All box plots generated.")
