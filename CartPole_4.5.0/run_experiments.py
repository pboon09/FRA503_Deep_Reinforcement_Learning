#!/usr/bin/env python3
"""Automated HW2 experiment runner.

Executes 4 experimental suites, collects logs + Q-tables, and generates plots:

  Suite 1: Baseline       - all 4 algorithms with default config
  Suite 2: Action Resol.  - Q_Learning with num_of_action = 5, 25, 50
  Suite 3: State Resol.   - Q_Learning with different discretize_state_weight
  Suite 4: Deployment     - play.py evaluation with epsilon=0

    python run_experiments.py
"""

import glob
import json
import os
import shutil
import subprocess
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Paths (relative to this script's directory)
# ─────────────────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(ROOT, "scripts", "RL_Algorithm", "train.py")
PLAY_SCRIPT = os.path.join(ROOT, "scripts", "RL_Algorithm", "play.py")
PLOT_TRAINING = os.path.join(ROOT, "scripts", "RL_Algorithm", "visualize", "plot_training.py")
PLOT_Q_SURFACE = os.path.join(ROOT, "scripts", "RL_Algorithm", "visualize", "plot_q_surface.py")
CONFIG_PATH = os.path.join(ROOT, "scripts", "RL_Algorithm", "configs", "rl_config.json")

TASK = "Stabilize-Isaac-Cartpole-v0"
TASK_SHORT = "Stabilize"
NUM_ENVS = 256

ALL_ALGOS = ["MC", "SARSA", "Q_Learning", "Double_Q_Learning"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_newest(directory: str, pattern: str):
    """Return the most recently modified file matching the glob pattern."""
    matches = sorted(glob.glob(os.path.join(directory, pattern)), key=os.path.getmtime)
    return matches[-1] if matches else None


def train(algorithm: str) -> bool:
    """Run a single training session. Returns True on success."""
    env = os.environ.copy()
    env["RL_ALGORITHM"] = algorithm

    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--task", TASK,
        "--num_envs", str(NUM_ENVS),
        "--headless",
    ]

    print(f"\n{'='*60}")
    print(f"  TRAIN: {algorithm}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, env=env, cwd=ROOT)
    return result.returncode == 0


def evaluate(algorithm: str, qtable_path: str, num_episodes: int = 10,
             output_csv: str = "evaluation_results.csv") -> bool:
    """Run play.py to evaluate a trained Q-table. Returns True on success."""
    cmd = [
        sys.executable, PLAY_SCRIPT,
        "--task", TASK,
        "--algorithm", algorithm,
        "--qtable_path", qtable_path,
        "--num_episodes", str(num_episodes),
        "--output_csv", output_csv,
        "--headless",
    ]

    print(f"\n{'='*60}")
    print(f"  EVALUATE: {algorithm} ({os.path.basename(qtable_path)})")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, env=os.environ.copy(), cwd=ROOT)
    return result.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# Config mutator
# ─────────────────────────────────────────────────────────────────────────────

def mutate_config(updates: dict) -> dict:
    """Read rl_config.json, apply updates to 'shared' block, write back.

    Returns the original config dict (pass to restore_config to revert).
    """
    with open(CONFIG_PATH, "r") as f:
        original = json.load(f)

    modified = json.loads(json.dumps(original))  # deep copy
    for key, value in updates.items():
        modified["shared"][key] = value

    with open(CONFIG_PATH, "w") as f:
        json.dump(modified, f, indent=4)

    print(f"  CONFIG MUTATED: {updates}")
    return original


def restore_config(original_data: dict):
    """Revert rl_config.json to its original content."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(original_data, f, indent=4)
    print("  CONFIG RESTORED to original.")


# ─────────────────────────────────────────────────────────────────────────────
# Collect helpers
# ─────────────────────────────────────────────────────────────────────────────

def collect_csv_named(algo: str, dest_dir: str, dest_name: str):
    """Copy the newest training CSV to dest_dir/dest_name."""
    src_dir = os.path.join(ROOT, "logs", TASK_SHORT, algo)
    newest = find_newest(src_dir, "training_log_*.csv")
    if newest is None:
        print(f"  WARNING: No CSV found in {src_dir}")
        return None
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, dest_name)
    shutil.copy2(newest, dest)
    print(f"  {os.path.basename(newest)} -> {dest}")
    return dest


def collect_qtable_named(algo: str, dest_dir: str, dest_name: str):
    """Copy the newest Q-table JSON to dest_dir/dest_name."""
    src_dir = os.path.join(ROOT, "q_value", TASK_SHORT, algo)
    newest = find_newest(src_dir, "*.json")
    if newest is None:
        print(f"  WARNING: No Q-table found in {src_dir}")
        return None
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, dest_name)
    shutil.copy2(newest, dest)
    print(f"  {os.path.basename(newest)} -> {dest}")
    return dest


def run_plots(csv_paths: list, qtable_paths: list, figures_dir: str):
    """Generate training and Q-surface plots for a set of experiments."""
    if csv_paths:
        os.makedirs(figures_dir, exist_ok=True)
        cmd = [sys.executable, PLOT_TRAINING, "--logs"] + csv_paths + ["--output", figures_dir]
        print(f"\n  PLOT TRAINING -> {figures_dir}/")
        subprocess.run(cmd, cwd=ROOT)

    if qtable_paths:
        q_fig_dir = os.path.join(figures_dir, "q_surface")
        os.makedirs(q_fig_dir, exist_ok=True)
        cmd = [sys.executable, PLOT_Q_SURFACE, "--qtable"] + qtable_paths + ["--output", q_fig_dir]
        print(f"\n  PLOT Q-SURFACE -> {q_fig_dir}/")
        subprocess.run(cmd, cwd=ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  HW2 Full Experiment Suite Runner")
    print(f"  Task:      {TASK}")
    print(f"  Num envs:  {NUM_ENVS}")
    print(f"  Algos:     {ALL_ALGOS}")
    print("=" * 60)

    # ── Suite 1: Baseline (all 4 algos, default config) ──────────────────
    print(f"\n{'#'*60}")
    print("  SUITE 1: Baseline")
    print(f"{'#'*60}")

    suite1_dir = os.path.join(ROOT, "experiments", "suite_1_baseline")
    suite1_fig = os.path.join(ROOT, "figures", "suite_1_baseline")
    suite1_csvs = []
    suite1_qtables = []

    for algo in ALL_ALGOS:
        ok = train(algo)
        if not ok:
            print(f"  ERROR: {algo} training failed, skipping.")
            continue
        csv_path = collect_csv_named(algo, suite1_dir, f"{algo}.csv")
        qt_path = collect_qtable_named(algo, suite1_dir, f"{algo}.json")
        if csv_path:
            suite1_csvs.append(csv_path)
        if qt_path:
            suite1_qtables.append(qt_path)

    run_plots(suite1_csvs, suite1_qtables, suite1_fig)

    # ── Suite 2: Action Resolution (Q_Learning, num_of_action sweep) ─────
    print(f"\n{'#'*60}")
    print("  SUITE 2: Action Resolution")
    print(f"{'#'*60}")

    suite2_dir = os.path.join(ROOT, "experiments", "suite_2_action")
    suite2_fig = os.path.join(ROOT, "figures", "suite_2_action")
    suite2_csvs = []
    suite2_qtables = []

    # Copy baseline Q_Learning as the act_5 reference
    baseline_ql_csv = os.path.join(suite1_dir, "Q_Learning.csv")
    baseline_ql_qt = os.path.join(suite1_dir, "Q_Learning.json")
    if os.path.isfile(baseline_ql_csv):
        os.makedirs(suite2_dir, exist_ok=True)
        dest = os.path.join(suite2_dir, "Q_Learning_act_5.csv")
        shutil.copy2(baseline_ql_csv, dest)
        suite2_csvs.append(dest)
        print(f"  Copied baseline -> {dest}")
    if os.path.isfile(baseline_ql_qt):
        os.makedirs(suite2_dir, exist_ok=True)
        dest = os.path.join(suite2_dir, "Q_Learning_act_5.json")
        shutil.copy2(baseline_ql_qt, dest)
        suite2_qtables.append(dest)
        print(f"  Copied baseline -> {dest}")

    for n_act in [25, 50]:
        original = mutate_config({"num_of_action": n_act})
        try:
            ok = train("Q_Learning")
            if not ok:
                print(f"  ERROR: Q_Learning (act={n_act}) training failed.")
                continue
            csv_path = collect_csv_named("Q_Learning", suite2_dir,
                                         f"Q_Learning_act_{n_act}.csv")
            qt_path = collect_qtable_named("Q_Learning", suite2_dir,
                                            f"Q_Learning_act_{n_act}.json")
            if csv_path:
                suite2_csvs.append(csv_path)
            if qt_path:
                suite2_qtables.append(qt_path)
        finally:
            restore_config(original)

    run_plots(suite2_csvs, suite2_qtables, suite2_fig)

    # ── Suite 3: State Resolution (Q_Learning, discretize_state_weight sweep)
    print(f"\n{'#'*60}")
    print("  SUITE 3: State Resolution")
    print(f"{'#'*60}")

    suite3_dir = os.path.join(ROOT, "experiments", "suite_3_state")
    suite3_fig = os.path.join(ROOT, "figures", "suite_3_state")
    suite3_csvs = []
    suite3_qtables = []

    # Copy baseline Q_Learning as the mid reference
    if os.path.isfile(baseline_ql_csv):
        os.makedirs(suite3_dir, exist_ok=True)
        dest = os.path.join(suite3_dir, "Q_Learning_mid_1_8_1_8.csv")
        shutil.copy2(baseline_ql_csv, dest)
        suite3_csvs.append(dest)
        print(f"  Copied baseline -> {dest}")
    if os.path.isfile(baseline_ql_qt):
        os.makedirs(suite3_dir, exist_ok=True)
        dest = os.path.join(suite3_dir, "Q_Learning_mid_1_8_1_8.json")
        shutil.copy2(baseline_ql_qt, dest)
        suite3_qtables.append(dest)
        print(f"  Copied baseline -> {dest}")

    state_weight_configs = [
        {"weights": [1, 2, 1, 2], "label": "low_1_2_1_2"},
        {"weights": [2, 16, 2, 16], "label": "high_2_16_2_16"},
    ]

    for cfg in state_weight_configs:
        original = mutate_config({"discretize_state_weight": cfg["weights"]})
        try:
            ok = train("Q_Learning")
            if not ok:
                print(f"  ERROR: Q_Learning (state={cfg['label']}) training failed.")
                continue
            csv_path = collect_csv_named("Q_Learning", suite3_dir,
                                         f"Q_Learning_{cfg['label']}.csv")
            qt_path = collect_qtable_named("Q_Learning", suite3_dir,
                                            f"Q_Learning_{cfg['label']}.json")
            if csv_path:
                suite3_csvs.append(csv_path)
            if qt_path:
                suite3_qtables.append(qt_path)
        finally:
            restore_config(original)

    run_plots(suite3_csvs, suite3_qtables, suite3_fig)

    # ── Suite 4: Deployment Evaluation (play.py with Suite 1 Q-tables) ───
    print(f"\n{'#'*60}")
    print("  SUITE 4: Deployment Evaluation")
    print(f"{'#'*60}")

    suite4_dir = os.path.join(ROOT, "experiments", "suite_4_deployment")
    os.makedirs(suite4_dir, exist_ok=True)
    eval_csv = os.path.join(suite4_dir, "evaluation_results.csv")

    for algo in ALL_ALGOS:
        qtable_path = os.path.join(suite1_dir, f"{algo}.json")
        if not os.path.isfile(qtable_path):
            print(f"  WARNING: No Q-table for {algo}, skipping evaluation.")
            continue
        evaluate(algo, qtable_path, num_episodes=10, output_csv=eval_csv)

    if os.path.isfile(eval_csv):
        print(f"\n  Evaluation results: {eval_csv}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ALL EXPERIMENT SUITES COMPLETE")
    print(f"  Suite 1 (Baseline):    {suite1_dir}/")
    print(f"  Suite 2 (Action Res):  {suite2_dir}/")
    print(f"  Suite 3 (State Res):   {suite3_dir}/")
    print(f"  Suite 4 (Deployment):  {suite4_dir}/")
    print(f"  Figures:               {os.path.join(ROOT, 'figures')}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
