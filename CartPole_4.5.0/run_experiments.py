#!/usr/bin/env python3
"""Automated HW2 experiment runner.

Executes 4 experimental suites, collects logs + Q-tables, and generates plots:

  Suite 1: Baseline       - all 4 algorithms with default config
  Suite 2: Action Resol.  - all 4 algorithms with num_of_action = 3, 5, 11, 21
  Suite 3: State Resol.   - all 4 algorithms with weights [1,4,1,4], [1,8,1,8], [2,16,2,16]
  Suite 4: Deployment     - play.py evaluation with epsilon=0, video recording, bar charts

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
PLOT_DEPLOYMENT = os.path.join(ROOT, "scripts", "RL_Algorithm", "visualize", "plot_deployment.py")
PLOT_ANALYSIS = os.path.join(ROOT, "scripts", "RL_Algorithm", "visualize", "plot_analysis.py")
PLOT_REPORT = os.path.join(ROOT, "scripts", "RL_Algorithm", "visualize", "plot_report_figures.py")
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
             output_csv: str = "evaluation_results.csv",
             video: bool = False, video_dir: str = None,
             trajectory_dir: str = None) -> bool:
    """Run play.py to evaluate a trained Q-table. Returns True on success."""
    cmd = [
        sys.executable, PLAY_SCRIPT,
        "--task", TASK,
        "--algorithm", algorithm,
        "--qtable_path", qtable_path,
        "--num_episodes", str(num_episodes),
        "--output_csv", output_csv,
    ]

    if video and video_dir:
        cmd.extend(["--video", "--video_dir", video_dir])
    else:
        cmd.append("--headless")

    if trajectory_dir:
        cmd.extend(["--trajectory_dir", trajectory_dir])

    print(f"\n{'='*60}")
    print(f"  EVALUATE: {algorithm} ({os.path.basename(qtable_path)})")
    if video:
        print(f"  VIDEO -> {video_dir}")
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

    # ── Suite 2: Action Resolution (all algos, num_of_action sweep) ──────
    print(f"\n{'#'*60}")
    print("  SUITE 2: Action Resolution")
    print(f"{'#'*60}")

    suite2_dir = os.path.join(ROOT, "experiments", "suite_2_action")
    suite2_csvs = []
    suite2_qtables = []

    # Copy ALL baseline algorithms as the act_5 reference
    for algo in ALL_ALGOS:
        baseline_csv = os.path.join(suite1_dir, f"{algo}.csv")
        baseline_qt = os.path.join(suite1_dir, f"{algo}.json")
        if os.path.isfile(baseline_csv):
            os.makedirs(suite2_dir, exist_ok=True)
            dest = os.path.join(suite2_dir, f"{algo}_act_5.csv")
            shutil.copy2(baseline_csv, dest)
            suite2_csvs.append(dest)
            print(f"  Copied baseline -> {dest}")
        if os.path.isfile(baseline_qt):
            os.makedirs(suite2_dir, exist_ok=True)
            dest = os.path.join(suite2_dir, f"{algo}_act_5.json")
            shutil.copy2(baseline_qt, dest)
            suite2_qtables.append(dest)
            print(f"  Copied baseline -> {dest}")

    # Train ALL algorithms for each action resolution
    for n_act in [3, 11, 21]:
        original = mutate_config({"num_of_action": n_act})
        try:
            for algo in ALL_ALGOS:
                ok = train(algo)
                if not ok:
                    print(f"  ERROR: {algo} (act={n_act}) training failed.")
                    continue
                csv_path = collect_csv_named(algo, suite2_dir,
                                             f"{algo}_act_{n_act}.csv")
                qt_path = collect_qtable_named(algo, suite2_dir,
                                               f"{algo}_act_{n_act}.json")
                if csv_path:
                    suite2_csvs.append(csv_path)
                if qt_path:
                    suite2_qtables.append(qt_path)
        finally:
            restore_config(original)

    # ── Suite 3: State Resolution (all algos, discretize_state_weight sweep)
    print(f"\n{'#'*60}")
    print("  SUITE 3: State Resolution")
    print(f"{'#'*60}")

    suite3_dir = os.path.join(ROOT, "experiments", "suite_3_state")
    suite3_csvs = []
    suite3_qtables = []

    # Copy ALL baseline algorithms as the mid reference
    for algo in ALL_ALGOS:
        baseline_csv = os.path.join(suite1_dir, f"{algo}.csv")
        baseline_qt = os.path.join(suite1_dir, f"{algo}.json")
        if os.path.isfile(baseline_csv):
            os.makedirs(suite3_dir, exist_ok=True)
            dest = os.path.join(suite3_dir, f"{algo}_mid_1_8_1_8.csv")
            shutil.copy2(baseline_csv, dest)
            suite3_csvs.append(dest)
            print(f"  Copied baseline -> {dest}")
        if os.path.isfile(baseline_qt):
            os.makedirs(suite3_dir, exist_ok=True)
            dest = os.path.join(suite3_dir, f"{algo}_mid_1_8_1_8.json")
            shutil.copy2(baseline_qt, dest)
            suite3_qtables.append(dest)
            print(f"  Copied baseline -> {dest}")

    state_weight_configs = [
        {"weights": [1, 4, 1, 4], "label": "low_1_4_1_4"},
        {"weights": [2, 16, 2, 16], "label": "high_2_16_2_16"},
    ]

    # Train ALL algorithms for each state weight config
    for cfg in state_weight_configs:
        original = mutate_config({"discretize_state_weight": cfg["weights"]})
        try:
            for algo in ALL_ALGOS:
                ok = train(algo)
                if not ok:
                    print(f"  ERROR: {algo} (state={cfg['label']}) training failed.")
                    continue
                csv_path = collect_csv_named(algo, suite3_dir,
                                             f"{algo}_{cfg['label']}.csv")
                qt_path = collect_qtable_named(algo, suite3_dir,
                                               f"{algo}_{cfg['label']}.json")
                if csv_path:
                    suite3_csvs.append(csv_path)
                if qt_path:
                    suite3_qtables.append(qt_path)
        finally:
            restore_config(original)

    # ── Suite 4: Deployment Evaluation (play.py with Suite 1 Q-tables) ───
    print(f"\n{'#'*60}")
    print("  SUITE 4: Deployment Evaluation")
    print(f"{'#'*60}")

    suite4_dir = os.path.join(ROOT, "experiments", "suite_4_deployment")
    suite4_video_dir = os.path.join(suite4_dir, "videos")
    suite4_traj_dir = os.path.join(suite4_dir, "trajectories")
    os.makedirs(suite4_dir, exist_ok=True)
    eval_csv = os.path.join(suite4_dir, "evaluation_results.csv")

    # Remove stale CSV so headers are fresh
    if os.path.isfile(eval_csv):
        os.remove(eval_csv)

    for algo in ALL_ALGOS:
        qtable_path = os.path.join(suite1_dir, f"{algo}.json")
        if not os.path.isfile(qtable_path):
            print(f"  WARNING: No Q-table for {algo}, skipping evaluation.")
            continue
        algo_video_dir = os.path.join(suite4_video_dir, algo)
        evaluate(algo, qtable_path, num_episodes=10, output_csv=eval_csv,
                 video=True, video_dir=algo_video_dir,
                 trajectory_dir=suite4_traj_dir)

    # ── Report Figures (6 composite figures) ─────────────────────────────
    print(f"\n{'#'*60}")
    print("  REPORT FIGURES (6 composite figures)")
    print(f"{'#'*60}")

    figures_dir = os.path.join(ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    report_cmd = [sys.executable, PLOT_REPORT, "--output", figures_dir]
    print(f"  PLOT REPORT -> {figures_dir}/")
    subprocess.run(report_cmd, cwd=ROOT)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ALL EXPERIMENT SUITES COMPLETE")
    print(f"  Suite 1 (Baseline):    {suite1_dir}/")
    print(f"  Suite 2 (Action Res):  {suite2_dir}/")
    print(f"  Suite 3 (State Res):   {suite3_dir}/")
    print(f"  Suite 4 (Deployment):  {suite4_dir}/")
    print(f"  Figures:               {figures_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
