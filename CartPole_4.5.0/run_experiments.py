#!/usr/bin/env python3
"""Automated HW2 experiment runner.

Trains all 4 baseline algorithms, collects logs, and generates comparison plots:

    python run_experiments.py
"""

import glob
import os
import shutil
import subprocess
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Paths (relative to this script's directory)
# ─────────────────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(ROOT, "scripts", "RL_Algorithm", "train.py")
PLOT_TRAINING = os.path.join(ROOT, "scripts", "RL_Algorithm", "visualize", "plot_training.py")
EXPERIMENTS_DIR = os.path.join(ROOT, "experiments", "baseline_test")
FIGURES_DIR = os.path.join(ROOT, "figures", "baseline_test")

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


def collect_csv(algo: str):
    """Copy the newest training CSV to experiments/baseline_test/{algo}.csv."""
    src_dir = os.path.join(ROOT, "logs", TASK_SHORT, algo)
    newest = find_newest(src_dir, "training_log_*.csv")
    if newest is None:
        print(f"  WARNING: No CSV found in {src_dir}")
        return None
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    dest = os.path.join(EXPERIMENTS_DIR, f"{algo}.csv")
    shutil.copy2(newest, dest)
    print(f"  {os.path.basename(newest)} -> {dest}")
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  HW2 Baseline Experiment Runner")
    print(f"  Task:      {TASK}")
    print(f"  Num envs:  {NUM_ENVS}")
    print(f"  Algos:     {ALL_ALGOS}")
    print("=" * 60)

    csv_paths = []

    for algo in ALL_ALGOS:
        ok = train(algo)
        if not ok:
            print(f"  ERROR: {algo} training failed, skipping.")
            continue
        csv = collect_csv(algo)
        if csv:
            csv_paths.append(csv)

    # Generate comparison plots
    if csv_paths:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        cmd = [sys.executable, PLOT_TRAINING, "--logs"] + csv_paths + ["--output", FIGURES_DIR]
        print(f"\n{'='*60}")
        print(f"  PLOTTING -> {FIGURES_DIR}/")
        print(f"{'='*60}\n")
        subprocess.run(cmd, cwd=ROOT)

    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"  Logs:    {EXPERIMENTS_DIR}/")
    print(f"  Figures: {FIGURES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
