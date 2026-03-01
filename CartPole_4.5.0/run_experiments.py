#!/usr/bin/env python3
"""Automated HW2 experiment runner.

Single command to train all algorithms, run resolution sweeps, and generate plots:

    python run_experiments.py
    python run_experiments.py --best-algo Q_Learning
    python run_experiments.py --suites 1        # baseline only
    python run_experiments.py --suites 1 2 3    # all suites

Suites:
  1. Baseline:          MC, SARSA, Q_Learning, Double_Q_Learning
  2. State Resolution:  best algo with coarse / medium / fine discretize_state_weight
  3. Action Resolution: best algo with 2 / 10 / 100 num_of_action
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Paths (relative to this script's directory)
# ─────────────────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "scripts", "RL_Algorithm", "configs", "rl_config.json")
TRAIN_SCRIPT = os.path.join(ROOT, "scripts", "RL_Algorithm", "train.py")
PLOT_TRAINING = os.path.join(ROOT, "scripts", "RL_Algorithm", "visualize", "plot_training.py")
PLOT_Q_SURFACE = os.path.join(ROOT, "scripts", "RL_Algorithm", "visualize", "plot_q_surface.py")
EXPERIMENTS_DIR = os.path.join(ROOT, "experiments")
FIGURES_DIR = os.path.join(ROOT, "figures")

TASK = "Stabilize-Isaac-Cartpole-v0"
TASK_SHORT = "Stabilize"
NUM_ENVS = 256

ALL_ALGOS = ["MC", "SARSA", "Q_Learning", "Double_Q_Learning"]


# ─────────────────────────────────────────────────────────────────────────────
# Config Manager — read / override / restore rl_config.json
# ─────────────────────────────────────────────────────────────────────────────

class ConfigManager:
    def __init__(self, path: str):
        self.path = path
        with open(path, "r") as f:
            self.original = json.load(f)

    def override(self, **kwargs):
        """Write config with shared-key overrides applied."""
        cfg = json.loads(json.dumps(self.original))  # deep copy
        for key, val in kwargs.items():
            cfg["shared"][key] = val
        with open(self.path, "w") as f:
            json.dump(cfg, f, indent=4)
        overrides = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"  Config override: {overrides}")

    def restore(self):
        """Restore original config to disk."""
        with open(self.path, "w") as f:
            json.dump(self.original, f, indent=4)


# ─────────────────────────────────────────────────────────────────────────────
# Execution Engine
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Artifact Router — find newest file, copy to experiment folder
# ─────────────────────────────────────────────────────────────────────────────

def find_newest(directory: str, pattern: str):
    """Return the most recently modified file matching the glob pattern."""
    matches = sorted(glob.glob(os.path.join(directory, pattern)), key=os.path.getmtime)
    return matches[-1] if matches else None


def collect_artifact(src_dir: str, pattern: str, dest_dir: str, dest_name: str):
    """Copy the newest file matching pattern from src_dir to dest_dir/dest_name."""
    newest = find_newest(src_dir, pattern)
    if newest is None:
        print(f"  WARNING: No match for {pattern} in {src_dir}")
        return None
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, dest_name)
    shutil.copy2(newest, dest)
    print(f"  {os.path.basename(newest)} -> {dest}")
    return dest


def collect_csv(algo: str, dest_dir: str, dest_name: str):
    src = os.path.join(ROOT, "logs", TASK_SHORT, algo)
    return collect_artifact(src, "training_log_*.csv", dest_dir, dest_name)


def collect_qtable(algo: str, dest_dir: str, dest_name: str):
    src = os.path.join(ROOT, "q_value", TASK_SHORT, algo)
    return collect_artifact(src, "*.json", dest_dir, dest_name)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def run_plot_training(csv_paths: list[str], output_dir: str):
    if not csv_paths:
        return
    cmd = [sys.executable, PLOT_TRAINING, "--logs"] + csv_paths + ["--output", output_dir]
    print(f"\n  plot_training -> {output_dir}/")
    subprocess.run(cmd, cwd=ROOT)


def run_plot_qtable(json_paths: list[str], output_dir: str):
    if not json_paths:
        return
    cmd = [sys.executable, PLOT_Q_SURFACE, "--qtable"] + json_paths + ["--output", output_dir]
    print(f"  plot_q_surface -> {output_dir}/")
    subprocess.run(cmd, cwd=ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Suite 1: Baseline Comparison
# ─────────────────────────────────────────────────────────────────────────────

def suite_1_baseline(cfg_mgr: ConfigManager):
    print("\n" + "#" * 60)
    print("  SUITE 1: Baseline Comparison")
    print("#" * 60)

    suite_dir = os.path.join(EXPERIMENTS_DIR, "suite_1_baseline")
    fig_dir = os.path.join(FIGURES_DIR, "suite_1_baseline")
    csv_paths, qt_paths = [], []

    cfg_mgr.restore()

    for algo in ALL_ALGOS:
        train(algo)
        csv = collect_csv(algo, suite_dir, f"{algo}.csv")
        qt = collect_qtable(algo, suite_dir, f"{algo}.json")
        if csv:
            csv_paths.append(csv)
        if qt:
            qt_paths.append(qt)

    run_plot_training(csv_paths, fig_dir)
    run_plot_qtable(qt_paths, fig_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Suite 2: State Resolution Sweep
# ─────────────────────────────────────────────────────────────────────────────

STATE_CONFIGS = {
    "coarse":  [1, 2, 1, 2],
    "medium":  [1, 8, 1, 8],
    "fine":    [2, 16, 2, 16],
}


def suite_2_state_resolution(cfg_mgr: ConfigManager, best_algo: str):
    print("\n" + "#" * 60)
    print(f"  SUITE 2: State Resolution Sweep ({best_algo})")
    print("#" * 60)

    suite_dir = os.path.join(EXPERIMENTS_DIR, "suite_2_state_res")
    fig_dir = os.path.join(FIGURES_DIR, "suite_2_state_res")
    csv_paths, qt_paths = [], []

    for tag, weights in STATE_CONFIGS.items():
        cfg_mgr.override(discretize_state_weight=weights)
        train(best_algo)
        cfg_mgr.restore()

        label = f"{best_algo}_{tag}"
        csv = collect_csv(best_algo, suite_dir, f"{label}.csv")
        qt = collect_qtable(best_algo, suite_dir, f"{label}.json")
        if csv:
            csv_paths.append(csv)
        if qt:
            qt_paths.append(qt)

    run_plot_training(csv_paths, fig_dir)
    run_plot_qtable(qt_paths, fig_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Suite 3: Action Resolution Sweep
# ─────────────────────────────────────────────────────────────────────────────

ACTION_CONFIGS = {
    "binary_2act":   2,
    "coarse_10act":  10,
    "fine_100act":   100,
}


def suite_3_action_resolution(cfg_mgr: ConfigManager, best_algo: str):
    print("\n" + "#" * 60)
    print(f"  SUITE 3: Action Resolution Sweep ({best_algo})")
    print("#" * 60)

    suite_dir = os.path.join(EXPERIMENTS_DIR, "suite_3_action_res")
    fig_dir = os.path.join(FIGURES_DIR, "suite_3_action_res")
    csv_paths, qt_paths = [], []

    for tag, n_actions in ACTION_CONFIGS.items():
        cfg_mgr.override(num_of_action=n_actions)
        train(best_algo)
        cfg_mgr.restore()

        label = f"{best_algo}_{tag}"
        csv = collect_csv(best_algo, suite_dir, f"{label}.csv")
        qt = collect_qtable(best_algo, suite_dir, f"{label}.json")
        if csv:
            csv_paths.append(csv)
        if qt:
            qt_paths.append(qt)

    run_plot_training(csv_paths, fig_dir)
    run_plot_qtable(qt_paths, fig_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="HW2 automated experiment runner.")
    p.add_argument(
        "--best-algo", default="Double_Q_Learning", choices=ALL_ALGOS,
        help="Algorithm for resolution sweep suites (default: Double_Q_Learning).",
    )
    p.add_argument(
        "--suites", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3],
        help="Which suites to run (default: 1 2 3).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  HW2 Experiment Runner")
    print(f"  Config:    {CONFIG_PATH}")
    print(f"  Task:      {TASK}")
    print(f"  Num envs:  {NUM_ENVS}")
    print(f"  Best algo: {args.best_algo}")
    print(f"  Suites:    {args.suites}")
    print("=" * 60)

    cfg_mgr = ConfigManager(CONFIG_PATH)

    try:
        if 1 in args.suites:
            suite_1_baseline(cfg_mgr)
        if 2 in args.suites:
            suite_2_state_resolution(cfg_mgr, args.best_algo)
        if 3 in args.suites:
            suite_3_action_resolution(cfg_mgr, args.best_algo)
    finally:
        cfg_mgr.restore()
        print("\nConfig restored to original.")

    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"  Figures:     {FIGURES_DIR}/")
    print(f"  Experiments: {EXPERIMENTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
