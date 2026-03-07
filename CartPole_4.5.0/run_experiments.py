#!/usr/bin/env python3
"""Automated HW2 experiment runner.

Executes 8 experimental suites, collects logs + Q-tables, and generates plots:

  Suite 1: Baseline       - all 4 algorithms with default config
  Suite 2: Action Resol.  - all 4 algorithms with num_of_action = 3, 5, 11, 21
  Suite 3: State Resol.   - all 4 algorithms with weights [1,4,1,4], [1,8,1,8], [2,16,2,16]
  Suite 4: Deployment     - play.py evaluation with epsilon=0, video recording
  Suite 5: LR Sweep       - all 4 algorithms with alpha = 0.01, 0.05, 0.1, 0.3, 0.5, 0.9
  Suite 6: Epsilon Sched  - per_step, per_episode, fixed epsilon decay modes
  Suite 7: Gamma Sweep    - all 4 algorithms with gamma = 0.9, 0.95, 0.99, 0.999, 1.0
  Suite 8: Q0 Init Sweep  - all 4 algorithms with Q0 = 0, 10, 50, 100

    python run_experiments.py
"""

import argparse
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


def mutate_config_full(shared_updates: dict = None, algo_lr: float = None) -> dict:
    """Read rl_config.json, apply shared updates and/or set all algorithm LRs.

    Returns the original config dict (pass to restore_config to revert).
    """
    with open(CONFIG_PATH, "r") as f:
        original = json.load(f)

    modified = json.loads(json.dumps(original))  # deep copy
    if shared_updates:
        for key, value in shared_updates.items():
            modified["shared"][key] = value
    if algo_lr is not None:
        for algo_name in modified["algorithms"]:
            modified["algorithms"][algo_name]["learning_rate"] = algo_lr

    with open(CONFIG_PATH, "w") as f:
        json.dump(modified, f, indent=4)

    msg_parts = []
    if shared_updates:
        msg_parts.append(f"shared={shared_updates}")
    if algo_lr is not None:
        msg_parts.append(f"all_algo_lr={algo_lr}")
    print(f"  CONFIG MUTATED: {', '.join(msg_parts)}")
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

SUITE_NAMES = {
    "1": "Baseline",
    "2": "Action Resolution",
    "3": "State Resolution",
    "4": "Deployment",
    "5": "LR Sweep",
    "6": "Epsilon Schedule",
    "7": "Gamma Sweep",
    "8": "Q0 Init Sweep",
    "plots": "Report Figures",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="HW2 experiment runner. Select suites to run.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "suites", nargs="*", default=["ALL"],
        help=(
            "Which suites to run. Options:\n"
            "  ALL          - run everything (default)\n"
            "  1            - Baseline\n"
            "  2            - Action Resolution sweep\n"
            "  3            - State Resolution sweep\n"
            "  4            - Deployment evaluation\n"
            "  5            - Learning Rate sweep\n"
            "  6            - Epsilon Schedule sweep\n"
            "  7            - Gamma sweep\n"
            "  8            - Q0 Init sweep\n"
            "  plots        - Generate report figures only\n"
            "\n"
            "Examples:\n"
            "  python run_experiments.py              # run all\n"
            "  python run_experiments.py 1 4 plots    # baseline + deploy + plots\n"
            "  python run_experiments.py 1 2 3        # suites 1-3\n"
            "  python run_experiments.py plots        # just regenerate figures\n"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve which suites to run
    if "ALL" in [s.upper() for s in args.suites]:
        run_suites = set(SUITE_NAMES.keys())
    else:
        run_suites = set(args.suites)
        invalid = run_suites - set(SUITE_NAMES.keys())
        if invalid:
            print(f"ERROR: Unknown suite(s): {invalid}")
            print(f"Valid options: ALL, {', '.join(SUITE_NAMES.keys())}")
            sys.exit(1)

    selected = [f"  {k}: {SUITE_NAMES[k]}" for k in sorted(SUITE_NAMES) if k in run_suites]

    print("=" * 60)
    print("  HW2 Full Experiment Suite Runner")
    print(f"  Task:      {TASK}")
    print(f"  Num envs:  {NUM_ENVS}")
    print(f"  Algos:     {ALL_ALGOS}")
    print(f"  Suites:")
    for s in selected:
        print(s)
    print("=" * 60)

    # ── Suite 1: Baseline (all 4 algos, default config) ──────────────────
    suite1_dir = os.path.join(ROOT, "experiments", "suite_1_baseline")

    if "1" in run_suites:
        print(f"\n{'#'*60}")
        print("  SUITE 1: Baseline")
        print(f"{'#'*60}")

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
    suite2_dir = os.path.join(ROOT, "experiments", "suite_2_action")

    if "2" in run_suites:
        print(f"\n{'#'*60}")
        print("  SUITE 2: Action Resolution")
        print(f"{'#'*60}")

        # Copy ALL baseline algorithms as the act_5 reference
        for algo in ALL_ALGOS:
            baseline_csv = os.path.join(suite1_dir, f"{algo}.csv")
            baseline_qt = os.path.join(suite1_dir, f"{algo}.json")
            if os.path.isfile(baseline_csv):
                os.makedirs(suite2_dir, exist_ok=True)
                dest = os.path.join(suite2_dir, f"{algo}_act_5.csv")
                shutil.copy2(baseline_csv, dest)
                print(f"  Copied baseline -> {dest}")
            if os.path.isfile(baseline_qt):
                os.makedirs(suite2_dir, exist_ok=True)
                dest = os.path.join(suite2_dir, f"{algo}_act_5.json")
                shutil.copy2(baseline_qt, dest)
                print(f"  Copied baseline -> {dest}")

        for n_act in [3, 11, 21]:
            original = mutate_config({"num_of_action": n_act})
            try:
                for algo in ALL_ALGOS:
                    ok = train(algo)
                    if not ok:
                        print(f"  ERROR: {algo} (act={n_act}) training failed.")
                        continue
                    collect_csv_named(algo, suite2_dir, f"{algo}_act_{n_act}.csv")
                    collect_qtable_named(algo, suite2_dir, f"{algo}_act_{n_act}.json")
            finally:
                restore_config(original)

    # ── Suite 3: State Resolution (all algos, discretize_state_weight sweep)
    suite3_dir = os.path.join(ROOT, "experiments", "suite_3_state")

    if "3" in run_suites:
        print(f"\n{'#'*60}")
        print("  SUITE 3: State Resolution")
        print(f"{'#'*60}")

        # Copy ALL baseline algorithms as the mid reference
        for algo in ALL_ALGOS:
            baseline_csv = os.path.join(suite1_dir, f"{algo}.csv")
            baseline_qt = os.path.join(suite1_dir, f"{algo}.json")
            if os.path.isfile(baseline_csv):
                os.makedirs(suite3_dir, exist_ok=True)
                dest = os.path.join(suite3_dir, f"{algo}_mid_1_8_1_8.csv")
                shutil.copy2(baseline_csv, dest)
                print(f"  Copied baseline -> {dest}")
            if os.path.isfile(baseline_qt):
                os.makedirs(suite3_dir, exist_ok=True)
                dest = os.path.join(suite3_dir, f"{algo}_mid_1_8_1_8.json")
                shutil.copy2(baseline_qt, dest)
                print(f"  Copied baseline -> {dest}")

        state_weight_configs = [
            {"weights": [1, 4, 1, 4], "label": "low_1_4_1_4"},
            {"weights": [2, 16, 2, 16], "label": "high_2_16_2_16"},
        ]

        for cfg in state_weight_configs:
            original = mutate_config({"discretize_state_weight": cfg["weights"]})
            try:
                for algo in ALL_ALGOS:
                    ok = train(algo)
                    if not ok:
                        print(f"  ERROR: {algo} (state={cfg['label']}) training failed.")
                        continue
                    collect_csv_named(algo, suite3_dir, f"{algo}_{cfg['label']}.csv")
                    collect_qtable_named(algo, suite3_dir, f"{algo}_{cfg['label']}.json")
            finally:
                restore_config(original)

    # ── Suite 4: Deployment Evaluation (play.py with Suite 1 Q-tables) ───
    suite4_dir = os.path.join(ROOT, "experiments", "suite_4_deployment")

    if "4" in run_suites:
        print(f"\n{'#'*60}")
        print("  SUITE 4: Deployment Evaluation")
        print(f"{'#'*60}")

        suite4_video_dir = os.path.join(suite4_dir, "videos")
        suite4_traj_dir = os.path.join(suite4_dir, "trajectories")
        os.makedirs(suite4_dir, exist_ok=True)
        eval_csv = os.path.join(suite4_dir, "evaluation_results.csv")

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

    # ── Suite 5: Learning Rate Sweep ─────────────────────────────────────
    suite5_dir = os.path.join(ROOT, "experiments", "suite_5_lr")

    if "5" in run_suites:
        print(f"\n{'#'*60}")
        print("  SUITE 5: Learning Rate Sweep")
        print(f"{'#'*60}")

        # Copy baseline as lr_0.1 reference
        for algo in ALL_ALGOS:
            baseline_csv = os.path.join(suite1_dir, f"{algo}.csv")
            baseline_qt = os.path.join(suite1_dir, f"{algo}.json")
            if os.path.isfile(baseline_csv):
                os.makedirs(suite5_dir, exist_ok=True)
                dest = os.path.join(suite5_dir, f"{algo}_lr_0.1.csv")
                shutil.copy2(baseline_csv, dest)
                print(f"  Copied baseline -> {dest}")
            if os.path.isfile(baseline_qt):
                os.makedirs(suite5_dir, exist_ok=True)
                dest = os.path.join(suite5_dir, f"{algo}_lr_0.1.json")
                shutil.copy2(baseline_qt, dest)
                print(f"  Copied baseline -> {dest}")

        for lr_val in [0.01, 0.05, 0.3, 0.5, 0.9]:
            original = mutate_config_full(algo_lr=lr_val)
            try:
                for algo in ALL_ALGOS:
                    ok = train(algo)
                    if not ok:
                        print(f"  ERROR: {algo} (lr={lr_val}) training failed.")
                        continue
                    collect_csv_named(algo, suite5_dir, f"{algo}_lr_{lr_val}.csv")
                    collect_qtable_named(algo, suite5_dir, f"{algo}_lr_{lr_val}.json")
            finally:
                restore_config(original)

    # ── Suite 6: Epsilon Schedule Sweep ───────────────────────────────────
    suite6_dir = os.path.join(ROOT, "experiments", "suite_6_epsilon")

    if "6" in run_suites:
        print(f"\n{'#'*60}")
        print("  SUITE 6: Epsilon Schedule Sweep")
        print(f"{'#'*60}")

        # Copy baseline as eps_per_step_0.9995 reference
        for algo in ALL_ALGOS:
            baseline_csv = os.path.join(suite1_dir, f"{algo}.csv")
            baseline_qt = os.path.join(suite1_dir, f"{algo}.json")
            if os.path.isfile(baseline_csv):
                os.makedirs(suite6_dir, exist_ok=True)
                dest = os.path.join(suite6_dir, f"{algo}_per_step_0.9995.csv")
                shutil.copy2(baseline_csv, dest)
                print(f"  Copied baseline -> {dest}")
            if os.path.isfile(baseline_qt):
                os.makedirs(suite6_dir, exist_ok=True)
                dest = os.path.join(suite6_dir, f"{algo}_per_step_0.9995.json")
                shutil.copy2(baseline_qt, dest)
                print(f"  Copied baseline -> {dest}")

        eps_configs = [
            {"mode": "per_step",    "decay": 0.9995, "start": 1.0, "label": "per_step_0.9995"},
            {"mode": "per_step",    "decay": 0.999,  "start": 1.0, "label": "per_step_0.999"},
            {"mode": "per_step",    "decay": 0.9999, "start": 1.0, "label": "per_step_0.9999"},
            {"mode": "per_episode", "decay": 0.995,  "start": 1.0, "label": "per_episode_0.995"},
            {"mode": "per_episode", "decay": 0.99,   "start": 1.0, "label": "per_episode_0.99"},
            {"mode": "per_episode", "decay": 0.9995, "start": 1.0, "label": "per_episode_0.9995"},
            {"mode": "fixed",       "decay": 1.0,    "start": 0.1, "label": "fixed_0.1"},
        ]

        for ecfg in eps_configs:
            original = mutate_config_full(shared_updates={
                "epsilon_decay_mode": ecfg["mode"],
                "epsilon_decay": ecfg["decay"],
                "start_epsilon": ecfg["start"],
            })
            try:
                for algo in ALL_ALGOS:
                    ok = train(algo)
                    if not ok:
                        print(f"  ERROR: {algo} (eps={ecfg['label']}) training failed.")
                        continue
                    collect_csv_named(algo, suite6_dir, f"{algo}_{ecfg['label']}.csv")
                    collect_qtable_named(algo, suite6_dir, f"{algo}_{ecfg['label']}.json")
            finally:
                restore_config(original)

    # ── Suite 7: Discount Factor Sweep ────────────────────────────────────
    suite7_dir = os.path.join(ROOT, "experiments", "suite_7_gamma")

    if "7" in run_suites:
        print(f"\n{'#'*60}")
        print("  SUITE 7: Discount Factor Sweep")
        print(f"{'#'*60}")

        # Copy baseline as gamma_0.99 reference
        for algo in ALL_ALGOS:
            baseline_csv = os.path.join(suite1_dir, f"{algo}.csv")
            baseline_qt = os.path.join(suite1_dir, f"{algo}.json")
            if os.path.isfile(baseline_csv):
                os.makedirs(suite7_dir, exist_ok=True)
                dest = os.path.join(suite7_dir, f"{algo}_gamma_0.99.csv")
                shutil.copy2(baseline_csv, dest)
                print(f"  Copied baseline -> {dest}")
            if os.path.isfile(baseline_qt):
                os.makedirs(suite7_dir, exist_ok=True)
                dest = os.path.join(suite7_dir, f"{algo}_gamma_0.99.json")
                shutil.copy2(baseline_qt, dest)
                print(f"  Copied baseline -> {dest}")

        for gamma_val in [0.9, 0.95, 0.999, 1.0]:
            original = mutate_config_full(shared_updates={"discount": gamma_val})
            try:
                for algo in ALL_ALGOS:
                    ok = train(algo)
                    if not ok:
                        print(f"  ERROR: {algo} (gamma={gamma_val}) training failed.")
                        continue
                    collect_csv_named(algo, suite7_dir, f"{algo}_gamma_{gamma_val}.csv")
                    collect_qtable_named(algo, suite7_dir, f"{algo}_gamma_{gamma_val}.json")
            finally:
                restore_config(original)

    # ── Suite 8: Q₀ Initialization Sweep ──────────────────────────────────
    suite8_dir = os.path.join(ROOT, "experiments", "suite_8_q_init")

    if "8" in run_suites:
        print(f"\n{'#'*60}")
        print("  SUITE 8: Q0 Initialization Sweep")
        print(f"{'#'*60}")

        # Copy baseline as q_init_0.0 reference
        for algo in ALL_ALGOS:
            baseline_csv = os.path.join(suite1_dir, f"{algo}.csv")
            baseline_qt = os.path.join(suite1_dir, f"{algo}.json")
            if os.path.isfile(baseline_csv):
                os.makedirs(suite8_dir, exist_ok=True)
                dest = os.path.join(suite8_dir, f"{algo}_q_init_0.0.csv")
                shutil.copy2(baseline_csv, dest)
                print(f"  Copied baseline -> {dest}")
            if os.path.isfile(baseline_qt):
                os.makedirs(suite8_dir, exist_ok=True)
                dest = os.path.join(suite8_dir, f"{algo}_q_init_0.0.json")
                shutil.copy2(baseline_qt, dest)
                print(f"  Copied baseline -> {dest}")

        for q0_val in [10.0, 50.0, 100.0]:
            original = mutate_config_full(shared_updates={"q_init": q0_val})
            try:
                for algo in ALL_ALGOS:
                    ok = train(algo)
                    if not ok:
                        print(f"  ERROR: {algo} (q_init={q0_val}) training failed.")
                        continue
                    collect_csv_named(algo, suite8_dir, f"{algo}_q_init_{q0_val}.csv")
                    collect_qtable_named(algo, suite8_dir, f"{algo}_q_init_{q0_val}.json")
            finally:
                restore_config(original)

    # ── Report Figures ────────────────────────────────────────────────────
    figures_dir = os.path.join(ROOT, "figures")

    if "plots" in run_suites:
        print(f"\n{'#'*60}")
        print("  REPORT FIGURES")
        print(f"{'#'*60}")

        os.makedirs(figures_dir, exist_ok=True)
        report_cmd = [sys.executable, PLOT_REPORT, "--output", figures_dir]
        print(f"  PLOT REPORT -> {figures_dir}/")
        subprocess.run(report_cmd, cwd=ROOT)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPLETED SUITES:")
    for k in sorted(SUITE_NAMES):
        if k in run_suites:
            print(f"    {k}: {SUITE_NAMES[k]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
