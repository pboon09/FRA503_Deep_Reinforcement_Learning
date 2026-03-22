#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(ROOT, "scripts", "Function_based", "train.py")
PLAY_SCRIPT = os.path.join(ROOT, "scripts", "Function_based", "play.py")
PLOT_SCRIPT = os.path.join(ROOT, "scripts", "Function_based", "visualize", "plot_report_figures.py")

TASK = "Stabilize-Isaac-Cartpole-v0"
ALL_ALGOS = ["Linear_Q", "DQN", "MC_REINFORCE", "AC", "PPO"]
ALGO_ENVS = {"PPO": 256, "Linear_Q": 1, "DQN": 1, "MC_REINFORCE": 1, "AC": 1}


def train(algorithm):
    env = os.environ.copy()
    env["RL_ALGORITHM"] = algorithm
    num = ALGO_ENVS.get(algorithm, 1)
    cmd = [sys.executable, TRAIN_SCRIPT, "--task", TASK, "--num_envs", str(num), "--headless"]
    print(f"\n{'='*60}\n  TRAIN: {algorithm} (num_envs={num})\n{'='*60}")
    return subprocess.run(cmd, env=env, cwd=ROOT).returncode == 0


def evaluate(algorithm):
    env = os.environ.copy()
    env["RL_ALGORITHM"] = algorithm
    cmd = [sys.executable, PLAY_SCRIPT, "--task", TASK, "--num_envs", "1", "--headless"]
    print(f"\n{'='*60}\n  EVALUATE: {algorithm}\n{'='*60}")
    return subprocess.run(cmd, env=env, cwd=ROOT).returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suites", nargs="*", default=["ALL"],
                        help="ALL, train, deploy, plots")
    args = parser.parse_args()
    run = set(s.upper() for s in args.suites)

    if "ALL" in run:
        run = {"TRAIN", "DEPLOY", "PLOTS"}

    if "TRAIN" in run:
        for algo in ALL_ALGOS:
            train(algo)

    if "DEPLOY" in run:
        for algo in ALL_ALGOS:
            evaluate(algo)

    if "PLOTS" in run:
        figures_dir = os.path.join(ROOT, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        subprocess.run([sys.executable, PLOT_SCRIPT, "--output", figures_dir], cwd=ROOT)

    print("\nDone.")


if __name__ == "__main__":
    main()
