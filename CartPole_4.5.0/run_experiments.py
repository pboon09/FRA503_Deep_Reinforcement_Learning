#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(ROOT, "scripts", "Function_based", "train.py")
PLAY_SCRIPT = os.path.join(ROOT, "scripts", "Function_based", "play.py")
PLOT_SCRIPT = os.path.join(ROOT, "scripts", "Function_based", "visualize", "plot_report_figures.py")
TIMING_LOG = os.path.join(ROOT, "experiments", "timing_log.txt")

TASK = "Stabilize-Isaac-Cartpole-v0"
ALL_ALGOS = ["Linear_Q", "DQN", "MC_REINFORCE", "AC", "PPO"]
ALGO_ENVS = {"PPO": 256, "AC": 8, "MC_REINFORCE": 8, "DQN": 32, "Linear_Q": 32}


def log_timing(msg):
    os.makedirs(os.path.dirname(TIMING_LOG), exist_ok=True)
    with open(TIMING_LOG, "a") as f:
        f.write(msg + "\n")
    print(msg)


def train(algorithm):
    env = os.environ.copy()
    env["RL_ALGORITHM"] = algorithm
    num = ALGO_ENVS.get(algorithm, 1)
    cmd = [sys.executable, TRAIN_SCRIPT, "--task", TASK, "--num_envs", str(num), "--headless"]

    start = datetime.now()
    log_timing(f"[TRAIN] {algorithm} (num_envs={num}) started at {start.strftime('%Y-%m-%d %H:%M:%S')}")

    ok = subprocess.run(cmd, env=env, cwd=ROOT).returncode == 0

    end = datetime.now()
    elapsed = end - start
    mins = elapsed.total_seconds() / 60
    status = "OK" if ok else "FAILED"
    log_timing(f"[TRAIN] {algorithm} finished at {end.strftime('%Y-%m-%d %H:%M:%S')} | elapsed {mins:.1f} min | {status}")
    log_timing("")
    return ok


def evaluate(algorithm):
    env = os.environ.copy()
    env["RL_ALGORITHM"] = algorithm
    cmd = [sys.executable, PLAY_SCRIPT, "--task", TASK, "--num_envs", "1", "--headless"]

    start = datetime.now()
    log_timing(f"[DEPLOY] {algorithm} started at {start.strftime('%Y-%m-%d %H:%M:%S')}")

    ok = subprocess.run(cmd, env=env, cwd=ROOT).returncode == 0

    end = datetime.now()
    elapsed = end - start
    secs = elapsed.total_seconds()
    status = "OK" if ok else "FAILED"
    log_timing(f"[DEPLOY] {algorithm} finished at {end.strftime('%Y-%m-%d %H:%M:%S')} | elapsed {secs:.1f} sec | {status}")
    log_timing("")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suites", nargs="*", default=["ALL"],
                        help="ALL, train, deploy, plots")
    args = parser.parse_args()
    run = set(s.upper() for s in args.suites)

    if "ALL" in run:
        run = {"TRAIN", "DEPLOY", "PLOTS"}

    # Clear timing log for fresh run
    os.makedirs(os.path.dirname(TIMING_LOG), exist_ok=True)
    with open(TIMING_LOG, "w") as f:
        pass

    log_timing(f"{'='*60}")
    log_timing(f"Run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_timing(f"Suites: {run}")
    log_timing(f"Algo envs: {ALGO_ENVS}")
    log_timing(f"{'='*60}")
    log_timing("")

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

    log_timing(f"{'='*60}")
    log_timing(f"All done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_timing(f"{'='*60}")
    log_timing("")


if __name__ == "__main__":
    main()
