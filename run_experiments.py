"""
MMF Batch Experiment Runner
Usage:
    # Run base training on 3/4/5-tab:
    python run_experiments.py --stage base --tabs 3 4 5

    # Run few-shot fine-tuning (Exp3: fixed novel set, varying shots):
    python run_experiments.py --stage fewshot --tabs 3 4 5 --shots 5 10 15 20 30

    # Run continual onboarding (Exp4: fixed shots=20, varying novel counts):
    python run_experiments.py --stage onboard --tabs 3 4 5 --novels 5 10 15 20 25 30 35
"""

import subprocess
import sys
import os
import argparse
from datetime import datetime


def run_and_log(cmd, description, log_file):
    """Run a command and simultaneously log stdout to file and console."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    with open(log_file, 'a', encoding='utf-8') as f:
        header = (
            f"\n{'='*60}\n"
            f"Running: {description}\n"
            f"Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Command: {' '.join(cmd)}\n"
            f"{'='*60}\n"
        )
        f.write(header)
        print(header, end="")

        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False
            )
            print(res.stdout, end="")
            f.write(res.stdout)
            f.write("\n")

            if res.returncode != 0:
                msg = f"\n[WARNING] {description} failed with return code {res.returncode}\n"
                print(msg)
                f.write(msg)
        except Exception as e:
            msg = f"\n[ERROR] Execution failed: {e}\n"
            print(msg)
            f.write(msg)


def run_base_training(tabs):
    """Stage 1: Base training with train_enhanced.py."""
    print(f"\n{'#'*60}")
    print(f"# Stage 1: Base Training  (tabs={tabs})")
    print(f"{'#'*60}")
    for tab in tabs:
        config_path = f"configs/base_train/example_{tab}tab.json"
        if not os.path.exists(config_path):
            print(f"[SKIP] Config not found: {config_path}")
            continue
        log_file = f"lot_exp_result/base_train/base_{tab}tab.txt"
        cmd = [sys.executable, "train_enhanced.py", "--config", config_path]
        run_and_log(cmd, f"Base-Train_{tab}tab", log_file)


def run_fewshot(tabs, shots):
    """Stage 2 / Exp3: Few-shot fine-tuning with varying shots and tabs."""
    print(f"\n{'#'*60}")
    print(f"# Stage 2: Few-Shot Fine-Tuning  (tabs={tabs}, shots={shots})")
    print(f"{'#'*60}")
    for shot in shots:
        for tab in tabs:
            config_path = (
                f"configs/fewshot/"
                f"{tab}_tab/{shot}shot/"
                f"mmf_{shot}shot_{tab}tab.json"
            )
            if not os.path.exists(config_path):
                print(f"[SKIP] Config not found: {config_path}")
                continue
            log_file = f"lot_exp_result/fewshot/mmf_{shot}shot_{tab}tab.txt"
            cmd = [sys.executable, "finetune.py", "--config", config_path]
            run_and_log(cmd, f"FewShot_{shot}shot_{tab}tab", log_file)


def run_onboarding(tabs, novels, shot=20):
    """Exp4: Continual onboarding - fixed shots, varying novel class counts."""
    print(f"\n{'#'*60}")
    print(f"# Exp4: Continual Onboarding  (shot={shot}, novels={novels}, tabs={tabs})")
    print(f"{'#'*60}")
    for num_novel in novels:
        for tab in tabs:
            config_path = (
                f"configs/fewshot/"
                f"{tab}tab/{num_novel}novel/"
                f"mmf_{shot}shot_{num_novel}novel_{tab}tab.json"
            )
            if not os.path.exists(config_path):
                print(f"[SKIP] Config not found: {config_path}")
                continue
            log_file = f"lot_exp_result/onboard/mmf_{shot}shot_{num_novel}novel_{tab}tab.txt"
            cmd = [sys.executable, "finetune.py", "--config", config_path]
            run_and_log(cmd, f"Onboard_{num_novel}novel_{tab}tab", log_file)


def main():
    parser = argparse.ArgumentParser(description="MMF Batch Experiment Runner")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["base", "fewshot", "onboard"],
                        help="Experiment stage: base | fewshot | onboard")
    parser.add_argument("--tabs", type=int, nargs="+", default=[3, 4, 5],
                        help="Tab counts to run (default: 3 4 5)")
    parser.add_argument("--shots", type=int, nargs="+", default=[5, 10, 15, 20, 30],
                        help="K-shot values for fewshot stage")
    parser.add_argument("--novels", type=int, nargs="+",
                        default=[5, 10, 15, 20, 25, 30, 35],
                        help="Novel class counts for onboard stage")
    parser.add_argument("--onboard-shot", type=int, default=20,
                        help="Fixed K-shot for onboarding experiment (default: 20)")
    args = parser.parse_args()

    if args.stage == "base":
        run_base_training(args.tabs)
    elif args.stage == "fewshot":
        run_fewshot(args.tabs, args.shots)
    elif args.stage == "onboard":
        run_onboarding(args.tabs, args.novels, shot=args.onboard_shot)

    print(f"\n{'='*60}")
    print("All experiments finished.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
