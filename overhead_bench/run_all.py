"""Run all overhead_bench phases sequentially and aggregate results.

Usage::
    # Quick smoke test (verifies pipe-through, tiny shapes, single GPU):
    MMF_BENCH_GPU=1 python -m overhead_bench.run_all --smoke

    # Full run when GPU is free (user should free MMF_BENCH_GPU first):
    MMF_BENCH_GPU=1 python -m overhead_bench.run_all

    # Skip specific phases:
    MMF_BENCH_GPU=1 python -m overhead_bench.run_all --skip a f

Aggregated report is dumped to ``overhead_bench/results/results_all.json`` and
printed in a compact summary table.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))

PHASES = [
    ("a", "overhead_bench.a_static_complexity"),
    ("b", "overhead_bench.b_train_cost"),
    ("c", "overhead_bench.c_finetune_cost"),
    ("d", "overhead_bench.d_inference"),
    ("e", "overhead_bench.e_onboard_one_class"),
    ("f", "overhead_bench.f_ares_baseline"),
]


def run_phase(module: str, smoke: bool) -> int:
    cmd = [sys.executable, "-m", module]
    if smoke:
        cmd.append("--smoke")
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    print(f"\n>>>>> running {module} (smoke={smoke})")
    rc = subprocess.call(cmd, env=env, cwd=ROOT)
    return rc


def load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        return {"load_error": str(exc)}


def aggregate() -> dict:
    result_dir = os.path.join(HERE, "results")
    names = {
        "A": "a_static_complexity.json",
        "B": "b_train_cost.json",
        "C": "c_finetune_cost.json",
        "D": "d_inference.json",
        "E": "e_onboard_one_class.json",
        "F": "f_ares_baseline.json",
    }
    return {k: load_json(os.path.join(result_dir, v)) for k, v in names.items()}


def print_summary(all_results: dict) -> None:
    print("\n" + "=" * 66)
    print(" MMF overhead benchmark summary")
    print("=" * 66)

    A = all_results.get("A") or {}
    B = all_results.get("B") or {}
    C = all_results.get("C") or {}
    D = all_results.get("D") or {}
    E = all_results.get("E") or {}
    F = all_results.get("F") or {}

    def g(d, *keys, default=None):
        for k in keys:
            if d is None or not isinstance(d, dict):
                return default
            d = d.get(k)
        return d if d is not None else default

    print(f" GPU                : {g(A, 'gpu_info', 'name')}")
    print(f" Torch / CUDA cap   : {g(A, 'gpu_info', 'torch_version')} / "
          f"{g(A, 'gpu_info', 'cuda_capability')}")

    print("\n-- Phase A: static complexity --")
    a1 = g(A, "A1_params_by_N") or {}
    first = next(iter(a1.values()), {}) if a1 else {}
    print(f"   Params total                 : {first.get('total', '?'):,}")
    print(f"   ckpt fp32 MiB                : {g(A, 'A3_checkpoint_size', 'fp32_MiB')}")
    print(f"   ckpt fp16 MiB                : {g(A, 'A3_checkpoint_size', 'fp16_MiB')}")

    print("\n-- Phase B: initial training @(B=1, K=1) --")
    print(f"   step wall-clock mean ms      : {g(B, 'train_step_timing_ms', 'mean_ms')}")
    print(f"   peak GPU alloc MiB           : {g(B, 'train_step_peak_gpu_mem', 'allocated_MiB')}")

    print("\n-- Phase C: fine-tune 1 epoch, +30 novel --")
    print(f"   per-iter mean ms             : {g(C, 'per_train_step', 'timing_ms', 'mean_ms')}")
    print(f"   1-epoch wall-clock (s)       : {g(C, 'one_epoch_simulated', 'wall_clock_seconds')}")
    print(f"   peak GPU alloc (MiB) epoch   : {g(C, 'one_epoch_simulated', 'peak_gpu_mem', 'allocated_MiB')}")
    print(f"   onboard 1 new class (ms)     : {g(C, 'C3_onboard_one_new_class', 'timing_ms', 'mean_ms')}")
    print(f"   full ckpt MiB                : {g(C, 'storage', 'full_ckpt_MiB')}")
    print(f"   lean ckpt MiB                : {g(C, 'storage', 'lean_ckpt_MiB')}")
    print(f"   class_bank MiB               : {g(C, 'storage', 'class_bank_MiB')}")
    print(f"   lean + bank deployment MiB   : {g(C, 'storage', 'lean_deployment_MiB_total')}")

    print("\n-- Phase D: cached inference --")
    print(f"   numerical equivalence       : {g(D, 'numerical_equivalence_check', 'passed')} "
          f"(max abs diff={g(D, 'numerical_equivalence_check', 'max_abs_diff')})")
    for row in (g(D, "scalability_sweep") or []):
        print(f"   N={row['N']:>4}  latency={row['latency_ms']['mean_ms']:.3f} ms  "
              f"peak={row['peak_gpu_mem']['allocated_MiB']:.2f} MiB")

    print("\n-- Phase E: onboard one class (base bank reused) --")
    print(f"   single-class onboarding ms  : {g(E, 'E1_single_class_onboarding', 'timing_ms', 'mean_ms')}")
    print(f"   peak GPU MiB (stable across bank growth): "
          f"{[row['peak_gpu_mem']['allocated_MiB'] for row in (g(E, 'E2_bank_growth') or [])[:5]]}")

    print("\n-- Phase F: ARES one-vs-all baseline --")
    print(f"   per-model params             : {g(F, 'training', 'per_model_params')}")
    print(f"   total params (all N)         : {g(F, 'training', 'total_params_all_models')}")
    print(f"   train wall-clock (s) 1 epoch : {g(F, 'training', 'train_wallclock_seconds')}")
    print(f"   total saved MiB              : {g(F, 'training', 'total_saved_MiB')}")
    for row in (g(F, "inference_scalability") or []):
        print(f"   N={row['N']:>4}  latency={row['latency_ms']['mean_ms']:.3f} ms  "
              f"peak={row['peak_gpu_mem']['allocated_MiB']:.2f} MiB")

    print("=" * 66)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run each phase with --smoke")
    parser.add_argument("--skip", nargs="*", default=[], help="phase letters to skip (a b c d e f)")
    args = parser.parse_args()

    failures = []
    for letter, module in PHASES:
        if letter in set(s.lower() for s in args.skip):
            print(f"[skip] phase {letter}")
            continue
        rc = run_phase(module, args.smoke)
        if rc != 0:
            failures.append(letter)

    combined = aggregate()
    out_path = os.path.join(HERE, "results", "results_all.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nAggregated results saved to {out_path}")
    print_summary(combined)

    if failures:
        print(f"\nFAILED phases: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
