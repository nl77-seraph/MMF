# overhead_bench — MMF System-Overhead Benchmark Pipeline

This pipeline is used to generate the compute / memory / storage / scalability data required for the rebuttal. All scripts evaluate single-sample overhead under a **single GPU + batch=1** setting. By default, they use `CUDA_VISIBLE_DEVICES=1`, which can be changed via the `MMF_BENCH_GPU` environment variable.

## File Structure

| File                        | Purpose                                                                                                                                                                                                                                                      |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `bench_utils.py`            | Common utilities: single-GPU pinning, CUDA event timing, GPU memory / CPU RSS sampling, THOP FLOPs, and JSON dumping.                                                                                                                                        |
| `cached_inference_model.py` | MMF **lean inference model**: removes the `meta_learnet` and `feature_reweighting` submodules, and directly uses the cached `W_c` for broadcast multiplication. It is numerically equivalent to the original model.                                          |
| `a_static_complexity.py`    | A1 Params / A2 FLOPs (sweep over N, with the N-dependent formula) / A3 Checkpoint storage in fp32+fp16 / A4 Big-O formulas.                                                                                                                                  |
| `b_train_cost.py`           | Initial training @(B=1, K=1): GPU model / params / FLOPs / per-step latency / peak GPU + CPU memory.                                                                                                                                                         |
| `c_finetune_cost.py`        | Finetuning (+30 novel classes): per-step latency, one-epoch wall-clock time, GPU memory, saving **full ckpt / lean ckpt / class_bank**, and computing storage cost; C3 “onboard 1 class” only measures the latency of the support forward pass.              |
| `d_inference.py`            | Loads the lean ckpt + class_bank and measures per-sample latency + peak GPU memory for each N∈{5,10,30,95}; includes a built-in full-vs-cached numerical equivalence check.                                                                                  |
| `e_onboard_one_class.py`    | Freezes `feature_extractor` + `meta_learnet`, reuses the base bank, and measures the latency and GPU memory of single-class onboarding; also runs a bank-size sweep from 1→30 to demonstrate O(1) behavior.                                                  |
| `f_ares_baseline.py`        | Original ARES paper setting: N independent binary `Trans-WF` classifiers. Trains for one epoch on synthetic data → saves to disk → serially runs all models for one sample. Reports training latency, total storage, inference latency, and peak GPU memory. |
| `run_all.py`                | Runs A→F sequentially, aggregates outputs into `results/results_all.json`, and prints a summary table.                                                                                                                                                       |
| `configs/bench_config.json` | Reference parameters for the full experiment. This is only for bookkeeping; the default values in the scripts are consistent with it.                                                                                                                        |

## Usage

```bash
conda activate meta_finger
cd /root/works/MMF-1029-simple

# Smoke test for the full pipeline
# Each stage uses very small N/iter values to quickly verify the workflow.
MMF_BENCH_GPU=1 PYTHONPATH=. python -m overhead_bench.run_all --smoke

# Formal measurement
# Run this after the GPU becomes idle.
MMF_BENCH_GPU=1 PYTHONPATH=. python -m overhead_bench.run_all

# Run a single stage:
MMF_BENCH_GPU=1 PYTHONPATH=. python -m overhead_bench.d_inference --N-sweep 5 10 30 95

# Skip selected stages, e.g., if A/B have already been run:
MMF_BENCH_GPU=1 PYTHONPATH=. python -m overhead_bench.run_all --skip a b
```
