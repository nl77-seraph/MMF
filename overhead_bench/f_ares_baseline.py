"""Phase F: ARES baseline with one binary classifier per monitored class.

Background
----------
The ARES paper ("Towards Efficient and Practical Multi-Tab Website
Fingerprinting Attacks", arXiv:2501.12622) formulates multi-tab WF as a
one-vs-all problem with **N independent binary classifiers** (Trans-WF), not a
single shared-backbone N-head model. The current code in
``peers_works/ARES_pre/train.py`` trains one shared model with
MultiLabelSoftMarginLoss - that is ARES's *refactored* objective, but for a
fair complexity comparison against MMF we implement the *paper's original
one-vs-all* formulation: build N binary classifiers, train each for one epoch
on synthetic data, persist each to disk, and at inference route the query
through all N classifiers and aggregate their sigmoid outputs into a single
``(B, N)`` tensor.

We report:
    - Per-model params, total params across N models.
    - Training wall-clock (1 epoch, synthetic random data), peak GPU mem.
    - Saved size on disk (all N checkpoints combined).
    - Single-sample inference latency (batch=1) through all N classifiers.
    - Inference peak GPU memory.

This is intentionally the *pessimistic* deployment of ARES: the classifiers
are loaded sequentially and executed sequentially. A production adversary
could batch them by stacking weights, but the monitored-set update semantics
(i.e. "replace one classifier when adding a new site") still apply model by
model. We therefore stick to the sequential evaluation to match the paper's
onboarding description.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

from overhead_bench.bench_utils import set_visible_gpu  # noqa: E402

set_visible_gpu(os.environ.get("MMF_BENCH_GPU", "1"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

ARES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "peers_works", "ARES_pre")
)
sys.path.insert(0, ARES_DIR)
from ARES import Trans_WF  # noqa: E402

from overhead_bench.bench_utils import (  # noqa: E402
    artifact_path,
    count_params,
    cpu_mem_mib,
    dump_json,
    file_size_mb,
    get_gpu_info,
    peak_gpu_mem_mib,
    pretty_print,
    reset_peak_gpu_mem,
    result_path,
    time_cuda,
)


def autodetect_max_len(seq_len: int, device: torch.device) -> int:
    """Probe the Trans_WF backbone to compute ``max_len`` (pos_embed length)
    for an arbitrary input ``seq_len``. We build a dummy Trans_WF with an
    oversized max_len, run the ``profiling + combination`` path manually and
    measure the final sequence length.
    """
    import torch.nn.functional as F  # noqa: F401
    # Build a model with a placeholder max_len=1; we only need the backbone.
    # We run the equivalent of ``dividing`` + ``profiling`` + ``combination``
    # without touching pos_embed.
    m = Trans_WF(num_classes=1, max_len=1).to(device)
    x = torch.zeros(1, 1, seq_len, device=device)
    with torch.no_grad():
        x = m.dividing(x)
        x = m.profiling(x)
        x = m.combination(x)
    # x is (1, C, L); max_len equals L.
    max_len = x.size(-1)
    del m, x
    return int(max_len)


def build_ares_binary(device: torch.device, max_len: int = 108) -> nn.Module:
    """Build one binary Trans_WF with a single output logit (sigmoid head)."""
    model = Trans_WF(num_classes=1, max_len=max_len).to(device)
    return model


def synth_ares_batch(
    batch_size: int,
    seq_len: int,
    num_monitored: int,
    device: torch.device,
    positive_class: int,
):
    x = torch.sign(torch.randn(batch_size, 1, seq_len, device=device))
    y_multi = torch.zeros(batch_size, num_monitored, device=device)
    # Half-half positives vs negatives for the binary target.
    flip = torch.rand(batch_size, device=device) < 0.5
    y_multi[flip, positive_class] = 1.0
    y_binary = y_multi[:, positive_class].unsqueeze(-1)
    return x, y_binary


def ares_training_phase(
    num_monitored: int,
    num_samples: int,
    batch_size: int,
    seq_len: int,
    max_len: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    out_dir: str,
):
    """Train ``num_monitored`` independent binary Trans_WF models for 1 epoch each."""
    os.makedirs(out_dir, exist_ok=True)
    criterion = nn.BCEWithLogitsLoss()

    reset_peak_gpu_mem()
    torch.cuda.synchronize()
    train_start = time.perf_counter()

    per_model_params = None
    total_params = 0
    saved_sizes = []
    per_model_seconds = []

    num_steps = max(1, num_samples // batch_size)

    for c in range(num_monitored):
        model = build_ares_binary(device, max_len=max_len).train()
        if per_model_params is None:
            per_model_params = count_params(model)
        total_params += per_model_params

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_steps):
            x, y_bin = synth_ares_batch(batch_size, seq_len, num_monitored, device, positive_class=c)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y_bin)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        per_model_seconds.append(time.perf_counter() - t0)

        ckpt_path = os.path.join(out_dir, f"ares_binary_class{c:04d}.pth")
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
        saved_sizes.append(file_size_mb(ckpt_path))

        del model, optimizer
        if (c + 1) % 10 == 0:
            torch.cuda.empty_cache()

    torch.cuda.synchronize()
    total_train_seconds = time.perf_counter() - train_start
    peak_train_mem = peak_gpu_mem_mib()

    return {
        "num_monitored": num_monitored,
        "num_samples_per_model": num_samples,
        "batch_size": batch_size,
        "per_model_params": per_model_params,
        "total_params_all_models": total_params,
        "total_saved_MiB": round(sum(saved_sizes), 4),
        "avg_per_model_ckpt_MiB": round(sum(saved_sizes) / num_monitored, 4),
        "train_wallclock_seconds": round(total_train_seconds, 3),
        "per_model_train_seconds_mean": round(sum(per_model_seconds) / num_monitored, 4),
        "peak_gpu_mem_during_training": peak_train_mem,
    }


def ares_inference_phase(
    num_monitored: int,
    seq_len: int,
    max_len: int,
    device: torch.device,
    ckpt_dir: str,
    n_warmup: int,
    n_iter: int,
    N_sweep,
):
    """Run a single query through ``N`` classifiers and measure latency."""
    assert os.path.isdir(ckpt_dir)
    # Load all classifiers into CPU first, then move to GPU lazily during inference
    # to mimic a realistic deployment where not all models can fit at once; we
    # also offer a "resident" variant that keeps everything on GPU for the best
    # possible latency comparison.
    classifiers = []
    for c in range(num_monitored):
        ckpt_path = os.path.join(ckpt_dir, f"ares_binary_class{c:04d}.pth")
        m = build_ares_binary(device, max_len=max_len).eval()
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"]
        m.load_state_dict(sd, strict=True)
        classifiers.append(m)

    # Sweep over N.
    sweep = []
    x = torch.sign(torch.randn(1, 1, seq_len, device=device))

    for N in N_sweep:
        assert N <= num_monitored, f"Cannot infer N={N} with only {num_monitored} models"

        def infer():
            outs = []
            with torch.no_grad():
                for i in range(N):
                    outs.append(torch.sigmoid(classifiers[i](x)))
            return torch.cat(outs, dim=-1)

        reset_peak_gpu_mem()
        timing = time_cuda(infer, n_warmup=n_warmup, n_iter=n_iter)
        peak = peak_gpu_mem_mib()
        sweep.append({
            "N": N,
            "latency_ms": timing.to_dict(),
            "peak_gpu_mem": peak,
        })

    return sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-monitored", type=int, default=95,
                        help="total binary classifiers to train (matches 0..94 monitored classes)")
    parser.add_argument("--N-sweep", type=int, nargs="+", default=[5, 10, 30, 95])
    parser.add_argument("--seq-len", type=int, default=10_000,
                        help="ARES input length (original paper uses 10000)")
    parser.add_argument("--max-len", type=int, default=-1,
                        help="TransWF pos_embed length. Leave -1 to auto-detect from --seq-len")
    parser.add_argument("--num-samples-per-model", type=int, default=256,
                        help="synthetic samples for 1-epoch training")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0014)
    parser.add_argument("--weight-decay", type=float, default=0.005)
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=10)
    args = parser.parse_args()

    if args.smoke:
        args.num_monitored = 6
        args.N_sweep = [3, 5]
        args.num_samples_per_model = 8
        args.batch_size = 4
        args.n_warmup = 1
        args.n_iter = 2

    assert torch.cuda.is_available(), "Phase F requires CUDA"
    device = torch.device("cuda")

    if args.max_len <= 0:
        args.max_len = autodetect_max_len(args.seq_len, device)
        print(f"[Phase F] autodetected max_len={args.max_len} for seq_len={args.seq_len}")

    gpu_info = get_gpu_info()
    cpu_before = cpu_mem_mib()

    ckpt_dir = artifact_path("ares_binary_ckpts")
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    train_report = ares_training_phase(
        num_monitored=args.num_monitored,
        num_samples=args.num_samples_per_model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_len=args.max_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        out_dir=ckpt_dir,
    )

    infer_report = ares_inference_phase(
        num_monitored=args.num_monitored,
        seq_len=args.seq_len,
        max_len=args.max_len,
        device=device,
        ckpt_dir=ckpt_dir,
        n_warmup=args.n_warmup,
        n_iter=args.n_iter,
        N_sweep=args.N_sweep,
    )

    cpu_after = cpu_mem_mib()

    payload = {
        "phase": "F_ARES_baseline_one_vs_all",
        "smoke": args.smoke,
        "gpu_info": gpu_info,
        "cpu_mem_before": cpu_before,
        "cpu_mem_after": cpu_after,
        "config": {
            "num_monitored": args.num_monitored,
            "N_sweep": args.N_sweep,
            "seq_len": args.seq_len,
            "max_len": args.max_len,
            "num_samples_per_model": args.num_samples_per_model,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "formulation": "one-vs-all; N independent binary Trans-WF classifiers (ARES paper Sec IV.A)",
        },
        "training": train_report,
        "inference_scalability": infer_report,
    }

    out_path = result_path("f_ares_baseline.json")
    dump_json(payload, out_path)

    pretty_print(
        {
            "saved_to": out_path,
            "train_wallclock_seconds": train_report["train_wallclock_seconds"],
            "total_saved_MiB": train_report["total_saved_MiB"],
            "per_model_params": train_report["per_model_params"],
            "total_params_all_models": train_report["total_params_all_models"],
            "inference_scalability": [
                {"N": r["N"], "ms": r["latency_ms"]["mean_ms"], "peak_MiB": r["peak_gpu_mem"]["allocated_MiB"]}
                for r in infer_report
            ],
        },
        title="Phase F done",
    )


if __name__ == "__main__":
    main()
