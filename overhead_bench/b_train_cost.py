"""Phase B: Initial training cost of MMF at batch=1, shots_per_class=1.

This script measures a single forward+backward+optimizer-step iteration on
synthetic random tensors, which is sufficient to characterise the peak memory
and wall-clock because training cost scales linearly with batch count and
number of iterations (the per-step topology is identical).

Reported fields:
    - GPU type / CUDA capability (single card, controlled by MMF_BENCH_GPU).
    - Params (total + trainable).
    - FLOPs (MACs) for a single forward pass (sanity check against Phase A).
    - Wall-clock per iteration (warm + steady state), averaged over n_iter.
    - Peak GPU memory (allocated + reserved) during one train step.
    - Peak CPU RSS before/after.

The measured configuration intentionally follows the paper's "base training"
setting except for the batch size and shot count override:
    - num_classes = 60 (base classes).
    - L_query = 20_000, L_support = 10_000.
    - batch_size = 1, shots_per_class = 1.
    - Optimizer: Adam lr=5e-5 weight_decay=1e-4 (matching base_train config).
    - Loss: WeightedBCELoss(pos_weight=1.0) (simplified; irrelevant to cost).
"""

from __future__ import annotations

import argparse
import os
import sys

from overhead_bench.bench_utils import set_visible_gpu  # noqa: E402

set_visible_gpu(os.environ.get("MMF_BENCH_GPU", "1"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "models")))

from models.feature_extractors import EnhancedMultiMetaFingerNet  # noqa: E402

from overhead_bench.bench_utils import (  # noqa: E402
    count_params,
    cpu_mem_mib,
    current_gpu_mem_mib,
    dump_json,
    get_gpu_info,
    peak_gpu_mem_mib,
    pretty_print,
    profile_macs,
    reset_peak_gpu_mem,
    result_path,
    time_cuda,
)


def build_random_batch(
    num_classes: int,
    shots_per_class: int,
    L_query: int,
    L_support: int,
    device: torch.device,
    batch_size: int = 1,
):
    query = torch.randn(batch_size, L_query, device=device)
    support = torch.randn(num_classes, shots_per_class, L_support, device=device)
    masks = torch.ones(num_classes, shots_per_class, L_support, device=device)
    # Multi-hot labels: each sample has 1 positive class on average.
    labels = torch.zeros(batch_size, num_classes, device=device)
    for b in range(batch_size):
        labels[b, torch.randint(0, num_classes, (1,))] = 1.0
    return query, support, masks, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="use tiny N and fewer iters")
    parser.add_argument("--num-classes", type=int, default=60, help="number of base classes")
    parser.add_argument("--shots", type=int, default=1, help="shots_per_class")
    parser.add_argument("--batch-size", type=int, default=1, help="train batch size")
    parser.add_argument("--L-query", type=int, default=20_000)
    parser.add_argument("--L-support", type=int, default=10_000)
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    if args.smoke:
        args.num_classes = 5
        args.n_warmup = 1
        args.n_iter = 2

    assert torch.cuda.is_available(), "Phase B requires CUDA"
    device = torch.device("cuda")

    gpu_info = get_gpu_info()
    cpu_before = cpu_mem_mib()

    model = EnhancedMultiMetaFingerNet(
        num_classes=args.num_classes,
        dropout=0.15,
        support_blocks=0,
        use_se_in_df=True,
    ).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # ---- FLOPs sanity check (forward-only) ----
    model.eval()
    q_tmp, s_tmp, m_tmp, _ = build_random_batch(
        args.num_classes, args.shots, args.L_query, args.L_support, device, batch_size=args.batch_size,
    )
    macs_info = profile_macs(model, (q_tmp, s_tmp, m_tmp))
    model.train()
    del q_tmp, s_tmp, m_tmp
    torch.cuda.empty_cache()

    # ---- Build a persistent batch of random tensors (avoids measuring data-gen cost) ----
    query, support, masks, labels = build_random_batch(
        args.num_classes, args.shots, args.L_query, args.L_support, device, batch_size=args.batch_size,
    )

    # ---- One forward+backward+step as the "unit of work" ----
    def train_step():
        optimizer.zero_grad(set_to_none=True)
        out = model(query, support, masks)
        loss = criterion(out["logits"], labels)
        loss.backward()
        optimizer.step()
        return loss

    reset_peak_gpu_mem()
    timing = time_cuda(train_step, n_warmup=args.n_warmup, n_iter=args.n_iter)
    peak = peak_gpu_mem_mib()
    cur = current_gpu_mem_mib()
    cpu_after = cpu_mem_mib()

    params_total = count_params(model)
    params_trainable = count_params(model, trainable_only=True)

    payload = {
        "phase": "B_initial_training_cost",
        "smoke": args.smoke,
        "config": {
            "num_classes": args.num_classes,
            "shots_per_class": args.shots,
            "batch_size": args.batch_size,
            "L_query": args.L_query,
            "L_support": args.L_support,
            "optimizer": "Adam",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "loss": "BCEWithLogitsLoss",
        },
        "gpu_info": gpu_info,
        "cpu_mem_before": cpu_before,
        "cpu_mem_after": cpu_after,
        "params": {
            "total": params_total,
            "trainable": params_trainable,
            "total_MiB_fp32": round(params_total * 4 / 1024**2, 3),
        },
        "forward_macs": macs_info,
        "train_step_timing_ms": timing.to_dict(),
        "train_step_peak_gpu_mem": peak,
        "current_gpu_mem_after_step": cur,
    }

    out_path = result_path("b_train_cost.json")
    dump_json(payload, out_path)
    pretty_print(
        {
            "saved_to": out_path,
            "gpu": gpu_info.get("name"),
            "train_step_mean_ms": timing.mean_ms,
            "peak_alloc_MiB": peak["allocated_MiB"],
            "peak_reserved_MiB": peak["reserved_MiB"],
            "total_params": params_total,
        },
        title="Phase B done",
    )


if __name__ == "__main__":
    main()
