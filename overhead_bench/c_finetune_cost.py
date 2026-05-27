"""Phase C: Fine-tuning cost when onboarding 30 novel monitored classes.

Measures one fine-tuning epoch under the following constraints:
    - ``batch_size = 1``
    - ``repeat = 1`` (no data duplication)
    - ``k_shot`` configurable (default 20 to match the paper)
    - 30 novel classes appended to the base 60, so ``num_classes = 90``
    - Synthetic random data (no I/O from disk), because we are only measuring
      the compute/memory footprint, not accuracy.

After fine-tuning, we:
    1. Cache ``W_c`` for every class by a single support-branch pass.
    2. Persist:
       - full_ckpt.pth  - standard checkpoint (model + optimizer + scheduler).
       - lean_ckpt.pth  - backbone + classification_head only (no meta_learnet,
                          no feature_reweighting).
       - class_bank.pt  - (N, 256) tensor of cached reweighting features.
    3. Measure the sizes.
    4. Measure the ``C3`` "onboard one new monitored class" cost: a single
       support-branch forward with ``shots_per_class`` samples of a novel class.

Outputs ``overhead_bench/results/c_finetune_cost.json``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from overhead_bench.bench_utils import set_visible_gpu  # noqa: E402

set_visible_gpu(os.environ.get("MMF_BENCH_GPU", "1"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "models")))

from models.feature_extractors import EnhancedMultiMetaFingerNet  # noqa: E402

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
    tensor_size_mb,
    time_cuda,
)
from overhead_bench.cached_inference_model import (  # noqa: E402
    build_lean_state_dict,
    compute_class_bank,
)


def synth_finetune_batch(
    num_classes: int, k_shot: int, L_q: int, L_s: int, device: torch.device
):
    q = torch.randn(1, L_q, device=device)
    s = torch.randn(num_classes, k_shot, L_s, device=device)
    m = torch.ones(num_classes, k_shot, L_s, device=device)
    # One positive class per sample (random).
    lab = torch.zeros(1, num_classes, device=device)
    lab[0, torch.randint(0, num_classes, (1,))] = 1.0
    return q, s, m, lab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--base-classes", type=int, default=60)
    parser.add_argument("--novel-classes", type=int, default=30)
    parser.add_argument("--k-shot", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--L-query", type=int, default=20_000)
    parser.add_argument("--L-support", type=int, default=10_000)
    parser.add_argument("--num-iters-per-epoch", type=int, default=50,
                        help="synthetic iteration count to emulate one finetune epoch wall-clock")
    parser.add_argument("--bank-size", type=int, default=95,
                        help="number of W_c vectors to cache (>= num_classes). Extra slots represent "
                             "future classes that can be onboarded without retraining.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--n-iter-step", type=int, default=5,
                        help="iterations used to measure the per-step time (separate from epoch timing)")
    args = parser.parse_args()

    if args.smoke:
        args.base_classes = 5
        args.novel_classes = 3
        args.k_shot = 2
        args.num_iters_per_epoch = 3
        args.n_iter_step = 2
        args.bank_size = 8

    assert torch.cuda.is_available(), "Phase C requires CUDA"
    device = torch.device("cuda")

    num_classes = args.base_classes + args.novel_classes
    cfg = dict(
        batch_size=args.batch_size,
        k_shot=args.k_shot,
        num_classes=num_classes,
        base_classes=args.base_classes,
        novel_classes=args.novel_classes,
        L_query=args.L_query,
        L_support=args.L_support,
        num_iters_per_epoch=args.num_iters_per_epoch,
        optimizer="Adam",
        lr=args.lr,
        weight_decay=1e-4,
        loss="BCEWithLogitsLoss",
        finetune_mode="full",
    )

    gpu_info = get_gpu_info()
    cpu_before = cpu_mem_mib()

    # -------------------- Build model --------------------
    model = EnhancedMultiMetaFingerNet(
        num_classes=num_classes,
        dropout=0.15,
        support_blocks=0,
        use_se_in_df=True,
    ).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    # -------------------- Per-step timing --------------------
    q, s, m, y = synth_finetune_batch(num_classes, args.k_shot, args.L_query, args.L_support, device)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        out = model(q, s, m)
        loss = criterion(out["logits"], y)
        loss.backward()
        optimizer.step()

    reset_peak_gpu_mem()
    per_step_timing = time_cuda(train_step, n_warmup=args.n_warmup, n_iter=args.n_iter_step)
    peak_step = peak_gpu_mem_mib()

    # -------------------- Full-epoch wall-clock --------------------
    # Separately for pure compute (no data loading overhead because tensors are resident).
    reset_peak_gpu_mem()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.num_iters_per_epoch):
        train_step()
    torch.cuda.synchronize()
    epoch_seconds = time.perf_counter() - t0
    peak_epoch = peak_gpu_mem_mib()
    cpu_after_epoch = cpu_mem_mib()

    # -------------------- C3: onboard one new class --------------------
    reset_peak_gpu_mem()
    model.eval()
    single_support = torch.randn(1, args.k_shot, args.L_support, device=device)
    single_mask = torch.ones_like(single_support)

    def onboard_one():
        _ = model.support_forward(single_support, single_mask)

    onboard_timing = time_cuda(onboard_one, n_warmup=args.n_warmup, n_iter=args.n_iter_step)
    peak_onboard = peak_gpu_mem_mib()

    # -------------------- Cache class bank for all N classes --------------------
    # Cache at least ``bank_size`` W_c vectors. ``bank_size`` can exceed
    # ``num_classes`` to represent future classes to be onboarded lazily; this
    # only affects the class_bank size and has no effect on model weights.
    bank_size = max(args.bank_size, num_classes)
    reset_peak_gpu_mem()
    all_support = torch.randn(bank_size, args.k_shot, args.L_support, device=device)
    all_mask = torch.ones_like(all_support)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    class_bank = compute_class_bank(model, all_support, all_mask).detach().cpu()
    torch.cuda.synchronize()
    bank_compute_seconds = time.perf_counter() - t0
    peak_bank = peak_gpu_mem_mib()
    bank_tensor_MiB = tensor_size_mb(class_bank)

    # -------------------- Save checkpoints --------------------
    full_ckpt_path = artifact_path("c_full_ckpt.pth")
    lean_ckpt_path = artifact_path("c_lean_ckpt.pth")
    bank_path = artifact_path("c_class_bank.pt")

    # Full checkpoint: model + optimizer + scheduler (matches train_enhanced.py style).
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": cfg,
        },
        full_ckpt_path,
    )

    lean_sd = build_lean_state_dict(model)
    torch.save({"model_state_dict": lean_sd, "config": cfg}, lean_ckpt_path)
    torch.save(class_bank, bank_path)

    full_MiB = file_size_mb(full_ckpt_path)
    lean_MiB = file_size_mb(lean_ckpt_path)
    bank_MiB = file_size_mb(bank_path)

    params_total = count_params(model)
    params_lean = sum(v.numel() for v in lean_sd.values())

    payload = {
        "phase": "C_finetune_cost",
        "smoke": args.smoke,
        "gpu_info": gpu_info,
        "cpu_mem_before": cpu_before,
        "cpu_mem_after_epoch": cpu_after_epoch,
        "config": cfg,
        "per_train_step": {
            "timing_ms": per_step_timing.to_dict(),
            "peak_gpu_mem": peak_step,
        },
        "one_epoch_simulated": {
            "num_iters": args.num_iters_per_epoch,
            "wall_clock_seconds": round(epoch_seconds, 3),
            "per_iter_seconds_avg": round(epoch_seconds / args.num_iters_per_epoch, 4),
            "peak_gpu_mem": peak_epoch,
        },
        "C3_onboard_one_new_class": {
            "description": "single support-branch forward on one novel class's K-shot samples",
            "k_shot": args.k_shot,
            "timing_ms": onboard_timing.to_dict(),
            "peak_gpu_mem": peak_onboard,
        },
        "class_bank_cache": {
            "num_classes_trained": num_classes,
            "bank_size": bank_size,
            "bank_shape": list(class_bank.shape),
            "compute_seconds_all_classes": round(bank_compute_seconds, 3),
            "peak_gpu_mem_during_compute": peak_bank,
            "in_memory_MiB": bank_tensor_MiB,
            "on_disk_MiB": bank_MiB,
        },
        "storage": {
            "full_ckpt_path": full_ckpt_path,
            "full_ckpt_MiB": full_MiB,
            "lean_ckpt_path": lean_ckpt_path,
            "lean_ckpt_MiB": lean_MiB,
            "class_bank_path": bank_path,
            "class_bank_MiB": bank_MiB,
            "lean_deployment_MiB_total": round(lean_MiB + bank_MiB, 4),
            "params_total": params_total,
            "params_lean": params_lean,
            "params_stripped_ratio": round(1 - params_lean / params_total, 4),
        },
    }

    out_path = result_path("c_finetune_cost.json")
    dump_json(payload, out_path)

    pretty_print(
        {
            "saved_to": out_path,
            "epoch_seconds": round(epoch_seconds, 3),
            "per_iter_ms_mean": per_step_timing.mean_ms,
            "onboard_one_class_ms_mean": onboard_timing.mean_ms,
            "full_ckpt_MiB": full_MiB,
            "lean_ckpt_MiB": lean_MiB,
            "class_bank_MiB": bank_MiB,
            "N": num_classes,
        },
        title="Phase C done",
    )


if __name__ == "__main__":
    main()
