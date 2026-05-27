"""Phase E: Onboarding ONE new monitored class while reusing the cached base bank.

This phase answers the reviewer question:
    "Does fine-tuning support reusing the base-train-saved reweighting features
     for base classes and only computing class-specific features for novel
     classes?"

Answer: **Yes, provided that ``meta_learnet`` (and ``feature_extractor``) are
frozen during fine-tuning**, i.e. ``finetune_mode='head_only'`` or a custom
freeze policy that keeps the W_c generator static. Because the DF backbone and
meta_learnet are the only blocks that depend on ``support_data``, freezing them
makes every class's ``W_c`` independent of training iteration and of every
other class.

We therefore measure two numbers that matter for "maintaining freshness":

    E1.  Memory and wall-clock to compute ``W_c`` for **exactly one** novel
         class given ``K`` support samples (pure support-branch forward). This
         is O(1) w.r.t. the size of the existing monitored set.

    E2.  A "bank growth" simulation: start from a cached base bank of size 60,
         onboard novel classes one by one up to +30, and confirm that the peak
         per-onboarding GPU memory stays flat (i.e. does not scale with N).

Outputs ``overhead_bench/results/e_onboard_one_class.json``.
"""

from __future__ import annotations

import argparse
import os
import sys

from overhead_bench.bench_utils import set_visible_gpu  # noqa: E402

set_visible_gpu(os.environ.get("MMF_BENCH_GPU", "1"))

import torch  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "models")))

from models.feature_extractors import EnhancedMultiMetaFingerNet  # noqa: E402

from overhead_bench.bench_utils import (  # noqa: E402
    cpu_mem_mib,
    dump_json,
    get_gpu_info,
    peak_gpu_mem_mib,
    pretty_print,
    reset_peak_gpu_mem,
    result_path,
    tensor_size_mb,
    time_cuda,
)


def freeze_meta_path(model: EnhancedMultiMetaFingerNet) -> dict:
    """Freeze feature_extractor + meta_learnet + feature_reweighting.

    After this, base W_c vectors are stable across iterations and only the
    head needs fine-tuning. Returns a summary.
    """
    frozen = 0
    total = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if (
            name.startswith("feature_extractor.")
            or name.startswith("meta_learnet.")
            or name.startswith("feature_reweighting.")
        ):
            p.requires_grad = False
            frozen += p.numel()
    return {"frozen_params": frozen, "total_params": total,
            "frozen_ratio": round(frozen / max(total, 1), 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--base-classes", type=int, default=60)
    parser.add_argument("--novel-classes", type=int, default=30)
    parser.add_argument("--k-shot", type=int, default=20)
    parser.add_argument("--L-support", type=int, default=10_000)
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=20)
    args = parser.parse_args()

    if args.smoke:
        args.base_classes = 5
        args.novel_classes = 3
        args.k_shot = 2
        args.n_warmup = 1
        args.n_iter = 3

    assert torch.cuda.is_available(), "Phase E requires CUDA"
    device = torch.device("cuda")

    gpu_info = get_gpu_info()
    cpu_before = cpu_mem_mib()

    # -------------------- Build a trained-like model & freeze meta path --------------------
    base_N = args.base_classes
    model = EnhancedMultiMetaFingerNet(
        num_classes=base_N + args.novel_classes,
        dropout=0.15,
        support_blocks=0,
        use_se_in_df=True,
    ).to(device)
    freeze_info = freeze_meta_path(model)
    model.eval()  # frozen path -> deterministic W_c

    # -------------------- Pre-cache base bank (simulating a stored asset) --------------------
    base_support = torch.randn(base_N, args.k_shot, args.L_support, device=device)
    base_mask = torch.ones_like(base_support)
    with torch.no_grad():
        base_bank = model.support_forward(base_support, base_mask).detach().cpu()
    del base_support, base_mask
    torch.cuda.empty_cache()

    base_bank_MiB = tensor_size_mb(base_bank)

    # -------------------- E1: onboarding a single novel class --------------------
    novel_support = torch.randn(1, args.k_shot, args.L_support, device=device)
    novel_mask = torch.ones_like(novel_support)

    def onboard_one():
        with torch.no_grad():
            _ = model.support_forward(novel_support, novel_mask)

    reset_peak_gpu_mem()
    timing = time_cuda(onboard_one, n_warmup=args.n_warmup, n_iter=args.n_iter)
    peak_one = peak_gpu_mem_mib()

    # -------------------- E2: bank growth from 60 to 60+30 --------------------
    # We onboard one class at a time and monitor peak memory.
    growth_log = []
    current_bank = base_bank.clone()
    for step in range(args.novel_classes):
        reset_peak_gpu_mem()
        novel_sup = torch.randn(1, args.k_shot, args.L_support, device=device)
        novel_msk = torch.ones_like(novel_sup)
        with torch.no_grad():
            wc = model.support_forward(novel_sup, novel_msk).detach().cpu()
        current_bank = torch.cat([current_bank, wc], dim=0)
        peak = peak_gpu_mem_mib()
        growth_log.append({
            "step": step + 1,
            "bank_size_after": current_bank.size(0),
            "peak_gpu_mem": peak,
        })
        del novel_sup, novel_msk, wc

    final_bank_MiB = tensor_size_mb(current_bank)
    cpu_after = cpu_mem_mib()

    payload = {
        "phase": "E_onboard_one_class",
        "smoke": args.smoke,
        "gpu_info": gpu_info,
        "cpu_mem_before": cpu_before,
        "cpu_mem_after": cpu_after,
        "config": {
            "base_classes": base_N,
            "novel_classes": args.novel_classes,
            "k_shot": args.k_shot,
            "L_support": args.L_support,
            "finetune_mode": "head_only (meta path frozen)",
        },
        "freeze_summary": freeze_info,
        "E1_single_class_onboarding": {
            "timing_ms": timing.to_dict(),
            "peak_gpu_mem": peak_one,
            "base_bank_MiB": base_bank_MiB,
            "final_bank_MiB": final_bank_MiB,
            "per_class_storage_bytes_fp32": 256 * 4,
        },
        "E2_bank_growth": growth_log,
        "analysis_note": (
            "When feature_extractor + meta_learnet are frozen (finetune_mode='head_only' "
            "or a custom frozen-backbone policy), base class W_c values are stable "
            "across training. Therefore a base_class_bank.pt computed once at the end "
            "of base training can be reused throughout fine-tuning, and onboarding a new "
            "monitored class only requires one support-branch forward on its K support "
            "samples (memory and time both O(1) w.r.t. existing bank size)."
        ),
    }

    out_path = result_path("e_onboard_one_class.json")
    dump_json(payload, out_path)

    pretty_print(
        {
            "saved_to": out_path,
            "onboard_one_class_ms": timing.mean_ms,
            "peak_MiB_single_onboarding": peak_one["allocated_MiB"],
            "bank_growth_peak_flat": [row["peak_gpu_mem"]["allocated_MiB"] for row in growth_log],
        },
        title="Phase E done",
    )


if __name__ == "__main__":
    main()
