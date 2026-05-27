"""Phase A: MMF static complexity (params + FLOPs + checkpoint size + big-O).

Usage::
    python overhead_bench/a_static_complexity.py [--smoke]

Outputs ``overhead_bench/results/a_static_complexity.json`` containing
- A1. Per-submodule parameter counts.
- A2. Per-submodule FLOPs/MACs for a sweep of ``N`` with analytic N-formulas.
- A3. Checkpoint size on disk in fp32 and fp16.
- A4. Big-O formula string.
"""

from __future__ import annotations

import argparse
import os
import sys

# Must come before torch import (cascades through the package).
from overhead_bench.bench_utils import set_visible_gpu  # noqa: E402

set_visible_gpu(os.environ.get("MMF_BENCH_GPU", "1"))

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "models")))

from models.feature_extractors import (  # noqa: E402
    DFFeatureExtractor,
    EnhancedMetaLearnet,
    EnhancedMultiMetaFingerNet,
)
from models.classification_head_enhanced import EnhancedClassificationHead  # noqa: E402
from models.dynamic_conv1d import FeatureReweightingModule  # noqa: E402

from overhead_bench.bench_utils import (  # noqa: E402
    analytic_classifier_macs,
    analytic_crossclass_macs,
    analytic_reweighting_macs,
    analytic_topm_macs,
    count_params,
    dump_json,
    file_size_mb,
    get_gpu_info,
    pretty_print,
    profile_macs,
    result_path,
    artifact_path,
)


# Model shape constants (matching current MMF configs).
L_Q = 20_000        # query sequence length
L_S = 10_000        # support sequence length
L_PRIME = 80        # backbone output seq_len at L_Q=20000
C_MAIN = 256        # backbone final channel dim
N_HEADS = 8
TOPM_LAYERS = 2     # EnhancedMultiMetaFingerNet passes num_topm_layers=2 at feature_extractors.py:390
CROSS_LAYERS = 2    # num_cross_layers=2 at feature_extractors.py:391
K_SHOT_STATIC = 1   # shot count used when measuring meta_learnet FLOPs (batch=1 for scaling)


def _build_submodules(num_classes: int) -> dict:
    """Return a dict of independent submodules sized for ``num_classes``."""
    feature_extractor = DFFeatureExtractor(dropout=0.15, use_se=True)
    meta_learnet = EnhancedMetaLearnet(in_channels=2, out_channels=C_MAIN, dropout=0.15)
    feature_reweighting = FeatureReweightingModule(feature_dim=C_MAIN, kernel_size=1)
    classification_head = EnhancedClassificationHead(
        feature_dim=C_MAIN,
        num_classes=num_classes,
        seq_len=L_PRIME,
        num_topm_layers=TOPM_LAYERS,
        num_cross_layers=CROSS_LAYERS,
    )
    return {
        "feature_extractor": feature_extractor,
        "meta_learnet": meta_learnet,
        "feature_reweighting": feature_reweighting,
        "classification_head": classification_head,
    }


def a1_params(num_classes: int) -> dict:
    subs = _build_submodules(num_classes)
    params = {name: count_params(m) for name, m in subs.items()}
    params["total"] = sum(params.values())
    # Also report ``EnhancedMultiMetaFingerNet`` as a sanity check.
    full = EnhancedMultiMetaFingerNet(num_classes=num_classes, dropout=0.15, use_se_in_df=True)
    params["__full_net_check"] = count_params(full)
    return params


def a2_flops_one(num_classes: int) -> dict:
    """FLOPs/MACs breakdown for a single batch=1 end-to-end forward pass."""
    subs = _build_submodules(num_classes)

    # -- feature_extractor: backbone on (1, L_Q) --
    fe = subs["feature_extractor"].eval()
    x = torch.randn(1, L_Q)
    fe_macs = profile_macs(fe, (x,)) or {}

    # -- meta_learnet: support branch on (N, K=1, 2, L_S) --
    ml = subs["meta_learnet"].eval()
    sup = torch.randn(num_classes, K_SHOT_STATIC, 2, L_S)
    ml_macs = profile_macs(ml, (sup,)) or {}

    # -- feature_reweighting: dynamic conv (analytic; thop can't handle) --
    rw_macs_analytic = analytic_reweighting_macs(num_classes, C_MAIN, L_PRIME)

    # -- classification_head: analytic (thop mis-counts the per-class loop) --
    topm_macs_analytic = analytic_topm_macs(
        num_classes=num_classes,
        seq_len=L_PRIME,
        channels=C_MAIN,
        num_heads=N_HEADS,
        num_layers=TOPM_LAYERS,
    )
    cross_macs_analytic = analytic_crossclass_macs(
        num_classes=num_classes,
        channels=C_MAIN,
        num_layers=CROSS_LAYERS,
    )
    cls_mlp_analytic = analytic_classifier_macs(num_classes, C_MAIN)

    head_macs_total = topm_macs_analytic + cross_macs_analytic + cls_mlp_analytic

    total = (
        (fe_macs.get("macs", 0) or 0)
        + (ml_macs.get("macs", 0) or 0)
        + rw_macs_analytic
        + head_macs_total
    )

    return {
        "num_classes": num_classes,
        "batch_size": 1,
        "K_shot": K_SHOT_STATIC,
        "L_query": L_Q,
        "L_support": L_S,
        "L_prime_backbone_out": L_PRIME,
        "feature_extractor_macs_thop": fe_macs,
        "meta_learnet_macs_thop": ml_macs,
        "feature_reweighting_macs_analytic": rw_macs_analytic,
        "classification_head_macs_analytic": {
            "topm_self_attn": topm_macs_analytic,
            "cross_class_attn": cross_macs_analytic,
            "mlp_classifier": cls_mlp_analytic,
            "head_total": head_macs_total,
        },
        "total_macs_estimate": total,
        "total_gflops_2x_macs": round(2 * total / 1e9, 3),
    }


def a2_scaling_table(N_values) -> list:
    """Collect FLOPs for a sweep of ``N`` values to show scaling."""
    table = []
    for n in N_values:
        row = a2_flops_one(n)
        table.append(row)
    return table


def a3_checkpoint_size(num_classes: int) -> dict:
    net = EnhancedMultiMetaFingerNet(num_classes=num_classes, dropout=0.15, use_se_in_df=True)

    fp32_path = artifact_path(f"ckpt_fp32_N{num_classes}.pth")
    fp16_path = artifact_path(f"ckpt_fp16_N{num_classes}.pth")

    torch.save({"model_state_dict": net.state_dict()}, fp32_path)

    fp16_state = {k: v.half() if v.dtype == torch.float32 else v for k, v in net.state_dict().items()}
    torch.save({"model_state_dict": fp16_state}, fp16_path)

    fp32_mb = file_size_mb(fp32_path)
    fp16_mb = file_size_mb(fp16_path)

    # Reweighting-features bank size (ancillary asset at inference):
    reweighting_bank_bytes_fp32 = num_classes * C_MAIN * 4
    reweighting_bank_bytes_fp16 = num_classes * C_MAIN * 2

    return {
        "num_classes": num_classes,
        "fp32_path": fp32_path,
        "fp32_MiB": fp32_mb,
        "fp16_path": fp16_path,
        "fp16_MiB": fp16_mb,
        "reweighting_bank_MiB_fp32": round(reweighting_bank_bytes_fp32 / 1024**2, 4),
        "reweighting_bank_MiB_fp16": round(reweighting_bank_bytes_fp16 / 1024**2, 4),
    }


A4_BIG_O = (
    "Per-query MACs ≈ "
    "Θ(L·C0·k) + N·K·Θ(L_s·C0·k) + Θ(N·C·L')   # dynamic reweighting (1×1 depthwise) "
    "+ Θ(N·T·(L'·C^2 + L'^2·C))                 # Top-m self-attn, T=num_topm_layers "
    "+ Θ(X·(N·C^2 + N^2·C))                     # Cross-class MHSA, X=num_cross_layers "
    "+ Θ(N·C^2).                                 # Per-class MLP\n"
    "Dominant scaling terms: (i) Top-m self-attn is the main linear-in-N term, "
    "(ii) Cross-class MHSA is the only quadratic-in-N term. When W_c is cached, "
    "the meta_learnet term (N·K·Θ(L_s·C0·k)) drops to zero at inference."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="small N sweep for quick verification")
    parser.add_argument("--full-n", type=int, nargs="*", default=None,
                        help="override the full N sweep list")
    parser.add_argument("--ckpt-n", type=int, default=90,
                        help="num_classes to use for the fp32/fp16 checkpoint asset")
    args = parser.parse_args()

    if args.smoke:
        n_values = [5, 30]
        ckpt_n = 30
    else:
        n_values = args.full_n or [5, 10, 30, 60, 90, 95]
        ckpt_n = args.ckpt_n

    gpu_info = get_gpu_info()

    # A1 ran once at the same N as the first entry in n_values; meta_learnet
    # and feature_extractor and feature_reweighting params are N-invariant.
    # classification_head DOES grow with N (position embedding is N-invariant,
    # but cross_class_attn weights are N-invariant too; only the per-class
    # output bias grows, which we surface via the sweep in A2's head macs).
    params_table = {}
    for n in n_values:
        params_table[n] = a1_params(n)

    a2_table = a2_scaling_table(n_values)
    a3_info = a3_checkpoint_size(ckpt_n)

    payload = {
        "phase": "A_static_complexity",
        "smoke": args.smoke,
        "gpu_info": gpu_info,
        "A1_params_by_N": params_table,
        "A2_flops_by_N": a2_table,
        "A3_checkpoint_size": a3_info,
        "A4_big_O_formula": A4_BIG_O,
        "constants": {
            "L_query": L_Q,
            "L_support": L_S,
            "L_prime": L_PRIME,
            "C_main": C_MAIN,
            "num_heads": N_HEADS,
            "num_topm_layers": TOPM_LAYERS,
            "num_cross_layers": CROSS_LAYERS,
            "K_shot_static_measurement": K_SHOT_STATIC,
        },
    }

    out_path = result_path("a_static_complexity.json")
    dump_json(payload, out_path)
    pretty_print(
        {
            "saved_to": out_path,
            "gpu": gpu_info.get("name", "<no-gpu>"),
            "ckpt_fp32_MiB": a3_info["fp32_MiB"],
            "ckpt_fp16_MiB": a3_info["fp16_MiB"],
            "N_sweep": n_values,
        },
        title="Phase A done",
    )


if __name__ == "__main__":
    main()
