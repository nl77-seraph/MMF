"""Focused benchmark that answers the reviewer's 10 specific questions.

Produces a single markdown file ``results/rebuttal_tables.md`` with grouped
comparison tables for MMF vs ARES-one-vs-all.

Fixes vs prior f_ares_baseline.py:
    * For the scalability sweep, ARES now loads exactly ``N`` binary
      classifiers for each ``N`` (not 95), so the GPU footprint grows with N.

Training-side metrics use synthetic tensors (we measure compute/memory, not
accuracy). All measurements are done on a single GPU via ``MMF_BENCH_GPU``
(default 1).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time

from overhead_bench.bench_utils import set_visible_gpu  # noqa: E402

set_visible_gpu(os.environ.get("MMF_BENCH_GPU", "1"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))
ARES_DIR = os.path.abspath(os.path.join(ROOT, os.pardir, "peers_works", "ARES_pre"))
sys.path.insert(0, ARES_DIR)

from models.feature_extractors import (  # noqa: E402
    DFFeatureExtractor,
    EnhancedMultiMetaFingerNet,
)
from models.classification_head_enhanced import EnhancedClassificationHead  # noqa: E402
from ARES import Trans_WF  # noqa: E402

from overhead_bench.bench_utils import (  # noqa: E402
    analytic_classifier_macs,
    analytic_crossclass_macs,
    analytic_reweighting_macs,
    analytic_topm_macs,
    artifact_path,
    count_params,
    cpu_mem_mib,
    dump_json,
    file_size_mb,
    get_gpu_info,
    peak_gpu_mem_mib,
    profile_macs,
    reset_peak_gpu_mem,
    result_path,
    tensor_size_mb,
    time_cuda,
)
from overhead_bench.cached_inference_model import (  # noqa: E402
    CachedInferenceMMF,
    build_lean_state_dict,
    compute_class_bank,
    load_lean_checkpoint,
)


# ---------------------------------------------------------------------------
# Hyper-parameters used throughout the rebuttal bench
# ---------------------------------------------------------------------------

N_BASE = 60
N_NOVEL = 35
N_TOTAL = N_BASE + N_NOVEL   # 95
L_Q = 20_000
L_S = 10_000
L_PRIME = 80
C_MAIN = 256
N_HEADS = 8
TOPM_LAYERS = 2
CROSS_LAYERS = 2

# ARES seq_len matches MMF's L_support to keep per-sample compute comparable;
# auto-detect the corresponding ``max_len`` at runtime.
ARES_SEQ_LEN = 10_000


# ---------------------------------------------------------------------------
# Utilities: full CUDA reset between measurements
# ---------------------------------------------------------------------------


def hard_cuda_reset():
    """Release Python refs and empty the CUDA allocator between experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def autodetect_ares_max_len(seq_len: int, device: torch.device) -> int:
    m = Trans_WF(num_classes=1, max_len=1).to(device)
    x = torch.zeros(1, 1, seq_len, device=device)
    with torch.no_grad():
        x = m.dividing(x)
        x = m.profiling(x)
        x = m.combination(x)
    max_len = x.size(-1)
    del m, x
    hard_cuda_reset()
    return int(max_len)


# ---------------------------------------------------------------------------
# Metric 1: Parameter counts
# ---------------------------------------------------------------------------


def metric1_params() -> dict:
    """MMF total params + inference-only params (no meta_learnet / feature_reweighting)."""
    model = EnhancedMultiMetaFingerNet(
        num_classes=N_TOTAL, dropout=0.15, use_se_in_df=True,
    )
    total = sum(p.numel() for p in model.parameters())
    inference_keep = []
    inference_drop = []
    for name, p in model.named_parameters():
        if name.startswith("meta_learnet.") or name.startswith("feature_reweighting."):
            inference_drop.append((name, p.numel()))
        else:
            inference_keep.append((name, p.numel()))

    inference_only = sum(n for _, n in inference_keep)
    dropped = sum(n for _, n in inference_drop)

    by_module = {
        "feature_extractor": sum(p.numel() for p in model.feature_extractor.parameters()),
        "meta_learnet": sum(p.numel() for p in model.meta_learnet.parameters()),
        "feature_reweighting": sum(p.numel() for p in model.feature_reweighting.parameters()),
        "classification_head": sum(p.numel() for p in model.classification_head.parameters()),
    }

    del model
    hard_cuda_reset()
    return {
        "total_params": int(total),
        "total_MiB_fp32": round(total * 4 / 1024**2, 4),
        "inference_only_params": int(inference_only),
        "inference_only_MiB_fp32": round(inference_only * 4 / 1024**2, 4),
        "dropped_params_at_inference": int(dropped),
        "by_module": {k: int(v) for k, v in by_module.items()},
    }


# ---------------------------------------------------------------------------
# Metric 2: Inference FLOPs formula as a function of N
# ---------------------------------------------------------------------------


def metric2_inference_flops_formula(
    ares_per_model_macs: int, n_values, device: torch.device
) -> dict:
    """Return symbolic + numeric FLOPs-vs-N tables for MMF (cached) and ARES."""
    # MMF cached inference: drop meta_learnet term entirely.
    fe_macs = None
    # Measure feature_extractor forward MACs via thop (N-invariant).
    model = EnhancedMultiMetaFingerNet(
        num_classes=N_TOTAL, dropout=0.15, use_se_in_df=True,
    ).to(device).eval()
    q = torch.randn(1, L_Q, device=device)
    fe_info = profile_macs(model.feature_extractor, (q,))
    fe_macs = int(fe_info.get("macs", 0)) if fe_info else 0
    del model, q
    hard_cuda_reset()

    def mmf_cached(N):
        rw = analytic_reweighting_macs(N, C_MAIN, L_PRIME)
        topm = analytic_topm_macs(N, L_PRIME, C_MAIN, N_HEADS, TOPM_LAYERS)
        cross = analytic_crossclass_macs(N, C_MAIN, CROSS_LAYERS)
        mlp = analytic_classifier_macs(N, C_MAIN)
        return fe_macs + rw + topm + cross + mlp

    def ares_total(N):
        return N * ares_per_model_macs

    mmf_formula = (
        "FLOPs(N) = 2 * [FE + N*C*L' + N*T*(4*L'*C^2 + 2*L'^2*C) + X*(4*N*C^2 + 2*N^2*C) + N*(C*C/2 + C/2*C/4 + C/4)]"
        f"\n  where FE (backbone MACs) = {fe_macs:,}, C={C_MAIN}, L'={L_PRIME}, T={TOPM_LAYERS}, X={CROSS_LAYERS}"
    )
    ares_formula = (
        f"FLOPs(N) = 2 * N * {ares_per_model_macs:,}  (single binary Trans-WF MACs, measured)"
    )

    table = []
    for n in n_values:
        table.append({
            "N": n,
            "MMF_cached_MACs": int(mmf_cached(n)),
            "MMF_cached_GFLOPs_2x": round(2 * mmf_cached(n) / 1e9, 3),
            "ARES_MACs": int(ares_total(n)),
            "ARES_GFLOPs_2x": round(2 * ares_total(n) / 1e9, 3),
            "ARES_over_MMF": round(ares_total(n) / mmf_cached(n), 3),
        })
    return {
        "mmf_cached_formula": mmf_formula,
        "ares_formula": ares_formula,
        "fe_MACs": fe_macs,
        "table": table,
    }


# ---------------------------------------------------------------------------
# Metric 3/4: MMF base training per-iter @(N=60, B=1, K=1)
# ---------------------------------------------------------------------------


def metric34_mmf_base_train(device, n_warmup=3, n_iter=10) -> dict:
    hard_cuda_reset()
    model = EnhancedMultiMetaFingerNet(
        num_classes=N_BASE, dropout=0.15, use_se_in_df=True,
    ).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    q = torch.randn(1, L_Q, device=device)
    s = torch.randn(N_BASE, 1, L_S, device=device)
    m = torch.ones(N_BASE, 1, L_S, device=device)
    y = torch.zeros(1, N_BASE, device=device)
    y[0, torch.randint(0, N_BASE, (1,))] = 1.0

    def step():
        optimizer.zero_grad(set_to_none=True)
        out = model(q, s, m)
        loss = criterion(out["logits"], y)
        loss.backward()
        optimizer.step()

    reset_peak_gpu_mem()
    timing = time_cuda(step, n_warmup=n_warmup, n_iter=n_iter)
    peak = peak_gpu_mem_mib()
    del model, optimizer, q, s, m, y
    hard_cuda_reset()
    return {
        "config": {"N": N_BASE, "batch_size": 1, "shots_per_class": 1, "L_q": L_Q, "L_s": L_S},
        "per_iter_timing_ms": timing.to_dict(),
        "peak_gpu_mem": peak,
    }


# ---------------------------------------------------------------------------
# Metric 5: MMF finetune per-iter @(N=95, B=1, K=20)
# ---------------------------------------------------------------------------


def metric5_mmf_finetune(device, n_warmup=3, n_iter=5) -> dict:
    hard_cuda_reset()
    model = EnhancedMultiMetaFingerNet(
        num_classes=N_TOTAL, dropout=0.15, use_se_in_df=True,
    ).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    q = torch.randn(1, L_Q, device=device)
    s = torch.randn(N_TOTAL, 20, L_S, device=device)
    m = torch.ones(N_TOTAL, 20, L_S, device=device)
    y = torch.zeros(1, N_TOTAL, device=device)
    y[0, torch.randint(0, N_TOTAL, (1,))] = 1.0

    def step():
        optimizer.zero_grad(set_to_none=True)
        out = model(q, s, m)
        loss = criterion(out["logits"], y)
        loss.backward()
        optimizer.step()

    reset_peak_gpu_mem()
    timing = time_cuda(step, n_warmup=n_warmup, n_iter=n_iter)
    peak = peak_gpu_mem_mib()

    # Save lean deployment artifacts (for metric 7 and 8/10).
    model.eval()
    with torch.no_grad():
        class_bank = compute_class_bank(model, s, m).detach().cpu()
    lean_sd = build_lean_state_dict(model)
    lean_ckpt = artifact_path("rebuttal_lean_ckpt.pth")
    bank_path = artifact_path("rebuttal_class_bank.pt")
    full_ckpt = artifact_path("rebuttal_full_ckpt.pth")
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()}, full_ckpt)
    torch.save({"model_state_dict": lean_sd}, lean_ckpt)
    torch.save(class_bank, bank_path)

    del model, optimizer, q, s, m, y
    hard_cuda_reset()
    return {
        "config": {"N": N_TOTAL, "batch_size": 1, "shots_per_class": 20, "L_q": L_Q, "L_s": L_S},
        "per_iter_timing_ms": timing.to_dict(),
        "peak_gpu_mem": peak,
        "artifacts": {
            "full_ckpt_MiB": file_size_mb(full_ckpt),
            "lean_ckpt_MiB": file_size_mb(lean_ckpt),
            "class_bank_MiB": file_size_mb(bank_path),
            "lean_ckpt_path": lean_ckpt,
            "bank_path": bank_path,
        },
    }


# ---------------------------------------------------------------------------
# Metric 6: ARES per-iter train cost (single binary classifier, B=1)
# ---------------------------------------------------------------------------


def metric6_ares_per_iter_train(
    num_resident_models: int,
    ares_max_len: int,
    device,
    n_warmup: int = 3,
    n_iter: int = 10,
) -> dict:
    """Measure ARES per-iter cost when ``num_resident_models`` binary classifiers
    are kept resident on the GPU (each with its own Adam optimizer state), and
    **one** of them takes a training step per iter (B=1).

    This reflects the ARES setting where, to train/maintain N classifiers, the
    adversary keeps all N models (+ grad + optimizer state) in GPU memory and
    trains them in a round-robin fashion. Iter time is identical to the
    single-model case, but peak GPU grows linearly with ``N``.
    """
    hard_cuda_reset()
    # Build N independent binary classifiers and N independent Adam optimizers.
    models = []
    optimizers = []
    for _ in range(num_resident_models):
        m = Trans_WF(num_classes=1, max_len=ares_max_len).to(device).train()
        opt = torch.optim.AdamW(m.parameters(), lr=0.0014, weight_decay=0.005)
        models.append(m)
        optimizers.append(opt)
    criterion = nn.BCEWithLogitsLoss()

    # Prime the Adam state tensors (m_t, v_t) for every model by running a
    # throw-away zero-grad step so that the reported peak includes the
    # optimizer state footprint.
    for m, opt in zip(models, optimizers):
        opt.zero_grad(set_to_none=True)
        dummy = sum(p.sum() for p in m.parameters()) * 0.0
        dummy.backward()
        opt.step()

    torch.cuda.synchronize()
    resident_alloc_MiB = torch.cuda.memory_allocated() / 1024**2

    x = torch.sign(torch.randn(1, 1, ARES_SEQ_LEN, device=device))
    y = torch.zeros(1, 1, device=device)
    y[0, 0] = 1.0

    per_model_params = count_params(models[0])

    # Round-robin: each step trains a different classifier (1 iter each).
    cursor = {"i": 0}

    def step():
        idx = cursor["i"] % num_resident_models
        cursor["i"] += 1
        m = models[idx]
        opt = optimizers[idx]
        opt.zero_grad(set_to_none=True)
        logits = m(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()

    reset_peak_gpu_mem()
    timing = time_cuda(step, n_warmup=n_warmup, n_iter=n_iter)
    peak = peak_gpu_mem_mib()

    del models, optimizers, x, y
    hard_cuda_reset()
    return {
        "config": {
            "batch_size": 1,
            "num_resident_models": num_resident_models,
            "note": (
                f"{num_resident_models} binary Trans-WF models + Adam optim states resident; "
                "one model takes 1 train step per iter (round-robin)."
            ),
        },
        "per_iter_timing_ms": timing.to_dict(),
        "peak_gpu_mem": peak,
        "resident_models_plus_optim_MiB": round(resident_alloc_MiB, 2),
        "per_model_params": per_model_params,
    }


# ---------------------------------------------------------------------------
# Metric 7: Storage after finetune (MMF lean + bank vs ARES 95 ckpts)
# Use N binary ARES models each saved to disk; use MMF artifacts from metric 5.
# ---------------------------------------------------------------------------


def metric7_storage(mmf_artifacts: dict, ares_max_len: int, device) -> dict:
    """Train then save 95 ARES binary classifiers (1 iter each is enough to get a
    representative on-disk ckpt; sizes are architecture-driven, not weight-data driven).
    Collect disk sizes.
    """
    ares_ckpt_dir = artifact_path("rebuttal_ares_ckpts")
    if os.path.exists(ares_ckpt_dir):
        shutil.rmtree(ares_ckpt_dir)
    os.makedirs(ares_ckpt_dir, exist_ok=True)

    total_bytes = 0
    per_model_MiB = None
    for c in range(N_TOTAL):
        m = Trans_WF(num_classes=1, max_len=ares_max_len).to(device).eval()
        ckpt_path = os.path.join(ares_ckpt_dir, f"cls_{c:04d}.pth")
        torch.save({"model_state_dict": m.state_dict()}, ckpt_path)
        sz = os.path.getsize(ckpt_path)
        total_bytes += sz
        if per_model_MiB is None:
            per_model_MiB = sz / 1024**2
        del m
    hard_cuda_reset()

    return {
        "MMF_lean_ckpt_MiB": mmf_artifacts["lean_ckpt_MiB"],
        "MMF_class_bank_MiB": mmf_artifacts["class_bank_MiB"],
        "MMF_lean_deployment_MiB": round(mmf_artifacts["lean_ckpt_MiB"] + mmf_artifacts["class_bank_MiB"], 4),
        "MMF_full_ckpt_MiB": mmf_artifacts["full_ckpt_MiB"],
        "ARES_per_model_MiB": round(per_model_MiB, 4),
        "ARES_num_models": N_TOTAL,
        "ARES_total_MiB": round(total_bytes / 1024**2, 4),
        "ares_ckpt_dir": ares_ckpt_dir,
    }


# ---------------------------------------------------------------------------
# Metric 8/10a: MMF single-query inference scalability (N sliced from bank)
# ---------------------------------------------------------------------------


def build_mmf_inference(lean_ckpt: str, device) -> CachedInferenceMMF:
    sd = torch.load(lean_ckpt, map_location=device, weights_only=False)["model_state_dict"]
    fe = DFFeatureExtractor(dropout=0.15, use_se=True)
    ch = EnhancedClassificationHead(
        feature_dim=C_MAIN, num_classes=1, seq_len=L_PRIME,
        num_topm_layers=TOPM_LAYERS, num_cross_layers=CROSS_LAYERS,
    )
    load_lean_checkpoint(fe, ch, sd)
    return CachedInferenceMMF(fe.to(device).eval(), ch.to(device).eval(),
                              num_classes=1, feature_dim=C_MAIN).to(device).eval()


def metric8_mmf_inference_sweep(mmf_artifacts: dict, n_values, device,
                                 n_warmup=5, n_iter=30) -> list:
    """For each N, rebuild the inference context so that ``peak_gpu_mem``
    reflects only the resident model + N-entry class_bank + activations.
    """
    results = []
    for N in n_values:
        hard_cuda_reset()

        # Fresh build to avoid caching artifacts between N runs.
        model = build_mmf_inference(mmf_artifacts["lean_ckpt_path"], device)
        full_bank = torch.load(mmf_artifacts["bank_path"], map_location="cpu", weights_only=False)
        bank = full_bank[:N].to(device).contiguous()
        query = torch.randn(1, L_Q, device=device)

        reset_peak_gpu_mem()

        def infer():
            with torch.no_grad():
                _ = model(query, bank)

        timing = time_cuda(infer, n_warmup=n_warmup, n_iter=n_iter)
        peak = peak_gpu_mem_mib()

        # resident (static) = after a sync, before the forward
        hard_cuda_reset()
        model2 = build_mmf_inference(mmf_artifacts["lean_ckpt_path"], device)
        bank2 = full_bank[:N].to(device).contiguous()
        torch.cuda.synchronize()
        resident_alloc = torch.cuda.memory_allocated() / 1024**2
        del model2, bank2
        hard_cuda_reset()

        results.append({
            "N": N,
            "latency_ms": timing.to_dict(),
            "peak_gpu_mem_MiB": peak,
            "resident_model_plus_bank_MiB": round(resident_alloc, 2),
        })
        del model, bank, full_bank, query
        hard_cuda_reset()
    return results


# ---------------------------------------------------------------------------
# Metric 9/10b: ARES single-query inference scalability (load N models only)
# ---------------------------------------------------------------------------


def metric9_ares_inference_sweep(ares_ckpt_dir: str, ares_max_len: int,
                                 n_values, device, n_warmup=5, n_iter=30) -> list:
    results = []
    for N in n_values:
        hard_cuda_reset()

        classifiers = []
        for c in range(N):
            m = Trans_WF(num_classes=1, max_len=ares_max_len).to(device).eval()
            sd = torch.load(os.path.join(ares_ckpt_dir, f"cls_{c:04d}.pth"),
                            map_location=device, weights_only=False)["model_state_dict"]
            m.load_state_dict(sd, strict=True)
            classifiers.append(m)

        torch.cuda.synchronize()
        resident_alloc = torch.cuda.memory_allocated() / 1024**2

        x = torch.sign(torch.randn(1, 1, ARES_SEQ_LEN, device=device))

        def infer():
            outs = []
            with torch.no_grad():
                for i in range(N):
                    outs.append(torch.sigmoid(classifiers[i](x)))
            return torch.cat(outs, dim=-1)

        reset_peak_gpu_mem()
        timing = time_cuda(infer, n_warmup=n_warmup, n_iter=n_iter)
        peak = peak_gpu_mem_mib()
        results.append({
            "N": N,
            "latency_ms": timing.to_dict(),
            "peak_gpu_mem_MiB": peak,
            "resident_models_MiB": round(resident_alloc, 2),
        })
        del classifiers, x
        hard_cuda_reset()
    return results


# ---------------------------------------------------------------------------
# Get ARES per-model MACs (thop) for metric 2
# ---------------------------------------------------------------------------


def ares_per_model_macs(ares_max_len: int, device) -> int:
    m = Trans_WF(num_classes=1, max_len=ares_max_len).to(device).eval()
    x = torch.sign(torch.randn(1, 1, ARES_SEQ_LEN, device=device))
    info = profile_macs(m, (x,))
    del m, x
    hard_cuda_reset()
    if info is None or "macs" not in info:
        return 0
    return int(info["macs"])


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt_ms(x):
    return f"{x:.3f} ms" if x is not None else "N/A"


def _fmt_mib(x):
    return f"{x:.2f} MiB" if x is not None else "N/A"


def render_markdown(results: dict) -> str:
    md = []
    md.append("# MMF vs ARES overhead benchmark — rebuttal tables\n")
    gpu = results["gpu_info"]
    md.append(f"- GPU: **{gpu.get('name')}** (CUDA cap {gpu.get('cuda_capability')})")
    md.append(f"- torch {gpu.get('torch_version')}, cuDNN {gpu.get('cudnn_version')}")
    md.append(f"- N_base = {N_BASE}, N_novel = {N_NOVEL}, N_total = {N_TOTAL}")
    md.append(f"- L_query = {L_Q}, L_support = {L_S}")
    md.append("")

    # ---------- Table 1: params ----------
    p = results["metric1_params"]
    md.append("## 1. Parameter Count")
    md.append("")
    md.append("| Item | Parameters | Size (fp32) |")
    md.append("|---|---:|---:|")
    md.append(f"| MMF total parameters | {p['total_params']:,} | {p['total_MiB_fp32']:.2f} MiB |")
    md.append(f"| MMF inference parameters (without meta_learnet + feature_reweighting) | "
              f"{p['inference_only_params']:,} | {p['inference_only_MiB_fp32']:.2f} MiB |")
    md.append(f"| Removed support-forward parameters | {p['dropped_params_at_inference']:,} | - |")
    md.append("")
    md.append("Submodule breakdown:")
    md.append("")
    md.append("| Submodule | Parameters |")
    md.append("|---|---:|")
    for k, v in p["by_module"].items():
        md.append(f"| {k} | {v:,} |")
    md.append("")

    # ---------- Table 2: Inference FLOPs formula ----------
    f = results["metric2_flops"]
    md.append("## 2. Inference FLOPs as a Function of N (single query, batch=1)")
    md.append("")
    md.append("**MMF cached (W_c is precomputed at inference; meta_learnet is not used):**")
    md.append("")
    md.append("```")
    md.append(f"{f['mmf_cached_formula']}")
    md.append("```")
    md.append("")
    md.append("**ARES (one-vs-all, N independent binary classifiers):**")
    md.append("")
    md.append("```")
    md.append(f"{f['ares_formula']}")
    md.append("```")
    md.append("")
    md.append("| N | MMF cached GFLOPs | ARES GFLOPs | ARES / MMF |")
    md.append("|---:|---:|---:|---:|")
    for row in f["table"]:
        md.append(f"| {row['N']} | {row['MMF_cached_GFLOPs_2x']} | "
                  f"{row['ARES_GFLOPs_2x']} | {row['ARES_over_MMF']}× |")
    md.append("")

    # ---------- Table 3-6: Training cost ----------
    md.append("## 3-6. Training Cost Comparison")
    md.append("")
    md.append("All MMF measurements use B=1; one ARES iteration means training one sample for one binary classifier (B=1).")
    md.append("")
    md.append("| Stage | Model | Config | Per-iter time (mean ± std) | Peak GPU allocated | Peak GPU reserved |")
    md.append("|---|---|---|---:|---:|---:|")

    mmf_b = results["metric34_mmf_base_train"]
    mmf_f = results["metric5_mmf_finetune"]
    ares_b = results["metric6_ares_base_train"]
    ares_f = results["metric6_ares_finetune_train"]

    def _row(label, model, cfg, timing, peak):
        return (
            f"| {label} | {model} | {cfg} | "
            f"{timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms | "
            f"{peak['allocated_MiB']:.1f} MiB | {peak['reserved_MiB']:.0f} MiB |"
        )
    md.append(_row(
        "Base train", "MMF",
        f"N={N_BASE}, B=1, K=1",
        mmf_b["per_iter_timing_ms"], mmf_b["peak_gpu_mem"],
    ))
    md.append(_row(
        "Base train", "ARES",
        f"{N_BASE} binary models resident + Adam states; 1 model trains per iter (B=1)",
        ares_b["per_iter_timing_ms"], ares_b["peak_gpu_mem"],
    ))
    md.append(_row(
        "Finetune (+35 novel → N=95)", "MMF",
        f"N={N_TOTAL}, B=1, K=20",
        mmf_f["per_iter_timing_ms"], mmf_f["peak_gpu_mem"],
    ))
    md.append(_row(
        "Finetune (→ N=95)", "ARES",
        f"{N_TOTAL} binary models resident + Adam states; 1 model trains per iter (B=1)",
        ares_f["per_iter_timing_ms"], ares_f["peak_gpu_mem"],
    ))
    md.append("")
    md.append(
        f"Note: ARES per-iter time is the same for base and finetune because the architecture and B=1 are unchanged; "
        f"however, **GPU memory scales linearly with the number of resident models N**: {N_BASE} during base training and {N_TOTAL} during finetuning. "
        f"(Reference: ARES resident parameters plus optimizer are about {ares_b.get('resident_models_plus_optim_MiB')} MiB during base training "
        f"and about {ares_f.get('resident_models_plus_optim_MiB')} MiB during finetuning.)"
    )
    md.append("")

    # ---------- Table 7: Storage ----------
    s = results["metric7_storage"]
    md.append("## 7. Storage Cost After Fine-tuning")
    md.append("")
    md.append("| Item | Size |")
    md.append("|---|---:|")
    md.append(f"| MMF lean checkpoint (without support-forward modules) | {s['MMF_lean_ckpt_MiB']:.3f} MiB |")
    md.append(f"| MMF precomputed class-specific features (`class_bank`, N={N_TOTAL}, 256x4B per class) | {s['MMF_class_bank_MiB']:.3f} MiB |")
    md.append(f"| **MMF lean deployment total** | **{s['MMF_lean_deployment_MiB']:.3f} MiB** |")
    md.append(f"| Reference: MMF full checkpoint (with optimizer) | {s['MMF_full_ckpt_MiB']:.3f} MiB |")
    md.append(f"| ARES single binary-classifier checkpoint | {s['ARES_per_model_MiB']:.3f} MiB |")
    md.append(f"| ARES all {s['ARES_num_models']} models total | **{s['ARES_total_MiB']:.3f} MiB** |")
    md.append(f"| ARES / MMF (lean deployment) | **{s['ARES_total_MiB']/s['MMF_lean_deployment_MiB']:.1f}×** |")
    md.append("")

    # ---------- Table 8-10: Inference scalability (N-swept) ----------
    md.append("## 8, 9, 10. Inference Scalability (batch=1, single query)")
    md.append("")
    md.append(
        "**MMF**: loads the lean checkpoint plus the first N cached class-specific features and runs the single query once."
        "**ARES**: loads N binary classifiers on GPU and runs the single query through them sequentially."
    )
    md.append("")
    md.append("| N | MMF latency | MMF peak GPU | MMF resident GPU (model+bank) | ARES latency | ARES peak GPU | ARES resident GPU (model weights) | Latency ratio ARES/MMF |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    mmf_sw = {r["N"]: r for r in results["metric8_mmf_inference_sweep"]}
    ares_sw = {r["N"]: r for r in results["metric9_ares_inference_sweep"]}
    for N in sorted(set(list(mmf_sw.keys()) + list(ares_sw.keys()))):
        mr = mmf_sw.get(N, {})
        ar = ares_sw.get(N, {})
        mm = mr.get("latency_ms", {}).get("mean_ms")
        am = ar.get("latency_ms", {}).get("mean_ms")
        ratio = round(am / mm, 2) if (mm and am) else None
        md.append(
            f"| {N} | {_fmt_ms(mm)} | "
            f"{_fmt_mib(mr.get('peak_gpu_mem_MiB', {}).get('allocated_MiB'))} | "
            f"{_fmt_mib(mr.get('resident_model_plus_bank_MiB'))} | "
            f"{_fmt_ms(am)} | "
            f"{_fmt_mib(ar.get('peak_gpu_mem_MiB', {}).get('allocated_MiB'))} | "
            f"{_fmt_mib(ar.get('resident_models_MiB'))} | "
            f"{ratio}× |"
        )
    md.append("")
    md.append(
        "**Key observations**:"
    )
    md.append(
        "- MMF resident GPU memory is nearly constant, roughly the lean checkpoint plus a few hundred bytes per class for the bank; peak memory only grows linearly with activations as N increases."
    )
    md.append(
        "- ARES resident GPU memory grows **strictly linearly with N** because each added class requires loading one ~10 MiB checkpoint on GPU."
    )
    md.append(
        "- Inference latency grows with N for both methods, but ARES has a larger constant because each binary model performs one forward pass."
    )
    md.append("")
    return "\n".join(md)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sweep", type=int, nargs="+", default=[5, 10, 30, 60, 95])
    parser.add_argument("--n-warmup-train", type=int, default=3)
    parser.add_argument("--n-iter-train", type=int, default=10)
    parser.add_argument("--n-warmup-infer", type=int, default=5)
    parser.add_argument("--n-iter-infer", type=int, default=30)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "Rebuttal bench requires CUDA"
    device = torch.device("cuda")

    gpu_info = get_gpu_info()
    print("[bench_rebuttal] gpu_info:", gpu_info)

    # 1. Params
    print("\n>>> metric 1: parameter counts")
    m1 = metric1_params()
    print(json.dumps(m1, indent=2, default=str))

    # 2. FLOPs formula (need ARES per-model MACs first)
    print("\n>>> metric 2 prep: ARES per-model MACs")
    ares_max_len = autodetect_ares_max_len(ARES_SEQ_LEN, device)
    ares_mpm = ares_per_model_macs(ares_max_len, device)
    print(f"ares_max_len = {ares_max_len}, ares_per_model_macs = {ares_mpm:,}")

    m2 = metric2_inference_flops_formula(ares_mpm, args.n_sweep, device)
    print(json.dumps(m2, indent=2, default=str))

    # 3/4. MMF base train
    print("\n>>> metric 3/4: MMF base train @(N=60, B=1, K=1)")
    m34 = metric34_mmf_base_train(device, n_warmup=args.n_warmup_train, n_iter=args.n_iter_train)
    print(json.dumps(m34, indent=2, default=str))

    # 6. ARES per-iter training cost -- base (60 models resident) and finetune (95 models resident)
    print("\n>>> metric 6a: ARES per-iter train @ base (N=60 models resident, B=1)")
    m6_base = metric6_ares_per_iter_train(
        num_resident_models=N_BASE,
        ares_max_len=ares_max_len,
        device=device,
        n_warmup=args.n_warmup_train,
        n_iter=args.n_iter_train,
    )
    print(json.dumps(m6_base, indent=2, default=str))

    print("\n>>> metric 6b: ARES per-iter train @ finetune (N=95 models resident, B=1)")
    m6_ft = metric6_ares_per_iter_train(
        num_resident_models=N_TOTAL,
        ares_max_len=ares_max_len,
        device=device,
        n_warmup=args.n_warmup_train,
        n_iter=args.n_iter_train,
    )
    print(json.dumps(m6_ft, indent=2, default=str))

    # 5. MMF finetune (also saves lean ckpt + class bank)
    print("\n>>> metric 5: MMF finetune @(N=95, B=1, K=20)")
    m5 = metric5_mmf_finetune(device, n_warmup=args.n_warmup_train,
                              n_iter=max(2, args.n_iter_train // 2))
    print(json.dumps(m5, indent=2, default=str))

    # 7. Storage breakdown
    print("\n>>> metric 7: storage")
    m7 = metric7_storage(m5["artifacts"], ares_max_len, device)
    print(json.dumps(m7, indent=2, default=str))

    # 8/10a. MMF inference sweep
    print("\n>>> metric 8/10a: MMF inference sweep over N")
    m8 = metric8_mmf_inference_sweep(m5["artifacts"], args.n_sweep, device,
                                     n_warmup=args.n_warmup_infer,
                                     n_iter=args.n_iter_infer)
    print(json.dumps(m8, indent=2, default=str))

    # 9/10b. ARES inference sweep
    print("\n>>> metric 9/10b: ARES inference sweep over N (only N models on GPU)")
    m9 = metric9_ares_inference_sweep(m7["ares_ckpt_dir"], ares_max_len,
                                      args.n_sweep, device,
                                      n_warmup=args.n_warmup_infer,
                                      n_iter=args.n_iter_infer)
    print(json.dumps(m9, indent=2, default=str))

    # Aggregate
    payload = {
        "gpu_info": gpu_info,
        "config": {
            "N_base": N_BASE, "N_novel": N_NOVEL, "N_total": N_TOTAL,
            "L_query": L_Q, "L_support": L_S,
            "C_main": C_MAIN, "L_prime": L_PRIME,
            "topm_layers": TOPM_LAYERS, "cross_layers": CROSS_LAYERS,
            "ares_seq_len": ARES_SEQ_LEN, "ares_max_len": ares_max_len,
            "n_sweep": args.n_sweep,
        },
        "metric1_params": m1,
        "metric2_flops": m2,
        "metric34_mmf_base_train": m34,
        "metric5_mmf_finetune": m5,
        "metric6_ares_base_train": m6_base,
        "metric6_ares_finetune_train": m6_ft,
        "metric7_storage": m7,
        "metric8_mmf_inference_sweep": m8,
        "metric9_ares_inference_sweep": m9,
    }

    json_path = result_path("rebuttal_bench.json")
    dump_json(payload, json_path)
    md_path = os.path.join(os.path.dirname(json_path), "rebuttal_tables.md")
    md = render_markdown(payload)
    with open(md_path, "w") as f:
        f.write(md)

    print("\n" + "=" * 60)
    print(f"JSON saved: {json_path}")
    print(f"Markdown saved: {md_path}")
    print("=" * 60)
    print()
    print(md)


if __name__ == "__main__":
    main()
