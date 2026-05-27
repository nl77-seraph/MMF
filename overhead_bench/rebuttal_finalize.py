"""Finalize the rebuttal benchmark results (user-confirmed spec).

Changes relative to ``rebuttal_patch.py`` (v2):
1. ARES per-iter training stays serial (one sample -> N heads updated one
   after another in a Python for loop). This is exactly what the ARES
   reference code does; ``metric6`` from v2 already implements it, so we
   re-use its numbers.
2. ARES single-query inference is measured **sequentially** (N classifiers
   invoked in a Python for loop, matching the ARES reference code). The
   v2 vmap parallel numbers are discarded.
3. Finetune table only reports MMF @ K=1. The K=20 row is removed.
4. Renders both Chinese (``rebuttal_tables.md``) and English
   (``rebuttal_tables_en.md``) versions reflecting the above.
"""

from __future__ import annotations

import argparse
import json
import os
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

from ARES import Trans_WF  # noqa: E402

from overhead_bench.bench_utils import (  # noqa: E402
    dump_json,
    peak_gpu_mem_mib,
    reset_peak_gpu_mem,
    result_path,
    time_cuda,
)
from overhead_bench.bench_rebuttal import (  # noqa: E402
    ARES_SEQ_LEN,
    L_Q,
    L_S,
    N_BASE,
    N_NOVEL,
    N_TOTAL,
    autodetect_ares_max_len,
    hard_cuda_reset,
)


# ---------------------------------------------------------------------------
# Metric 9 (FINAL): ARES sequential inference — matches ARES reference code.
# ---------------------------------------------------------------------------


def metric9_ares_inference_sweep_sequential(
    ares_ckpt_dir: str,
    ares_max_len: int,
    n_values,
    device,
    n_warmup: int = 5,
    n_iter: int = 30,
) -> list:
    """N binary classifiers invoked one-by-one in a Python for loop.

    This is exactly how the ARES reference implementation predicts on a new
    query: iterate over every trained binary classifier, run its forward, and
    concatenate their sigmoid probabilities.
    """
    results = []
    for N in n_values:
        hard_cuda_reset()

        classifiers = []
        for c in range(N):
            m = Trans_WF(num_classes=1, max_len=ares_max_len).to(device).eval()
            sd = torch.load(
                os.path.join(ares_ckpt_dir, f"cls_{c:04d}.pth"),
                map_location=device, weights_only=False,
            )["model_state_dict"]
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
            "mode": "sequential_forloop",
        })
        del classifiers, x
        hard_cuda_reset()
    return results


# ---------------------------------------------------------------------------
# Markdown rendering — ZH / EN
# ---------------------------------------------------------------------------


def _fmt_ms(x): return f"{x:.3f} ms" if x is not None else "N/A"


def _fmt_mib(x): return f"{x:.2f} MiB" if x is not None else "N/A"


def _mmf_lean_total(s: dict) -> float:
    return round(s["MMF_lean_ckpt_MiB"] + s["MMF_class_bank_MiB"], 4)


def _render_train_row(label, model, cfg, timing, peak):
    return (
        f"| {label} | {model} | {cfg} | "
        f"{timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms | "
        f"{peak['allocated_MiB']:.1f} MiB | {peak['reserved_MiB']:.0f} MiB |"
    )


def render_zh(results: dict) -> str:
    md = []
    gpu = results["gpu_info"]
    md.append("# MMF vs ARES overhead benchmark — rebuttal tables")
    md.append("")
    md.append(f"- GPU: **{gpu.get('name')}** (CUDA cap {gpu.get('cuda_capability')})")
    md.append(f"- torch {gpu.get('torch_version')}, cuDNN {gpu.get('cudnn_version')}")
    md.append(f"- N_base = {N_BASE}, N_novel = {N_NOVEL}, N_total = {N_TOTAL}")
    md.append(f"- L_query = {L_Q}, L_support = {L_S}")
    md.append("")
    md.append("> **Final methodology:**")
    md.append(">")
    md.append("> 1. ARES training follows its **original serial implementation**: one sample is passed through N binary heads with forward+backward+optimizer.step, without extra parallelization, so the per-iter cost is about N times the single-head cost.")
    md.append(">")
    md.append("> 2. ARES inference also follows the **original sequential code path**: one query is passed through N binary heads in a Python loop, and outputs are concatenated into a (1, N) sigmoid probability vector. The ideal-case vmap parallelization is no longer used.")
    md.append(">")
    md.append("> 3. MMF fine-tuning keeps only the **K=1** measurement to align with the ARES setting of one support sample per class; the old K=20 measurement has been removed.")
    md.append("")

    # -------- 1. Params --------
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

    # -------- 2. FLOPs --------
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

    # -------- 3-6. Train cost --------
    md.append("## 3-6. Training Cost Comparison")
    md.append("")
    md.append("All MMF measurements use B=1; **one ARES iteration means forward+backward+step for all N heads**, following the original serial implementation.")
    md.append("")
    md.append("| Stage | Model | Config | Per-iter time (mean ± std) | Peak GPU allocated | Peak GPU reserved |")
    md.append("|---|---|---|---:|---:|---:|")

    mmf_b = results["metric34_mmf_base_train"]
    mmf_f_k1 = results["metric5b_mmf_finetune_K1"]
    ares_b = results["metric6_ares_base_train"]
    ares_f = results["metric6_ares_finetune_train"]

    md.append(_render_train_row(
        "Base train", "MMF", f"N={N_BASE}, B=1, K=1",
        mmf_b["per_iter_timing_ms"], mmf_b["peak_gpu_mem"],
    ))
    md.append(_render_train_row(
        "Base train", "ARES",
        f"N={N_BASE} heads resident + Adam; 1 sample -> serially update all {N_BASE} heads (B=1)",
        ares_b["per_iter_timing_ms"], ares_b["peak_gpu_mem"],
    ))
    md.append(_render_train_row(
        "Finetune (+35 novel → N=95)", "MMF", f"N={N_TOTAL}, B=1, K=1",
        mmf_f_k1["per_iter_timing_ms"], mmf_f_k1["peak_gpu_mem"],
    ))
    md.append(_render_train_row(
        "Finetune (→ N=95)", "ARES",
        f"N={N_TOTAL} heads resident + Adam; 1 sample -> serially update all {N_TOTAL} heads (B=1)",
        ares_f["per_iter_timing_ms"], ares_f["peak_gpu_mem"],
    ))
    md.append("")
    md.append(
        f"ARES resident parameters + Adam state, decoupled from per-iter time:"
        f" base stage ≈ {ares_b.get('resident_models_plus_optim_MiB')} MiB,"
        f" finetune stage ≈ {ares_f.get('resident_models_plus_optim_MiB')} MiB"
        f" (strictly linear in the number of heads)."
    )
    md.append("")

    # -------- 7. Storage --------
    s = results["metric7_storage"]
    md.append("## 7. Storage Cost After Fine-tuning")
    md.append("")
    md.append("| Item | Size |")
    md.append("|---|---:|")
    md.append(f"| MMF lean checkpoint (without support-forward modules) | {s['MMF_lean_ckpt_MiB']:.3f} MiB |")
    md.append(f"| MMF precomputed class-specific features (`class_bank`, N={N_TOTAL}, 256x4B per class) | {s['MMF_class_bank_MiB']:.3f} MiB |")
    md.append(f"| **MMF lean deployment total** | **{_mmf_lean_total(s):.3f} MiB** |")
    md.append(f"| Reference: MMF full checkpoint (with optimizer) | {s['MMF_full_ckpt_MiB']:.3f} MiB |")
    md.append(f"| ARES single binary-classifier checkpoint | {s['ARES_per_model_MiB']:.3f} MiB |")
    md.append(f"| ARES all {s['ARES_num_models']} models total | **{s['ARES_total_MiB']:.3f} MiB** |")
    md.append(f"| ARES / MMF (lean deployment) | **{s['ARES_total_MiB']/_mmf_lean_total(s):.1f}×** |")
    md.append("")

    # -------- 8-10. Inference scalability --------
    md.append("## 8, 9, 10. Inference Scalability (batch=1, single query)")
    md.append("")
    md.append(
        "**MMF**: loads the lean checkpoint plus the first N cached class-specific features and runs the single query once."
        "**ARES**: follows the original code path, processing the same query through N binary classifiers in a Python loop and concatenating the outputs into a (1, N) sigmoid vector."
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
    # Dynamically pull the N=5 / N=95 numbers to keep narrative consistent.
    _ares_by_n = {r["N"]: r for r in results["metric9_ares_inference_sweep"]}
    _mmf_by_n = {r["N"]: r for r in results["metric8_mmf_inference_sweep"]}
    _ares5 = _ares_by_n[5]["latency_ms"]["mean_ms"]
    _ares95 = _ares_by_n[95]["latency_ms"]["mean_ms"]
    _mmf5 = _mmf_by_n[5]["latency_ms"]["mean_ms"]
    _mmf95 = _mmf_by_n[95]["latency_ms"]["mean_ms"]
    _ares_peak95 = _ares_by_n[95]["peak_gpu_mem_MiB"]["allocated_MiB"]
    _ares_res95 = _ares_by_n[95]["resident_models_MiB"]
    _mmf_peak95 = _mmf_by_n[95]["peak_gpu_mem_MiB"]["allocated_MiB"]
    _mem_ratio = _ares_peak95 / _mmf_peak95 if _mmf_peak95 else None
    _lat_ratio = _ares95 / _mmf95 if _mmf95 else None

    md.append("**Key observations**:")
    md.append("- **GPU memory**: MMF resident GPU memory is about 33 MiB, and peak allocated memory barely grows with N. "
              "The class-specific increment (`(B, N, L', C)` reweighted activation + `(N, heads, L', L')` topm attention"
              " + `(1, heads, N, N)` cross-class attention) is at most about 30 MiB and is absorbed by the backbone"
              f" intermediate feature-map peak for L_q={L_Q} because the allocator can reuse memory blocks released by the backbone. "
              "The O(N) / O(N²) constants of the class-specific part are far smaller than the static backbone overhead, "
              "so they are barely visible for N <= 95. If N is pushed beyond roughly 256, the N² term of cross-class attention "
              "will exceed the backbone peak and become more visible. "
              f"**ARES grows strictly linearly with N**: each resident head adds about 10 MiB of weights, plus forward activations; "
              f"at N=95 the resident memory is about {_ares_res95:.0f} MiB and the peak is about {_ares_peak95:.0f} MiB "
              f"(about {_mem_ratio:.0f}x MMF).")
    md.append(f"- **Inference latency**: the serial ARES latency grows strictly linearly with N "
              f"(about {_ares5:.1f} ms at N=5 and {_ares95:.1f} ms at N=95), because each additional class adds an independent binary Trans-WF forward pass. "
              f"MMF single-query latency also grows approximately linearly but with a much lower slope "
              f"(about {_mmf5:.1f} ms at N=5 and {_mmf95:.1f} ms at N=95), mainly from TopM self-attention in the classification head expanding along the `B*N` batch dimension and the O(N²) cross-class attention term. "
              "This cost enables cross-class inference and few-shot novel onboarding, which ARES one-vs-all cannot provide.")
    md.append(f"- **Overall assessment**: on the same A100, MMF single-query peak GPU memory is only about {_mmf_peak95:.0f} MiB, "
              f"while ARES needs about {_ares_peak95:.0f} MiB (about {_mem_ratio:.0f}x higher). "
              "Deployment disk storage is 8.4 MiB for MMF versus 926 MiB for ARES (about 110x higher for ARES). "
              f"At N=95, inference latency is about {_mmf95:.0f} ms for MMF and {_ares95:.0f} ms for ARES "
              f"(about {_lat_ratio:.1f}x higher for ARES). "
              "Together with ARES training-side cost, where every novel class requires training a separate binary classifier from scratch, "
              "MMF is substantially better on parameters, storage, latency, and onboarding cost than the one-vs-all baseline.")
    md.append("")
    return "\n".join(md)


def render_en(results: dict) -> str:
    md = []
    gpu = results["gpu_info"]
    md.append("# MMF vs ARES overhead benchmark — rebuttal tables (EN)")
    md.append("")
    md.append(f"- GPU: **{gpu.get('name')}** (CUDA cap {gpu.get('cuda_capability')})")
    md.append(f"- torch {gpu.get('torch_version')}, cuDNN {gpu.get('cudnn_version')}")
    md.append(f"- N_base = {N_BASE}, N_novel = {N_NOVEL}, N_total = {N_TOTAL}")
    md.append(f"- L_query = {L_Q}, L_support = {L_S}")
    md.append("")
    md.append("> **Final methodology**:")
    md.append(">")
    md.append("> 1. ARES training is measured with the **serial implementation used"
              " in its own code**: for every sample the trainer loops over all N"
              " binary heads and runs forward + backward + optimizer.step on each"
              " one. Per-iter cost therefore scales as N × single-head cost.")
    md.append(">")
    md.append("> 2. ARES single-query inference is also measured **sequentially** —"
              " a Python for-loop over the N binary classifiers — which is exactly"
              " what the ARES reference code does. No vmap / parallel trickery is"
              " applied.")
    md.append(">")
    md.append("> 3. The finetune table reports MMF @ **K = 1** only, matching the"
              " 1-shot regime ARES uses. The earlier K = 20 row has been removed.")
    md.append("")

    # -------- 1. Params --------
    p = results["metric1_params"]
    md.append("## 1. Parameter count")
    md.append("")
    md.append("| Item | Params | Size (fp32) |")
    md.append("|---|---:|---:|")
    md.append(f"| MMF total params | {p['total_params']:,} | {p['total_MiB_fp32']:.2f} MiB |")
    md.append(f"| MMF inference params (meta_learnet + feature_reweighting removed) | "
              f"{p['inference_only_params']:,} | {p['inference_only_MiB_fp32']:.2f} MiB |")
    md.append(f"| Support-forward params dropped at inference | {p['dropped_params_at_inference']:,} | - |")
    md.append("")
    md.append("Submodule breakdown:")
    md.append("")
    md.append("| Submodule | Params |")
    md.append("|---|---:|")
    for k, v in p["by_module"].items():
        md.append(f"| {k} | {v:,} |")
    md.append("")

    # -------- 2. FLOPs --------
    f = results["metric2_flops"]
    md.append("## 2. Inference FLOPs as a function of N (batch = 1)")
    md.append("")
    md.append("**MMF cached (W_c pre-computed; meta_learnet is bypassed):**")
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

    # -------- 3-6. Train cost --------
    md.append("## 3–6. Training cost comparison")
    md.append("")
    md.append("All MMF measurements use B = 1; **one ARES iter = serial "
              "forward + backward + optimizer.step on each of the N heads** "
              "(exactly as in the reference code).")
    md.append("")
    md.append("| Stage | Method | Config | per-iter time (mean ± std) | peak GPU allocated | peak GPU reserved |")
    md.append("|---|---|---|---:|---:|---:|")

    mmf_b = results["metric34_mmf_base_train"]
    mmf_f_k1 = results["metric5b_mmf_finetune_K1"]
    ares_b = results["metric6_ares_base_train"]
    ares_f = results["metric6_ares_finetune_train"]

    md.append(_render_train_row(
        "Base train", "MMF", f"N={N_BASE}, B=1, K=1",
        mmf_b["per_iter_timing_ms"], mmf_b["peak_gpu_mem"],
    ))
    md.append(_render_train_row(
        "Base train", "ARES",
        f"N={N_BASE} heads resident + Adam; 1 sample -> serial update over all {N_BASE} heads (B=1)",
        ares_b["per_iter_timing_ms"], ares_b["peak_gpu_mem"],
    ))
    md.append(_render_train_row(
        "Finetune (+35 novel -> N=95)", "MMF",
        f"N={N_TOTAL}, B=1, K=1",
        mmf_f_k1["per_iter_timing_ms"], mmf_f_k1["peak_gpu_mem"],
    ))
    md.append(_render_train_row(
        "Finetune (-> N=95)", "ARES",
        f"N={N_TOTAL} heads resident + Adam; 1 sample -> serial update over all {N_TOTAL} heads (B=1)",
        ares_f["per_iter_timing_ms"], ares_f["peak_gpu_mem"],
    ))
    md.append("")
    md.append(
        f"ARES resident params + Adam state (decoupled from per-iter time):"
        f" base stage ~= {ares_b.get('resident_models_plus_optim_MiB')} MiB,"
        f" finetune stage ~= {ares_f.get('resident_models_plus_optim_MiB')} MiB"
        f" (grows strictly linearly with the number of heads)."
    )
    md.append("")

    # -------- 7. Storage --------
    s = results["metric7_storage"]
    md.append("## 7. Post-finetune storage footprint")
    md.append("")
    md.append("| Item | Size |")
    md.append("|---|---:|")
    md.append(f"| MMF lean checkpoint (support-forward dropped) | {s['MMF_lean_ckpt_MiB']:.3f} MiB |")
    md.append(f"| MMF cached class-specific features (`class_bank`, N={N_TOTAL}, 256x4B per class) | {s['MMF_class_bank_MiB']:.3f} MiB |")
    md.append(f"| **MMF lean deployment total** | **{_mmf_lean_total(s):.3f} MiB** |")
    md.append(f"| (Ref) MMF full checkpoint (with optimizer) | {s['MMF_full_ckpt_MiB']:.3f} MiB |")
    md.append(f"| ARES per-head checkpoint | {s['ARES_per_model_MiB']:.3f} MiB |")
    md.append(f"| ARES {s['ARES_num_models']} heads combined | **{s['ARES_total_MiB']:.3f} MiB** |")
    md.append(f"| ARES / MMF (lean deployment) | **{s['ARES_total_MiB']/_mmf_lean_total(s):.1f}x** |")
    md.append("")

    # -------- 8-10. Inference scalability --------
    md.append("## 8, 9, 10. Inference scalability (batch = 1, single query)")
    md.append("")
    md.append(
        "**MMF**: load the lean ckpt + first N cached class-specific "
        "features, run one forward. **ARES**: matches the reference code — "
        "iterate the N binary classifiers with a Python for-loop over the "
        "same query and concatenate the sigmoid outputs to get (1, N)."
    )
    md.append("")
    md.append("| N | MMF latency | MMF peak GPU | MMF resident GPU (model+bank) | ARES latency | ARES peak GPU | ARES resident GPU (weights) | ARES / MMF latency |")
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
            f"{ratio}x |"
        )
    md.append("")
    _ares_by_n = {r["N"]: r for r in results["metric9_ares_inference_sweep"]}
    _mmf_by_n = {r["N"]: r for r in results["metric8_mmf_inference_sweep"]}
    _ares5 = _ares_by_n[5]["latency_ms"]["mean_ms"]
    _ares95 = _ares_by_n[95]["latency_ms"]["mean_ms"]
    _mmf5 = _mmf_by_n[5]["latency_ms"]["mean_ms"]
    _mmf95 = _mmf_by_n[95]["latency_ms"]["mean_ms"]
    _ares_peak95 = _ares_by_n[95]["peak_gpu_mem_MiB"]["allocated_MiB"]
    _ares_res95 = _ares_by_n[95]["resident_models_MiB"]
    _mmf_peak95 = _mmf_by_n[95]["peak_gpu_mem_MiB"]["allocated_MiB"]
    _mem_ratio = _ares_peak95 / _mmf_peak95 if _mmf_peak95 else None
    _lat_ratio = _ares95 / _mmf95 if _mmf95 else None

    md.append("**Key observations**:")
    md.append("- **Memory**: MMF's resident GPU stays around 33 MiB and its "
              "peak allocated is nearly flat in N. The class-specific overhead "
              "grows as O(N) for the reweighted activation `(B, N, L', C)` "
              "plus top-M attention, and as O(N^2) for the cross-class "
              "attention, but up to N = 95 the total increment (~30 MiB) is "
              f"absorbed by the much larger backbone peak over L_q = {L_Q} "
              "(the allocator reuses memory blocks released by the backbone). "
              "Growth only becomes visible beyond N ~ 256 (where the N^2 "
              "cross-class term starts to dominate). ARES, in contrast, grows "
              "**strictly linearly in N** — every new binary head adds "
              "~10 MiB of weights plus one more forward-activation slice — "
              f"reaching ~{_ares_res95:.0f} MiB resident and ~{_ares_peak95:.0f} "
              f"MiB peak at N = 95, about {_mem_ratio:.0f}x the MMF deployment.")
    md.append("- **Latency**: ARES's sequential inference is **strictly "
              f"linear in N** (~{_ares5:.1f} ms at N = 5, ~{_ares95:.1f} ms "
              "at N = 95) because every extra class adds an independent "
              "binary Trans-WF forward. MMF's single-query latency also "
              f"grows with N but with a much smaller slope ({_mmf5:.1f} -> "
              f"{_mmf95:.1f} ms as N goes 5 -> 95); the TopM self-attention "
              "over `B*N` and the O(N^2) cross-class attention are the cost, "
              "and in exchange MMF gets the cross-class reasoning that "
              "enables few-shot novel onboarding, something the ARES "
              "one-vs-all formulation cannot do.")
    md.append(f"- **Overall**: on a single A100, MMF uses ~{_mmf_peak95:.0f} "
              f"MiB of peak GPU per query against ~{_ares_peak95:.0f} MiB "
              f"for ARES (~{_mem_ratio:.0f}x less), 8.4 MiB vs 926 MiB of "
              "on-disk weights (~110x less), and "
              f"{_mmf95:.0f} ms vs {_ares95:.0f} ms of per-query latency at "
              f"N = 95 (~{_lat_ratio:.1f}x faster). On top of that, ARES "
              "must train a brand-new binary classifier from scratch for "
              "every novel class at finetune time, so MMF is preferable in "
              "all four axes — parameters, storage, latency, and per-class "
              "onboarding cost.")
    md.append("")
    return "\n".join(md)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-in",
                        default=result_path("rebuttal_bench_v2.json"),
                        help="v2 JSON (already has metric5b K=1 and serial "
                             "metric6); its metric9 (vmap parallel) will be "
                             "overwritten by the sequential measurement.")
    parser.add_argument("--json-out",
                        default=result_path("rebuttal_bench_final.json"))
    parser.add_argument("--md-zh-out",
                        default=os.path.join(HERE, "results", "rebuttal_tables.md"))
    parser.add_argument("--md-en-out",
                        default=os.path.join(HERE, "results", "rebuttal_tables_en.md"))
    parser.add_argument("--n-sweep", type=int, nargs="+",
                        default=[5, 10, 30, 60, 95])
    parser.add_argument("--n-warmup-infer", type=int, default=5)
    parser.add_argument("--n-iter-infer", type=int, default=30)
    parser.add_argument("--ares-ckpt-dir", type=str, default=None)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()

    if args.render_only:
        with open(args.json_in) as f:
            merged = json.load(f)
        md_zh = render_zh(merged)
        with open(args.md_zh_out, "w") as f:
            f.write(md_zh)
        md_en = render_en(merged)
        with open(args.md_en_out, "w") as f:
            f.write(md_en)
        print(f"[render-only] ZH markdown: {args.md_zh_out}")
        print(f"[render-only] EN markdown: {args.md_en_out}")
        return

    assert torch.cuda.is_available(), "rebuttal_finalize requires CUDA"
    device = torch.device("cuda")

    with open(args.json_in) as f:
        base = json.load(f)

    ares_max_len = base["config"]["ares_max_len"]
    if not ares_max_len:
        ares_max_len = autodetect_ares_max_len(ARES_SEQ_LEN, device)
    ares_ckpt_dir = args.ares_ckpt_dir or base["metric7_storage"]["ares_ckpt_dir"]
    if not os.path.isdir(ares_ckpt_dir):
        raise FileNotFoundError(
            f"ARES ckpt dir not found: {ares_ckpt_dir}. "
            "Pass --ares-ckpt-dir.")
    print(f"[finalize] reusing ARES ckpt dir: {ares_ckpt_dir}")
    print(f"[finalize] ares_max_len = {ares_max_len}")

    print("\n>>> FINAL metric 9: ARES **sequential** inference (for-loop over N)")
    t0 = time.time()
    m9_seq = metric9_ares_inference_sweep_sequential(
        ares_ckpt_dir=ares_ckpt_dir,
        ares_max_len=ares_max_len,
        n_values=args.n_sweep,
        device=device,
        n_warmup=args.n_warmup_infer,
        n_iter=args.n_iter_infer,
    )
    print(json.dumps(m9_seq, indent=2, default=str))
    print(f"[time] metric 9 sequential: {time.time() - t0:.1f}s")

    merged = dict(base)
    merged["metric9_ares_inference_sweep"] = m9_seq
    merged.setdefault("patches", {})["final"] = {
        "ares_training_semantic": "serial: 1 sample -> update all N heads in a for-loop",
        "ares_inference_mode": "sequential_forloop (ARES reference code)",
        "mmf_finetune_reported": "K=1 only (K=20 row removed)",
    }

    dump_json(merged, args.json_out)
    print(f"\n[OK] JSON saved: {args.json_out}")

    md_zh = render_zh(merged)
    with open(args.md_zh_out, "w") as f:
        f.write(md_zh)
    print(f"[OK] ZH markdown saved: {args.md_zh_out}")

    md_en = render_en(merged)
    with open(args.md_en_out, "w") as f:
        f.write(md_en)
    print(f"[OK] EN markdown saved: {args.md_en_out}")


if __name__ == "__main__":
    main()
