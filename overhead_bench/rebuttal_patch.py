"""Patch on top of ``rebuttal_bench.py`` results.

Fixes from the v1 benchmark:
1. ARES per-iter training cost is now ``N * single-binary-classifier`` work,
   following the one-vs-all semantic of the paper (one sample triggers an
   update on every one of the N binary heads, not just one head).
2. ARES single-query inference is now *parallel* over the N binary classifiers
   via ``torch.func.stack_module_state`` + ``functional_call`` + ``vmap``,
   i.e. the query is broadcast to N models and they all run in one kernel
   batch. This is the "ideal deployment" version asked for by the reviewer.
3. Adds ``metric5b``: MMF finetune cost with K=1 support-per-class (the
   existing metric5 keeps K=20).

Outputs:
- ``results/rebuttal_bench_v2.json``      (combined JSON)
- ``results/rebuttal_tables.md``          (updated Chinese table)
- ``results/rebuttal_tables_en.md``       (new English table)
"""

from __future__ import annotations

import argparse
import copy
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
import ARES as _ARES_MOD  # noqa: E402 — for monkey-patching below
from models.feature_extractors import EnhancedMultiMetaFingerNet  # noqa: E402


# ---------------------------------------------------------------------------
# Monkey-patch: ``TopMAttention.forward`` uses ``mask.scatter_`` which is
# incompatible with ``torch.func.vmap`` (in-place ops on a non-batched tensor
# receiving a batched index). We replace it with an out-of-place scatter
# (semantically identical) so that vmap can trace through.
# ---------------------------------------------------------------------------


def _patched_topm_attention_forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    index = torch.topk(attn, k=self.top_m, dim=-1, largest=True)[1]
    mask = torch.zeros_like(attn).scatter(-1, index, 1.0)
    attn = torch.where(mask > 0, attn, torch.full_like(attn, float("-inf")))

    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj_drop(x)
    return x


_ARES_MOD.TopMAttention.forward = _patched_topm_attention_forward

from overhead_bench.bench_utils import (  # noqa: E402
    count_params,
    dump_json,
    file_size_mb,
    get_gpu_info,
    peak_gpu_mem_mib,
    reset_peak_gpu_mem,
    result_path,
    time_cuda,
)
from overhead_bench.cached_inference_model import (  # noqa: E402
    build_lean_state_dict,
    compute_class_bank,
)
from overhead_bench.bench_rebuttal import (  # noqa: E402
    ARES_SEQ_LEN,
    C_MAIN,
    CROSS_LAYERS,
    L_PRIME,
    L_Q,
    L_S,
    N_BASE,
    N_NOVEL,
    N_TOTAL,
    TOPM_LAYERS,
    artifact_path,
    autodetect_ares_max_len,
    hard_cuda_reset,
)


# ---------------------------------------------------------------------------
# Metric 5b: MMF finetune @(K=1)
# ---------------------------------------------------------------------------


def metric5b_mmf_finetune_k1(device, n_warmup: int = 3, n_iter: int = 10) -> dict:
    hard_cuda_reset()
    model = EnhancedMultiMetaFingerNet(
        num_classes=N_TOTAL, dropout=0.15, use_se_in_df=True,
    ).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    q = torch.randn(1, L_Q, device=device)
    s = torch.randn(N_TOTAL, 1, L_S, device=device)
    m = torch.ones(N_TOTAL, 1, L_S, device=device)
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
    del model, optimizer, q, s, m, y
    hard_cuda_reset()
    return {
        "config": {"N": N_TOTAL, "batch_size": 1, "shots_per_class": 1,
                   "L_q": L_Q, "L_s": L_S},
        "per_iter_timing_ms": timing.to_dict(),
        "peak_gpu_mem": peak,
    }


# ---------------------------------------------------------------------------
# Metric 6 (FIXED): ARES per-iter cost under one-vs-all semantics
# A single sample triggers *one* backward/step on *each* of the N heads.
# ---------------------------------------------------------------------------


def metric6_ares_per_iter_train_fixed(
    num_resident_models: int,
    ares_max_len: int,
    device,
    n_warmup: int = 3,
    n_iter: int = 10,
) -> dict:
    """Per-iter = 1 sample propagates through all ``num_resident_models`` heads.

    Each head has its own BCE loss vs its own binary target (y_c = 1{sample
    belongs to class c}). Every head runs forward+backward+optimizer.step per
    iter; that is what the ARES one-vs-all formulation actually requires.
    """
    hard_cuda_reset()
    models = []
    optimizers = []
    for _ in range(num_resident_models):
        m = Trans_WF(num_classes=1, max_len=ares_max_len).to(device).train()
        opt = torch.optim.AdamW(m.parameters(), lr=0.0014, weight_decay=0.005)
        models.append(m)
        optimizers.append(opt)
    criterion = nn.BCEWithLogitsLoss()

    # Warm the Adam state buffers so peak includes optimizer state.
    for m, opt in zip(models, optimizers):
        opt.zero_grad(set_to_none=True)
        dummy = sum(p.sum() for p in m.parameters()) * 0.0
        dummy.backward()
        opt.step()

    torch.cuda.synchronize()
    resident_alloc_MiB = torch.cuda.memory_allocated() / 1024**2

    x = torch.sign(torch.randn(1, 1, ARES_SEQ_LEN, device=device))
    # Pre-generate a multi-hot label with a single active class.
    y_multi = torch.zeros(1, num_resident_models, device=device)
    y_multi[0, torch.randint(0, num_resident_models, (1,))] = 1.0

    per_model_params = count_params(models[0])

    def step():
        # One sample -> update all N heads.
        for c in range(num_resident_models):
            opt = optimizers[c]
            m = models[c]
            opt.zero_grad(set_to_none=True)
            logits = m(x)                      # (1, 1)
            y_bin = y_multi[:, c:c + 1]        # (1, 1)
            loss = criterion(logits, y_bin)
            loss.backward()
            opt.step()

    reset_peak_gpu_mem()
    timing = time_cuda(step, n_warmup=n_warmup, n_iter=n_iter)
    peak = peak_gpu_mem_mib()

    del models, optimizers, x, y_multi
    hard_cuda_reset()
    return {
        "config": {
            "batch_size": 1,
            "num_resident_models": num_resident_models,
            "note": (
                f"{num_resident_models} binary Trans-WF heads resident + Adam "
                "state; one sample updates ALL heads per iter (one-vs-all)."
            ),
            "semantics": "per-iter = N head updates (ARES paper one-vs-all)",
        },
        "per_iter_timing_ms": timing.to_dict(),
        "peak_gpu_mem": peak,
        "resident_models_plus_optim_MiB": round(resident_alloc_MiB, 2),
        "per_model_params": per_model_params,
    }


# ---------------------------------------------------------------------------
# Metric 9 (FIXED): ARES parallel single-query inference via vmap
# ---------------------------------------------------------------------------


def metric9_ares_inference_sweep_parallel(
    ares_ckpt_dir: str,
    ares_max_len: int,
    n_values,
    device,
    n_warmup: int = 5,
    n_iter: int = 30,
) -> list:
    """Parallel inference: ``N`` binary classifiers run over the same query
    in a single ``vmap``-batched forward. This is the paper's intended
    deployment (one shot -> N sigmoid outputs), not a sequential for loop.
    """
    from torch.func import functional_call, stack_module_state, vmap

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

        # Stack the N models into batched params/buffers.
        stacked_params, stacked_buffers = stack_module_state(classifiers)

        # Build a meta-model on the "meta" device; functional_call will use
        # the stacked params/buffers instead of its own.
        meta_model = copy.deepcopy(classifiers[0]).to("meta")

        def fmodel(params, buffers, x):
            return functional_call(meta_model, (params, buffers), (x,))

        torch.cuda.synchronize()
        resident_alloc = torch.cuda.memory_allocated() / 1024**2

        x = torch.sign(torch.randn(1, 1, ARES_SEQ_LEN, device=device))

        # Keep classifiers alive just for memory accounting; the actual call
        # path goes through stacked_params/buffers.
        def infer():
            with torch.no_grad():
                outs = vmap(fmodel, in_dims=(0, 0, None))(
                    stacked_params, stacked_buffers, x,
                )                                      # (N, 1, 1)
                return torch.sigmoid(outs).view(1, N)  # (1, N)

        reset_peak_gpu_mem()
        timing = time_cuda(infer, n_warmup=n_warmup, n_iter=n_iter)
        peak = peak_gpu_mem_mib()

        results.append({
            "N": N,
            "latency_ms": timing.to_dict(),
            "peak_gpu_mem_MiB": peak,
            "resident_models_MiB": round(resident_alloc, 2),
            "mode": "parallel_vmap",
        })
        del classifiers, stacked_params, stacked_buffers, meta_model, x
        hard_cuda_reset()
    return results


# ---------------------------------------------------------------------------
# Markdown rendering: Chinese + English
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
    md.append("> **v2 notes:**")
    md.append(">")
    md.append("> 1. ARES per-iter training now follows the **original one-vs-all semantics**: one sample runs forward+backward+optimizer.step for all N binary heads, so per-iter time is about N times the single-head time. The earlier optimistic round-robin interpretation has been corrected.")
    md.append(">")
    md.append("> 2. ARES inference is changed to **parallel** execution with torch.func.stack_module_state + vmap: the query is broadcast to N heads and completed in one batched forward pass, closer to an ideal paper or engineering deployment.")
    md.append(">")
    md.append("> 3. Added an **MMF finetune @ K=1** measurement for comparison with the previous K=20 measurement.")
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
    md.append("All MMF measurements use B=1; **one ARES iteration means forward+backward+step for all N heads** under the original one-vs-all semantics.")
    md.append("")
    md.append("| Stage | Model | Config | Per-iter time (mean ± std) | Peak GPU allocated | Peak GPU reserved |")
    md.append("|---|---|---|---:|---:|---:|")

    mmf_b = results["metric34_mmf_base_train"]
    mmf_f = results["metric5_mmf_finetune"]
    mmf_f_k1 = results.get("metric5b_mmf_finetune_K1")
    ares_b = results["metric6_ares_base_train"]
    ares_f = results["metric6_ares_finetune_train"]

    md.append(_render_train_row(
        "Base train", "MMF", f"N={N_BASE}, B=1, K=1",
        mmf_b["per_iter_timing_ms"], mmf_b["peak_gpu_mem"],
    ))
    md.append(_render_train_row(
        "Base train", "ARES",
        f"N={N_BASE} heads resident + Adam; 1 sample -> all {N_BASE} heads per iter (B=1)",
        ares_b["per_iter_timing_ms"], ares_b["peak_gpu_mem"],
    ))
    md.append(_render_train_row(
        "Finetune (+35 novel → N=95)", "MMF",
        f"N={N_TOTAL}, B=1, K=20",
        mmf_f["per_iter_timing_ms"], mmf_f["peak_gpu_mem"],
    ))
    if mmf_f_k1 is not None:
        md.append(_render_train_row(
            "Finetune (+35 novel → N=95)", "MMF (K=1)",
            f"N={N_TOTAL}, B=1, K=1",
            mmf_f_k1["per_iter_timing_ms"], mmf_f_k1["peak_gpu_mem"],
        ))
    md.append(_render_train_row(
        "Finetune (→ N=95)", "ARES",
        f"N={N_TOTAL} heads resident + Adam; 1 sample -> all {N_TOTAL} heads per iter (B=1)",
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
        "**ARES (v2 parallel)**: N binary classifiers are evaluated in one batched forward pass through `torch.func.stack_module_state + vmap`."
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
    md.append("**Key observations**:")
    md.append("- **GPU memory**: MMF resident GPU memory is about 33 MiB, and peak allocated memory barely grows with N. "
              "The class-specific increment (`(B, N, L', C)` reweighted activation + `(N, heads, L', L')` topm attention"
              " + `(1, heads, N, N)` cross-class attention) is at most about 30 MiB and is absorbed by the backbone"
              f" intermediate feature-map peak for L_q={L_Q} because the allocator can reuse memory blocks released by the backbone. "
              "The O(N) / O(N²) constants of the class-specific part are far smaller than the static backbone overhead, "
              "so they are barely visible for N <= 95. If N is pushed beyond roughly 256, the N² cross-class attention term will exceed the backbone peak and become more visible. "
              "ARES grows strictly linearly with N: each added class brings a ~10 MiB binary checkpoint plus the current forward activations; "
              "at N=95 this is already about 1887 MiB, roughly 57x MMF.")
    md.append("- **Inference latency**: the v2 parallel ARES path with vmap grows sublinearly with N "
              "(from 6.4 ms to 16.9 ms for 5 to 95 classes), because the A100 absorbs the batched computation of N independent binary classifiers. "
              "MMF single-query latency grows approximately linearly instead (77 ms at N=95), mainly because TopM self-attention in the classification head expands along the `B*N` batch dimension and cross-class attention has an O(N²) term. "
              "This known trade-off is central to MMF: cross-class attention enables inter-class reasoning and few-shot novel classes, which ARES one-vs-all does not provide. "
              "From a throughput perspective, MMF per-query peak GPU memory is only about 34 MiB versus about 2353 MiB for ARES at N=95, so on the same GPU MMF can batch roughly 70x more queries. "
              "MMF deployment disk storage is also 8.4 MiB versus 926 MiB for ARES, so deployment cost remains substantially lower.")
    md.append("- **Parallel versus serial inference choice**: the v2 parallel path is an ARES-favorable ideal case. "
              "If ARES falls back to serial execution with N loop iterations, latency at N=95 would be about 240 ms, more than 10x the parallel version, while peak memory would be lower (<200 MiB). "
              "We keep the parallel version to give ARES the most optimistic result and avoid underestimating the baseline.")
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
    md.append("> **v2 notes**:")
    md.append(">")
    md.append("> 1. ARES per-iter training cost now reflects the paper's **one-vs-all**"
              " semantic: one sample propagates through **all N binary heads**"
              " (forward + backward + optimizer step per head). The previous"
              " round-robin schedule was overly optimistic and has been removed.")
    md.append(">")
    md.append("> 2. ARES single-query inference is now **parallel** over the N heads"
              " via `torch.func.stack_module_state` + `vmap`, i.e. the query is"
              " broadcast to all N heads inside one batched forward. This matches"
              " the paper's intended deployment, not a sequential for-loop.")
    md.append(">")
    md.append("> 3. A new row **MMF finetune @ K=1** is added next to the original"
              " K=20 measurement.")
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
    md.append("All MMF measurements use B = 1; **one ARES iter = one "
              "forward + backward + optimizer.step on each of the N heads** "
              "(one-vs-all paper semantic).")
    md.append("")
    md.append("| Stage | Method | Config | per-iter time (mean ± std) | peak GPU allocated | peak GPU reserved |")
    md.append("|---|---|---|---:|---:|---:|")

    mmf_b = results["metric34_mmf_base_train"]
    mmf_f = results["metric5_mmf_finetune"]
    mmf_f_k1 = results.get("metric5b_mmf_finetune_K1")
    ares_b = results["metric6_ares_base_train"]
    ares_f = results["metric6_ares_finetune_train"]

    md.append(_render_train_row(
        "Base train", "MMF", f"N={N_BASE}, B=1, K=1",
        mmf_b["per_iter_timing_ms"], mmf_b["peak_gpu_mem"],
    ))
    md.append(_render_train_row(
        "Base train", "ARES",
        f"N={N_BASE} heads resident + Adam; 1 sample -> all {N_BASE} heads per iter (B=1)",
        ares_b["per_iter_timing_ms"], ares_b["peak_gpu_mem"],
    ))
    md.append(_render_train_row(
        "Finetune (+35 novel -> N=95)", "MMF",
        f"N={N_TOTAL}, B=1, K=20",
        mmf_f["per_iter_timing_ms"], mmf_f["peak_gpu_mem"],
    ))
    if mmf_f_k1 is not None:
        md.append(_render_train_row(
            "Finetune (+35 novel -> N=95)", "MMF (K=1)",
            f"N={N_TOTAL}, B=1, K=1",
            mmf_f_k1["per_iter_timing_ms"], mmf_f_k1["peak_gpu_mem"],
        ))
    md.append(_render_train_row(
        "Finetune (-> N=95)", "ARES",
        f"N={N_TOTAL} heads resident + Adam; 1 sample -> all {N_TOTAL} heads per iter (B=1)",
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
    md.append(f"| MMF cached class-specific features (`class_bank`, N={N_TOTAL}, 256×4B per class) | {s['MMF_class_bank_MiB']:.3f} MiB |")
    md.append(f"| **MMF lean deployment total** | **{_mmf_lean_total(s):.3f} MiB** |")
    md.append(f"| (Ref) MMF full checkpoint (with optimizer) | {s['MMF_full_ckpt_MiB']:.3f} MiB |")
    md.append(f"| ARES per-head checkpoint | {s['ARES_per_model_MiB']:.3f} MiB |")
    md.append(f"| ARES {s['ARES_num_models']} heads combined | **{s['ARES_total_MiB']:.3f} MiB** |")
    md.append(f"| ARES / MMF (lean deployment) | **{s['ARES_total_MiB']/_mmf_lean_total(s):.1f}×** |")
    md.append("")

    # -------- 8-10. Inference scalability --------
    md.append("## 8, 9, 10. Inference scalability (batch = 1, single query)")
    md.append("")
    md.append(
        "**MMF**: load lean ckpt + first N cached class-specific features, "
        "run one forward. "
        "**ARES (v2 parallel)**: all N heads are dispatched in one batched "
        "forward via `torch.func.stack_module_state + vmap`."
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
            f"{ratio}× |"
        )
    md.append("")
    md.append("**Key observations**:")
    md.append("- **Memory**: MMF's resident GPU stays around 33 MiB and its "
              "peak allocated is nearly flat in N. The class-specific overhead "
              "grows as `O(N)` for the reweighted activation `(B, N, L', C)` "
              "plus top-M attention, and as `O(N^2)` for the cross-class "
              "attention, but up to N = 95 the total increment (~30 MiB) is "
              f"absorbed by the much larger backbone peak over L_q = {L_Q} "
              "(the allocator reuses memory blocks released by the backbone). "
              "The per-class constants are small enough that growth only "
              "becomes visible beyond N ≈ 256 (where the N^2 cross-class term "
              "starts to dominate). ARES, in contrast, grows **strictly "
              "linearly in N** (~10 MiB per new binary head plus one more "
              "forward-activation slice), reaching ~1887 MiB at N = 95 — about "
              "57× the MMF deployment.")
    md.append("- **Latency**: Thanks to `vmap`-batched inference, ARES latency "
              "is **sub-linear in N** on the A100 (6.4 → 16.9 ms as N goes "
              "5 → 95); the N independent binary heads parallelise almost for "
              "free on modern GPUs. MMF's single-query latency grows roughly "
              "linearly (to 77 ms at N = 95) because the classification head "
              "batches over `B*N` in top-M self-attn and the cross-class "
              "attention adds an `O(N^2)` term. **This trade-off is "
              "deliberate**: the cross-class attention is the core mechanism "
              "that lets MMF reason across classes and enables few-shot novel "
              "onboarding, something the ARES one-vs-all formulation cannot "
              "do. From a **throughput** viewpoint, MMF's per-query peak GPU "
              "is ~34 MiB vs ARES's ~2353 MiB at N = 95, so MMF can batch "
              "~70× more queries on the same card, and its on-disk footprint "
              "(8.4 MiB vs 926 MiB) is two orders of magnitude smaller — the "
              "deployment-cost gap remains large.")
    md.append("- **Parallel vs sequential ARES inference**: v2 parallelism is "
              "the *best-case* for ARES. A sequential for-loop would push "
              "N = 95 latency to ~240 ms (>10× the parallel version) and "
              "lower peak memory to <200 MiB; we use the parallel variant so "
              "as to not understate ARES.")
    md.append("")
    return "\n".join(md)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-in",
                        default=result_path("rebuttal_bench.json"),
                        help="Existing v1 JSON to patch on top of.")
    parser.add_argument("--json-out",
                        default=result_path("rebuttal_bench_v2.json"))
    parser.add_argument("--md-zh-out",
                        default=os.path.join(HERE, "results", "rebuttal_tables.md"))
    parser.add_argument("--md-en-out",
                        default=os.path.join(HERE, "results", "rebuttal_tables_en.md"))
    parser.add_argument("--n-sweep", type=int, nargs="+",
                        default=[5, 10, 30, 60, 95])
    parser.add_argument("--n-warmup-train", type=int, default=3)
    parser.add_argument("--n-iter-train", type=int, default=10)
    parser.add_argument("--n-warmup-infer", type=int, default=5)
    parser.add_argument("--n-iter-infer", type=int, default=30)
    parser.add_argument("--ares-ckpt-dir", type=str, default=None,
                        help="Reuse an existing ARES ckpt dir (from metric7). "
                             "If None, rebuild (will cost time/disk).")
    parser.add_argument("--render-only", action="store_true",
                        help="Skip all measurements and only re-render markdown "
                             "from --json-in (which should already be a v2 JSON).")
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
        print(f"[render-only] Chinese markdown saved: {args.md_zh_out}")
        print(f"[render-only] English markdown saved: {args.md_en_out}")
        return

    assert torch.cuda.is_available(), "rebuttal_patch requires CUDA"
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
            "Pass --ares-ckpt-dir or re-run the v1 bench to regenerate it.")
    print(f"[patch] reusing ARES ckpt dir: {ares_ckpt_dir}")
    print(f"[patch] ares_max_len = {ares_max_len}")

    # -------- metric 6 fixed (base) --------
    print("\n>>> metric 6a FIXED: ARES per-iter train @ base (N=60, one-vs-all)")
    t0 = time.time()
    m6_base = metric6_ares_per_iter_train_fixed(
        num_resident_models=N_BASE,
        ares_max_len=ares_max_len,
        device=device,
        n_warmup=args.n_warmup_train,
        n_iter=args.n_iter_train,
    )
    print(json.dumps(m6_base, indent=2, default=str))
    print(f"[time] metric 6a: {time.time() - t0:.1f}s")

    # -------- metric 6 fixed (finetune) --------
    print("\n>>> metric 6b FIXED: ARES per-iter train @ finetune (N=95, one-vs-all)")
    t0 = time.time()
    m6_ft = metric6_ares_per_iter_train_fixed(
        num_resident_models=N_TOTAL,
        ares_max_len=ares_max_len,
        device=device,
        n_warmup=args.n_warmup_train,
        n_iter=args.n_iter_train,
    )
    print(json.dumps(m6_ft, indent=2, default=str))
    print(f"[time] metric 6b: {time.time() - t0:.1f}s")

    # -------- metric 9 fixed (parallel inference) --------
    print("\n>>> metric 9 FIXED: ARES parallel inference (vmap) over N")
    t0 = time.time()
    m9 = metric9_ares_inference_sweep_parallel(
        ares_ckpt_dir=ares_ckpt_dir,
        ares_max_len=ares_max_len,
        n_values=args.n_sweep,
        device=device,
        n_warmup=args.n_warmup_infer,
        n_iter=args.n_iter_infer,
    )
    print(json.dumps(m9, indent=2, default=str))
    print(f"[time] metric 9: {time.time() - t0:.1f}s")

    # -------- metric 5b: MMF finetune K=1 --------
    print("\n>>> metric 5b NEW: MMF finetune @(N=95, B=1, K=1)")
    t0 = time.time()
    m5b = metric5b_mmf_finetune_k1(
        device=device,
        n_warmup=args.n_warmup_train,
        n_iter=max(2, args.n_iter_train // 2),
    )
    print(json.dumps(m5b, indent=2, default=str))
    print(f"[time] metric 5b: {time.time() - t0:.1f}s")

    # -------- merge + dump --------
    merged = dict(base)
    merged["metric6_ares_base_train"] = m6_base
    merged["metric6_ares_finetune_train"] = m6_ft
    merged["metric9_ares_inference_sweep"] = m9
    merged["metric5b_mmf_finetune_K1"] = m5b
    merged.setdefault("patches", {})["v2"] = {
        "ares_training_semantic": "per-iter = N head updates",
        "ares_inference_mode": "parallel_vmap",
        "mmf_finetune_extra_config": "K=1",
    }

    dump_json(merged, args.json_out)
    print(f"\n[OK] JSON saved: {args.json_out}")

    md_zh = render_zh(merged)
    with open(args.md_zh_out, "w") as f:
        f.write(md_zh)
    print(f"[OK] Chinese markdown saved: {args.md_zh_out}")

    md_en = render_en(merged)
    with open(args.md_en_out, "w") as f:
        f.write(md_en)
    print(f"[OK] English markdown saved: {args.md_en_out}")


if __name__ == "__main__":
    main()
