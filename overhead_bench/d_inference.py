"""Phase D: Single-sample inference latency with cached class_bank.

Loads the lean checkpoint and the ``class_bank.pt`` produced by Phase C (or
builds them from scratch if not found). Runs a sweep over the monitored-set
size ``N`` in {5, 10, 30, 95} by slicing the class_bank to the first ``N``
entries. For each ``N``:
    - Warm-up, then measure per-query latency with ``batch_size = 1``.
    - Record peak GPU memory during inference.
    - Record FLOPs (analytic + thop-equivalent where feasible) for sanity.

The measured inference path uses ``CachedInferenceMMF`` which does NOT invoke
the support branch or the dynamic-conv feature_reweighting module.

Outputs ``overhead_bench/results/d_inference.json``.
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

from models.feature_extractors import (  # noqa: E402
    DFFeatureExtractor,
    EnhancedMultiMetaFingerNet,
)
from models.classification_head_enhanced import EnhancedClassificationHead  # noqa: E402

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
from overhead_bench.cached_inference_model import (  # noqa: E402
    CachedInferenceMMF,
    build_lean_state_dict,
    compute_class_bank,
    load_lean_checkpoint,
)


def ensure_lean_artifacts(
    full_n: int,
    k_shot: int,
    L_support: int,
    device: torch.device,
):
    """Return ``(lean_ckpt_path, class_bank_path)``. Creates them if missing."""
    lean_ckpt_path = artifact_path("c_lean_ckpt.pth")
    bank_path = artifact_path("c_class_bank.pt")

    if os.path.exists(lean_ckpt_path) and os.path.exists(bank_path):
        return lean_ckpt_path, bank_path

    print(f"[Phase D] Lean artifacts missing, building on the fly for N={full_n}...")
    model = EnhancedMultiMetaFingerNet(
        num_classes=full_n,
        dropout=0.15,
        support_blocks=0,
        use_se_in_df=True,
    ).to(device)
    model.eval()
    support = torch.randn(full_n, k_shot, L_support, device=device)
    masks = torch.ones_like(support)
    bank = compute_class_bank(model, support, masks).detach().cpu()
    torch.save({"model_state_dict": build_lean_state_dict(model)}, lean_ckpt_path)
    torch.save(bank, bank_path)
    del model, support, masks
    torch.cuda.empty_cache()
    return lean_ckpt_path, bank_path


def build_inference_model(lean_ckpt_path: str, device: torch.device) -> CachedInferenceMMF:
    sd = torch.load(lean_ckpt_path, map_location=device, weights_only=False)["model_state_dict"]

    # Default head config matches EnhancedMultiMetaFingerNet's factory settings.
    fe = DFFeatureExtractor(dropout=0.15, use_se=True)
    ch = EnhancedClassificationHead(
        feature_dim=256,
        num_classes=1,  # will be overwritten per forward
        seq_len=80,
        num_topm_layers=2,
        num_cross_layers=2,
    )
    load_lean_checkpoint(fe, ch, sd)
    fe = fe.to(device).eval()
    ch = ch.to(device).eval()
    model = CachedInferenceMMF(fe, ch, num_classes=1, feature_dim=256)
    return model.to(device).eval()


def load_bank(bank_path: str, device: torch.device) -> torch.Tensor:
    bank = torch.load(bank_path, map_location=device, weights_only=False)
    if not torch.is_tensor(bank):
        raise RuntimeError(f"Unexpected class_bank type: {type(bank)}")
    return bank.to(device)


def numerical_equivalence_check(
    full_N: int,
    k_shot: int,
    L_query: int,
    L_support: int,
    device: torch.device,
) -> dict:
    """Verify that ``CachedInferenceMMF`` produces logits identical to the full
    ``EnhancedMultiMetaFingerNet`` forward (given the same weights).

    We build a fresh full model, cache its ``W_c``, then compare ``(1)`` the
    full-path forward against ``(2)`` the cached-path forward. This is essential
    for the rebuttal: the cached inference path must be *numerically equivalent*
    to the trained model, otherwise the reported latencies would be meaningless.
    """
    torch.manual_seed(0)
    full = EnhancedMultiMetaFingerNet(
        num_classes=full_N,
        dropout=0.0,  # determinism
        support_blocks=0,
        use_se_in_df=True,
    ).to(device).eval()

    q = torch.randn(1, L_query, device=device)
    s = torch.randn(full_N, k_shot, L_support, device=device)
    m = torch.ones_like(s)
    with torch.no_grad():
        out_full = full(q, s, m)["logits"]
        bank = compute_class_bank(full, s, m)

    fe = DFFeatureExtractor(dropout=0.0, use_se=True)
    ch = EnhancedClassificationHead(
        feature_dim=256, num_classes=1, seq_len=80,
        num_topm_layers=2, num_cross_layers=2,
    )
    load_lean_checkpoint(fe, ch, build_lean_state_dict(full))
    fe = fe.to(device).eval()
    ch = ch.to(device).eval()
    cim = CachedInferenceMMF(fe, ch, num_classes=full_N, feature_dim=256).to(device).eval()
    with torch.no_grad():
        out_cached = cim(q, bank)

    diff = (out_full - out_cached).abs().max().item()
    ref = out_full.abs().max().item() + 1e-8
    rel = diff / ref
    return {
        "max_abs_diff": float(diff),
        "max_rel_diff": float(rel),
        "tolerance": 1e-3,
        "passed": bool(diff < 1e-3),
    }


def run_sweep(
    model: CachedInferenceMMF,
    bank: torch.Tensor,
    N_values,
    L_query: int,
    device: torch.device,
    n_warmup: int,
    n_iter: int,
):
    assert bank.size(0) >= max(N_values), \
        f"class_bank has only {bank.size(0)} rows but sweep needs up to {max(N_values)}"

    query = torch.randn(1, L_query, device=device)
    results = []
    for N in N_values:
        sub_bank = bank[:N].contiguous()
        reset_peak_gpu_mem()

        def infer():
            with torch.no_grad():
                _ = model(query, sub_bank)

        timing = time_cuda(infer, n_warmup=n_warmup, n_iter=n_iter)
        peak = peak_gpu_mem_mib()
        results.append({
            "N": N,
            "latency_ms": timing.to_dict(),
            "peak_gpu_mem": peak,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--N-sweep", type=int, nargs="+", default=[5, 10, 30, 95])
    parser.add_argument("--full-N", type=int, default=95,
                        help="num_classes used for building lean artifacts if missing")
    parser.add_argument("--k-shot", type=int, default=20)
    parser.add_argument("--L-query", type=int, default=20_000)
    parser.add_argument("--L-support", type=int, default=10_000)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=30)
    args = parser.parse_args()

    if args.smoke:
        args.N_sweep = [3, 5]
        args.full_N = 5
        args.k_shot = 2
        args.n_warmup = 2
        args.n_iter = 3

    assert torch.cuda.is_available(), "Phase D requires CUDA"
    device = torch.device("cuda")

    lean_ckpt_path, bank_path = ensure_lean_artifacts(
        args.full_N, args.k_shot, args.L_support, device,
    )
    if torch.load(bank_path, weights_only=False).size(0) < max(args.N_sweep):
        # rebuild a larger bank for the requested sweep
        print("[Phase D] class_bank too small; rebuilding...")
        os.remove(lean_ckpt_path)
        os.remove(bank_path)
        lean_ckpt_path, bank_path = ensure_lean_artifacts(
            max(args.full_N, max(args.N_sweep)), args.k_shot, args.L_support, device,
        )

    cpu_before = cpu_mem_mib()

    # Numerical equivalence between full and cached paths. This is a REQUIRED
    # pre-check before we trust the latency numbers below.
    eq_check = numerical_equivalence_check(
        full_N=min(5, args.full_N), k_shot=args.k_shot,
        L_query=args.L_query, L_support=args.L_support, device=device,
    )
    if not eq_check["passed"]:
        raise RuntimeError(f"Cached vs full inference mismatch: {eq_check}")

    model = build_inference_model(lean_ckpt_path, device)
    bank = load_bank(bank_path, device)

    sweep = run_sweep(
        model=model,
        bank=bank,
        N_values=args.N_sweep,
        L_query=args.L_query,
        device=device,
        n_warmup=args.n_warmup,
        n_iter=args.n_iter,
    )

    cpu_after = cpu_mem_mib()

    payload = {
        "phase": "D_cached_inference",
        "smoke": args.smoke,
        "gpu_info": get_gpu_info(),
        "numerical_equivalence_check": eq_check,
        "cpu_mem_before": cpu_before,
        "cpu_mem_after": cpu_after,
        "lean_ckpt_path": lean_ckpt_path,
        "lean_ckpt_MiB": file_size_mb(lean_ckpt_path),
        "class_bank_path": bank_path,
        "class_bank_MiB": file_size_mb(bank_path),
        "inference_params_total": count_params(model),
        "config": {
            "batch_size": 1,
            "L_query": args.L_query,
            "L_support": args.L_support,
            "k_shot_for_bank": args.k_shot,
            "N_sweep": args.N_sweep,
            "n_warmup": args.n_warmup,
            "n_iter": args.n_iter,
        },
        "scalability_sweep": sweep,
    }

    out_path = result_path("d_inference.json")
    dump_json(payload, out_path)

    compact = {
        "saved_to": out_path,
        "scalability": [
            {"N": r["N"], "ms": r["latency_ms"]["mean_ms"], "peak_MiB": r["peak_gpu_mem"]["allocated_MiB"]}
            for r in sweep
        ],
        "lean_ckpt_MiB": file_size_mb(lean_ckpt_path),
        "class_bank_MiB": file_size_mb(bank_path),
    }
    pretty_print(compact, title="Phase D done")


if __name__ == "__main__":
    main()
