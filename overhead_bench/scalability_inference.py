"""Focused scalability/inference-overhead benchmark for the rebuttal.

This script compares deployment-time MMF inference with one-vs-all binary
classifier banks adapted from multi-tab baselines as the monitored set grows.
It intentionally uses synthetic single-query inputs and cached MMF class-bank
vectors; it does not evaluate recognition accuracy and does not run MMF's
support branch.

Outputs:
  - overhead_bench/results/scalability_inference.json
  - overhead_bench/results/scalability_inference_table.md
  - overhead_bench/results/scalability_inference.png
  - overhead_bench/results/scalability_inference.pdf
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import Callable, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
PEERS_ROOT = os.path.abspath(os.path.join(REPO_ROOT, os.pardir, "MMF_peers_works"))
ARES_DIR = os.path.join(PEERS_ROOT, "ARES_pre")
BAPM_DIR = os.path.join(PEERS_ROOT, "BAPM")
FMWF_DIR = os.path.join(PEERS_ROOT, "FMWF_code")

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "models"))

from models.classification_head_enhanced import EnhancedClassificationHead  # noqa: E402
from models.feature_extractors import DFFeatureExtractor  # noqa: E402
from overhead_bench.bench_utils import (  # noqa: E402
    current_gpu_mem_mib,
    dump_json,
    get_gpu_info,
    peak_gpu_mem_mib,
    reset_peak_gpu_mem,
    result_path,
    time_cuda,
)
from overhead_bench.cached_inference_model import (  # noqa: E402
    CachedInferenceMMF,
    load_lean_checkpoint,
)


DEFAULT_MMF_CKPT = os.path.join(
    REPO_ROOT,
    "experiments/base_training/3tab_5shot_medium_down_3_20260120_135412_ddp/checkpoints/best_model.pth",
)


def load_class(module_name: str, path: str, class_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {class_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


Trans_WF = load_class("overhead_ares_model", os.path.join(ARES_DIR, "ARES.py"), "Trans_WF")
BAPM = load_class("overhead_bapm_model", os.path.join(BAPM_DIR, "model.py"), "BAPM")
CNNmodel = load_class("overhead_fmwf_model", os.path.join(FMWF_DIR, "models.py"), "CNNmodel")


def load_mmf_lean_state(path: str) -> Dict[str, torch.Tensor] | None:
    if not path or not os.path.isfile(path):
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    lean = {}
    for key, value in state_dict.items():
        key = key[7:] if key.startswith("module.") else key
        if key.startswith("feature_extractor.") or key.startswith("classification_head."):
            lean[key] = value.detach().cpu()
    return lean


def build_mmf_model(num_classes: int, device: torch.device, lean_state: Dict[str, torch.Tensor] | None):
    feature_extractor = DFFeatureExtractor(dropout=0.15, use_se=True)
    classification_head = EnhancedClassificationHead(
        feature_dim=256,
        num_classes=num_classes,
        seq_len=80,
        num_topm_layers=2,
        num_cross_layers=2,
    )
    if lean_state:
        load_lean_checkpoint(feature_extractor, classification_head, lean_state)
    model = CachedInferenceMMF(
        feature_extractor=feature_extractor,
        classification_head=classification_head,
        num_classes=num_classes,
        feature_dim=256,
    )
    return model.to(device).eval()


def build_ares_model(device: torch.device, max_len: int):
    return Trans_WF(num_classes=1, max_len=max_len).to(device).eval()


def build_bapm_model(device: torch.device):
    return BAPM(num_classes=1, num_tab=3).to(device).eval()


def build_fmwf_model(device: torch.device):
    return CNNmodel(num_classes=1).to(device).eval()


def autodetect_ares_max_len(seq_len: int, device: torch.device) -> int:
    model = Trans_WF(num_classes=1, max_len=1).to(device).eval()
    x = torch.zeros(1, 1, seq_len, device=device)
    with torch.no_grad():
        y = model.dividing(x)
        y = model.profiling(y)
        y = model.combination(y)
    max_len = int(y.size(-1))
    del model, x, y
    torch.cuda.empty_cache()
    return max_len


def measure_mmf(
    num_classes: int,
    query: torch.Tensor,
    device: torch.device,
    lean_state: Dict[str, torch.Tensor] | None,
    n_warmup: int,
    n_iter: int,
) -> Dict[str, float]:
    torch.cuda.empty_cache()
    model = build_mmf_model(num_classes, device, lean_state)
    class_bank = torch.randn(num_classes, 256, device=device)

    def infer():
        with torch.no_grad():
            return model(query, class_bank)

    reset_peak_gpu_mem()
    resident = current_gpu_mem_mib()
    timing = time_cuda(infer, n_warmup=n_warmup, n_iter=n_iter).to_dict()
    peak = peak_gpu_mem_mib()

    result = {
        "latency_mean_ms": timing["mean_ms"],
        "latency_std_ms": timing["std_ms"],
        "latency_min_ms": timing["min_ms"],
        "latency_max_ms": timing["max_ms"],
        "gpu_allocated_MiB": max(peak["allocated_MiB"], resident["allocated_MiB"]),
        "gpu_reserved_MiB": max(peak["reserved_MiB"], resident["reserved_MiB"]),
    }
    del model, class_bank
    torch.cuda.empty_cache()
    return result


def measure_ares(
    num_classes: int,
    query: torch.Tensor,
    device: torch.device,
    max_len: int,
    n_warmup: int,
    n_iter: int,
) -> Dict[str, float]:
    torch.cuda.empty_cache()
    models = [build_ares_model(device, max_len=max_len) for _ in range(num_classes)]

    def infer():
        outputs = []
        with torch.no_grad():
            for model in models:
                outputs.append(model(query))
        return torch.cat(outputs, dim=-1)

    reset_peak_gpu_mem()
    resident = current_gpu_mem_mib()
    timing = time_cuda(infer, n_warmup=n_warmup, n_iter=n_iter).to_dict()
    peak = peak_gpu_mem_mib()

    result = {
        "latency_mean_ms": timing["mean_ms"],
        "latency_std_ms": timing["std_ms"],
        "latency_min_ms": timing["min_ms"],
        "latency_max_ms": timing["max_ms"],
        "gpu_allocated_MiB": max(peak["allocated_MiB"], resident["allocated_MiB"]),
        "gpu_reserved_MiB": max(peak["reserved_MiB"], resident["reserved_MiB"]),
    }
    del models
    torch.cuda.empty_cache()
    return result


def measure_binary_bank(
    name: str,
    num_classes: int,
    query: torch.Tensor,
    build_model: Callable[[torch.device], nn.Module],
    device: torch.device,
    n_warmup: int,
    n_iter: int,
) -> Dict[str, float]:
    torch.cuda.empty_cache()
    models = [build_model(device) for _ in range(num_classes)]

    def infer():
        outputs = []
        with torch.no_grad():
            for model in models:
                outputs.append(model(query))
        return torch.cat(outputs, dim=-1)

    reset_peak_gpu_mem()
    resident = current_gpu_mem_mib()
    timing = time_cuda(infer, n_warmup=n_warmup, n_iter=n_iter).to_dict()
    peak = peak_gpu_mem_mib()

    result = {
        "method": name,
        "latency_mean_ms": timing["mean_ms"],
        "latency_std_ms": timing["std_ms"],
        "latency_min_ms": timing["min_ms"],
        "latency_max_ms": timing["max_ms"],
        "gpu_allocated_MiB": max(peak["allocated_MiB"], resident["allocated_MiB"]),
        "gpu_reserved_MiB": max(peak["reserved_MiB"], resident["reserved_MiB"]),
    }
    del models
    torch.cuda.empty_cache()
    return result


def row_to_flat(row: Dict[str, object], methods: List[str]) -> Dict[str, float]:
    out = {"Num_classes": row["Num_classes"]}
    metrics = row["metrics"]
    for method in methods:
        stats = metrics[method]
        out[f"latency/{method}_ms"] = round(stats["latency_mean_ms"], 4)
    for method in methods:
        stats = metrics[method]
        out[f"GPU/{method}_MiB"] = round(stats["gpu_allocated_MiB"], 2)
    for method in methods:
        stats = metrics[method]
        out[f"latency_std/{method}_ms"] = round(stats["latency_std_ms"], 4)
    for method in methods:
        stats = metrics[method]
        out[f"GPU_reserved/{method}_MiB"] = round(stats["gpu_reserved_MiB"], 2)
    return out


def write_markdown(rows: Iterable[Dict[str, object]], methods: List[str], path: str) -> None:
    headers = ["Num_classes"]
    headers.extend([f"latency/{method}" for method in methods])
    headers.extend([f"GPU/{method}" for method in methods])
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---:"] * len(headers)) + "|",
    ]
    for row in rows:
        metrics = row["metrics"]
        cells = [str(row["Num_classes"])]
        cells.extend([f"{metrics[method]['latency_mean_ms']:.2f} ms" for method in methods])
        cells.extend([f"{metrics[method]['gpu_allocated_MiB']:.1f} MiB" for method in methods])
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_rows(rows: List[Dict[str, object]], methods: List[str], png_path: str, pdf_path: str) -> None:
    n = [r["Num_classes"] for r in rows]

    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 0.9,
    })

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    colors = {
    "MMF":  "#c82423",
    "ARES": "#f79059",
    "TMWF": "#3480b8",
    "FMWF": "#8dcec8",
    "BAPM": "#9bbf8a",
    }
    markers = {"MMF": "o", "ARES": "s", "BAPM": "^", "FMWF": "D"}

    panels = [(axes[0], "latency_mean_ms", "Latency (ms)"),
              (axes[1], "gpu_allocated_MiB", "Peak GPU memory (MiB)")]

    for ax, metric_key, ylabel in panels:
        for method in methods:
            y = [row["metrics"][method][metric_key] for row in rows]
            ax.plot(
                n, y,
                marker=markers.get(method, "o"),
                color=colors.get(method),
                linewidth=2.0,
                markersize=5,
                label=method,
            )
        ax.set_xlabel("Number of monitored classes")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        ax.set_xlim(min(n) - 2, max(n) + 2)

    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--N-sweep",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 95],
    )
    parser.add_argument("--mmf-query-len", type=int, default=20_000)
    parser.add_argument("--ares-seq-len", type=int, default=10_000)
    parser.add_argument("--baseline-seq-len", type=int, default=12_000)
    parser.add_argument("--ares-max-len", type=int, default=-1)
    parser.add_argument("--mmf-ckpt", type=str, default=DEFAULT_MMF_CKPT)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--output-prefix", type=str, default="scalability_inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.N_sweep = [5, 10]
        args.n_warmup = 1
        args.n_iter = 2

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for overhead benchmarking")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    if args.ares_seq_len % 4 != 0:
        raise ValueError("--ares-seq-len must be divisible by 4 for ARES Trans_WF")

    lean_state = load_mmf_lean_state(args.mmf_ckpt)
    if lean_state is None:
        print(f"[WARN] MMF checkpoint not found, using random MMF weights: {args.mmf_ckpt}")

    if args.ares_max_len <= 0:
        args.ares_max_len = autodetect_ares_max_len(args.ares_seq_len, device)
    print(f"ARES max_len={args.ares_max_len}")
    print(f"N sweep={args.N_sweep}")

    mmf_query = torch.sign(torch.randn(1, args.mmf_query_len, device=device))
    ares_query = torch.sign(torch.randn(1, 1, args.ares_seq_len, device=device))
    baseline_query = torch.sign(torch.randn(1, 1, args.baseline_seq_len, device=device))
    methods = ["MMF", "ARES", "BAPM", "FMWF"]

    rows: List[Dict[str, object]] = []
    for num_classes in args.N_sweep:
        print(f"\n[Benchmark] N={num_classes}")
        mmf_stats = measure_mmf(
            num_classes=num_classes,
            query=mmf_query,
            device=device,
            lean_state=lean_state,
            n_warmup=args.n_warmup,
            n_iter=args.n_iter,
        )
        print(
            f"  MMF:  {mmf_stats['latency_mean_ms']:.3f} ms, "
            f"{mmf_stats['gpu_allocated_MiB']:.1f} MiB"
        )

        ares_stats = measure_ares(
            num_classes=num_classes,
            query=ares_query,
            device=device,
            max_len=args.ares_max_len,
            n_warmup=args.n_warmup,
            n_iter=args.n_iter,
        )
        print(
            f"  ARES: {ares_stats['latency_mean_ms']:.3f} ms, "
            f"{ares_stats['gpu_allocated_MiB']:.1f} MiB"
        )

        bapm_stats = measure_binary_bank(
            name="BAPM",
            num_classes=num_classes,
            query=baseline_query,
            build_model=build_bapm_model,
            device=device,
            n_warmup=args.n_warmup,
            n_iter=args.n_iter,
        )
        print(
            f"  BAPM: {bapm_stats['latency_mean_ms']:.3f} ms, "
            f"{bapm_stats['gpu_allocated_MiB']:.1f} MiB"
        )

        fmwf_stats = measure_binary_bank(
            name="FMWF",
            num_classes=num_classes,
            query=baseline_query,
            build_model=build_fmwf_model,
            device=device,
            n_warmup=args.n_warmup,
            n_iter=args.n_iter,
        )
        print(
            f"  FMWF: {fmwf_stats['latency_mean_ms']:.3f} ms, "
            f"{fmwf_stats['gpu_allocated_MiB']:.1f} MiB"
        )

        rows.append({
            "Num_classes": num_classes,
            "metrics": {
                "MMF": mmf_stats,
                "ARES": ares_stats,
                "BAPM": bapm_stats,
                "FMWF": fmwf_stats,
            },
        })

    out_json = result_path(f"{args.output_prefix}.json")
    out_md = result_path(f"{args.output_prefix}_table.md")
    out_png = result_path(f"{args.output_prefix}.png")
    out_pdf = result_path(f"{args.output_prefix}.pdf")

    payload = {
        "purpose": "Focused scalability and inference-overhead comparison as monitored set grows.",
        "notes": [
            "MMF uses cached class-bank vectors and does not run the support branch online.",
            "For each N, MMF is instantiated with an N-row class bank.",
            "ARES, BAPM, and FMWF are adapted as extensible one-vs-all banks with exactly N binary models.",
            "Inputs are synthetic single-query tensors; this benchmark does not measure detection accuracy.",
        ],
        "gpu_info": get_gpu_info(),
        "config": {
            "N_sweep": args.N_sweep,
            "mmf_query_len": args.mmf_query_len,
            "ares_seq_len": args.ares_seq_len,
            "baseline_seq_len": args.baseline_seq_len,
            "ares_max_len": args.ares_max_len,
            "mmf_ckpt": args.mmf_ckpt,
            "n_warmup": args.n_warmup,
            "n_iter": args.n_iter,
            "seed": args.seed,
        },
        "methods": methods,
        "table": [row_to_flat(row, methods) for row in rows],
        "detailed": rows,
    }
    dump_json(payload, out_json)
    write_markdown(rows, methods, out_md)
    plot_rows(rows, methods, out_png, out_pdf)

    print("\nSaved:")
    print(f"  JSON: {out_json}")
    print(f"  Table: {out_md}")
    print(f"  Figure PNG: {out_png}")
    print(f"  Figure PDF: {out_pdf}")
    print("\nMarkdown table:\n")
    with open(out_md, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
