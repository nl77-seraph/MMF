"""Common benchmarking helpers shared by all overhead_bench scripts.

This file centralises:
- GPU selection (single-card evaluation, default card 1, override via ``MMF_BENCH_GPU``)
- CUDA-event based wall-clock timing with warm-up
- Peak GPU / CPU memory snapshots
- Parameter counting and thop-based MACs/FLOPs profiling
- Tensor/disk size helpers
- JSON dumping helper

NOTE on FLOPs accounting:
    - ``thop`` cannot trace ``torch.nn.functional.conv1d`` calls that are invoked
      with an externally supplied weight (our ``DynamicConv1d`` in
      ``models/dynamic_conv1d.py``). The dynamic reweighting contribution is
      therefore computed analytically as ``N * C * L'`` MACs (pure elementwise
      multiplication because kernel_size == 1 + groups == C).
    - ``thop`` also struggles with some advanced modules (Top-M masking, Cross
      Class MHSA). We fall back to a sub-module level analytic formula that is
      verified against a small ``N`` thop measurement. This keeps the reported
      numbers reproducible and easy to differentiate w.r.t. ``N``.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    from thop import profile as _thop_profile  # type: ignore
except ImportError:  # pragma: no cover
    _thop_profile = None


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def set_visible_gpu(gpu_id: Optional[str] = None) -> str:
    """Pin the current process to a single GPU before torch.cuda initialises.

    Call order matters: this must run **before** any ``torch.cuda`` API is
    touched. In practice, each overhead_bench entry point calls this first.
    """
    if gpu_id is None:
        gpu_id = os.environ.get("MMF_BENCH_GPU", "1")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return str(gpu_id)


def get_gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "available": True,
        "logical_index": idx,
        "name": props.name,
        "total_memory_MiB": round(props.total_memory / 1024**2, 1),
        "multi_processor_count": props.multi_processor_count,
        "cuda_capability": f"{props.major}.{props.minor}",
        "torch_version": torch.__version__,
        "cudnn_version": torch.backends.cudnn.version(),
    }


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def reset_peak_gpu_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def peak_gpu_mem_mib() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_MiB": 0.0, "reserved_MiB": 0.0}
    return {
        "allocated_MiB": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "reserved_MiB": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
    }


def current_gpu_mem_mib() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_MiB": 0.0, "reserved_MiB": 0.0}
    return {
        "allocated_MiB": round(torch.cuda.memory_allocated() / 1024**2, 2),
        "reserved_MiB": round(torch.cuda.memory_reserved() / 1024**2, 2),
    }


def cpu_mem_mib() -> Dict[str, float]:
    if psutil is None:
        return {"rss_MiB": -1.0, "vms_MiB": -1.0}
    p = psutil.Process(os.getpid())
    info = p.memory_info()
    return {
        "rss_MiB": round(info.rss / 1024**2, 2),
        "vms_MiB": round(info.vms / 1024**2, 2),
    }


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


@dataclass
class TimingResult:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    n_iter: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_ms": round(self.mean_ms, 4),
            "std_ms": round(self.std_ms, 4),
            "min_ms": round(self.min_ms, 4),
            "max_ms": round(self.max_ms, 4),
            "n_iter": self.n_iter,
        }


def time_cuda(fn: Callable[[], Any], n_warmup: int = 3, n_iter: int = 10) -> TimingResult:
    """Time a callable with CUDA events and a synchronising warm-up.

    The callable should take no arguments and perform a single iteration of the
    operation to be measured (e.g. one forward pass).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("time_cuda requires CUDA")

    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    starters = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    enders = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        starters[i].record()
        fn()
        enders[i].record()
    torch.cuda.synchronize()

    times_ms = [starters[i].elapsed_time(enders[i]) for i in range(n_iter)]
    mean = sum(times_ms) / n_iter
    var = sum((t - mean) ** 2 for t in times_ms) / n_iter
    std = var ** 0.5
    return TimingResult(
        mean_ms=mean,
        std_ms=std,
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        n_iter=n_iter,
    )


@contextmanager
def wallclock(name: str = ""):
    """Context manager that yields a dict and fills ``seconds`` on exit."""
    info: Dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield info
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        info["seconds"] = time.perf_counter() - start
        info["name"] = name


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------


def count_params(module: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def params_by_submodules(
    module: torch.nn.Module,
    names: Iterable[str],
) -> Dict[str, int]:
    """Return ``{sub_name: num_params}`` plus ``total``.

    ``names`` must be attribute names on ``module`` (nested dotted access NOT
    supported to keep things explicit).
    """
    out: Dict[str, int] = {}
    for n in names:
        if not hasattr(module, n):
            out[n] = 0
            continue
        sub = getattr(module, n)
        out[n] = count_params(sub)
    out["total"] = count_params(module)
    return out


# ---------------------------------------------------------------------------
# FLOPs helpers
# ---------------------------------------------------------------------------


def profile_macs(module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Optional[Dict[str, int]]:
    """Run thop.profile and return ``{macs, params}`` or None if unavailable."""
    if _thop_profile is None:
        return None
    try:
        macs, params = _thop_profile(module, inputs=inputs, verbose=False)
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}
    return {"macs": int(macs), "params": int(params)}


def analytic_reweighting_macs(num_classes: int, channels: int, seq_len: int) -> int:
    """MACs for the dynamic 1x1 depthwise Conv1D feature reweighting.

    Per-sample: for each of ``N`` classes, a pure channel-wise multiply over the
    query feature map of shape ``(C, L')``. Elementwise mul counts as 1 MAC in
    thop convention, so the total is ``N * C * L'``.
    """
    return int(num_classes * channels * seq_len)


def analytic_topm_macs(
    num_classes: int,
    seq_len: int,
    channels: int,
    num_heads: int,
    num_layers: int,
) -> int:
    """MACs for ``num_layers`` SimplifiedTopMAttention layers applied per-class.

    Per layer per class (batch=1):
    - QKV projection: ``L' * C * 3C``
    - Attention scores (QK^T): ``H * L'^2 * (C/H)`` = ``L'^2 * C``
    - Softmax top-m gating (negligible MAC)
    - Attention output (attn @ V): ``L'^2 * C``
    - Output projection: ``L' * C * C``
    Total per class per layer: ``3 L' C^2 + 2 L'^2 C + L' C^2 = 4 L' C^2 + 2 L'^2 C``
    """
    per_layer_per_class = 4 * seq_len * channels * channels + 2 * seq_len * seq_len * channels
    return int(num_classes * num_layers * per_layer_per_class)


def analytic_crossclass_macs(
    num_classes: int,
    channels: int,
    num_layers: int,
) -> int:
    """MACs for Cross-Class MHSA over the ``(N, C)`` class-feature tensor.

    Per layer (batch=1):
    - QKV: ``N * C * 3C``
    - Attn scores: ``N^2 * C``
    - Attn output: ``N^2 * C``
    - Output proj: ``N * C * C``
    Total per layer: ``4 N C^2 + 2 N^2 C``
    """
    per_layer = 4 * num_classes * channels * channels + 2 * num_classes * num_classes * channels
    return int(num_layers * per_layer)


def analytic_classifier_macs(num_classes: int, channels: int) -> int:
    """MLP classifier: (C -> C/2 -> C/4 -> 1) applied per class.

    Per class: ``C*(C/2) + (C/2)*(C/4) + (C/4)*1``
    """
    c = channels
    per_class = c * (c // 2) + (c // 2) * (c // 4) + (c // 4) * 1
    return int(num_classes * per_class)


# ---------------------------------------------------------------------------
# Disk / tensor size
# ---------------------------------------------------------------------------


def tensor_size_mb(tensor: torch.Tensor) -> float:
    return round(tensor.element_size() * tensor.numel() / 1024**2, 4)


def file_size_mb(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    return round(os.path.getsize(path) / 1024**2, 4)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def dump_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def pretty_print(d: Dict[str, Any], title: str = "") -> None:
    if title:
        bar = "=" * max(40, len(title) + 4)
        print(bar)
        print(f" {title}")
        print(bar)
    print(json.dumps(d, indent=2, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def result_path(name: str) -> str:
    return os.path.join(RESULTS_DIR, name)


def artifact_path(name: str) -> str:
    return os.path.join(ARTIFACTS_DIR, name)
