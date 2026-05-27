import argparse
import contextlib
import io
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from data.multi_tab_generator import CombinationSampler


DATA_ROOT = Path("/data/datasets/benchdata/MMF_datasets/datasets")
DEFAULT_SOURCE_ROOT = DATA_ROOT / "OW_split"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "Overlap_statisctic"
BASE_CLASSES = list(range(60))
BIN_WIDTH = 0.2

BIN_LABELS = [
    "[0.0,0.2)",
    "[0.2,0.4)",
    "[0.4,0.6)",
    "[0.6,0.8)",
    "[0.8,1.0)",
    ">=1.0",
]

PROFILE_SCALES = {
    "paper": {
        "train": (35000, 7),
        "test": (17500, 3),
    },
    "smoke": {
        "train": (10, 2),
        "test": (6, 1),
    },
}


def overlap_tag(value: float) -> str:
    return f"ov0p{int(round(value * 10))}"


def bin_index(ratio: float) -> int:
    if ratio >= 1.0 - 1e-6:
        return len(BIN_LABELS) - 1
    return min(int(ratio / BIN_WIDTH), len(BIN_LABELS) - 2)


def load_duration_index(source_root: Path, classes: Iterable[int], splits: Iterable[str]) -> Dict[str, Dict[int, List[float]]]:
    index: Dict[str, Dict[int, List[float]]] = {}
    for split in splits:
        split_index: Dict[int, List[float]] = {}
        for class_id in tqdm(list(classes), desc=f"index durations {split}"):
            class_dir = source_root / split / str(class_id)
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")

            durations = []
            for path in sorted(class_dir.glob("*.pkl")):
                with path.open("rb") as f:
                    payload = pickle.load(f)
                time = payload["time"]
                duration = float(time[-1] - time[0])
                if duration > 0:
                    durations.append(duration)

            if not durations:
                raise ValueError(f"No valid samples for class {class_id} split {split}")
            split_index[class_id] = durations
        index[split] = split_index
    return index


def actual_pair_ratios(durations: List[float], configured_ratios: List[float]) -> Tuple[List[float], List[float]]:
    start_times = [0.0]
    for i in range(1, len(durations)):
        prev_start = start_times[i - 1]
        prev_duration = durations[i - 1]
        configured_ratio = configured_ratios[i - 1]
        start_times.append(prev_start + prev_duration * (1.0 - configured_ratio))

    actual = []
    for i in range(len(durations) - 1):
        a0 = start_times[i]
        a1 = start_times[i] + durations[i]
        b0 = start_times[i + 1]
        b1 = start_times[i + 1] + durations[i + 1]
        overlap = max(0.0, min(a1, b1) - max(a0, b0))
        denom = max(min(durations[i], durations[i + 1]), 1e-12)
        actual.append(min(overlap / denom, 1.0))
    return actual, start_times


def update_stats(stats: Dict, ratios: List[float]) -> None:
    if not ratios:
        return

    stats["num_samples"] += 1
    stats["total_pairs"] += len(ratios)
    max_ratio = max(ratios)
    stats["max_ratios"].append(max_ratio)

    if any(r > 0.5 for r in ratios):
        stats["sample_any_gt_0p5"] += 1
    if any(r >= 1.0 - 1e-6 for r in ratios):
        stats["sample_any_full"] += 1

    for ratio in ratios:
        stats["bin_counts"][bin_index(ratio)] += 1
        if ratio > 0.5:
            stats["pair_gt_0p5"] += 1
        if ratio >= 1.0 - 1e-6:
            stats["pair_full"] += 1


def empty_stats() -> Dict:
    return {
        "num_samples": 0,
        "total_pairs": 0,
        "bin_counts": np.zeros(len(BIN_LABELS), dtype=np.int64),
        "sample_any_gt_0p5": 0,
        "sample_any_full": 0,
        "pair_gt_0p5": 0,
        "pair_full": 0,
        "max_ratios": [],
    }


def finalize_stats(stats: Dict) -> Dict:
    bin_counts = stats["bin_counts"]
    total_samples = max(stats["num_samples"], 1)
    total_pairs = max(stats["total_pairs"], 1)
    max_arr = np.array(stats["max_ratios"], dtype=np.float64) if stats["max_ratios"] else np.array([0.0])

    return {
        "num_samples": int(stats["num_samples"]),
        "total_pairs": int(stats["total_pairs"]),
        "bin_counts": bin_counts.astype(int).tolist(),
        "bin_percent": (bin_counts / total_pairs * 100.0).tolist(),
        "sample_any_gt_0p5": int(stats["sample_any_gt_0p5"]),
        "sample_any_gt_0p5_percent": stats["sample_any_gt_0p5"] / total_samples * 100.0,
        "sample_any_full": int(stats["sample_any_full"]),
        "sample_any_full_percent": stats["sample_any_full"] / total_samples * 100.0,
        "pair_gt_0p5": int(stats["pair_gt_0p5"]),
        "pair_gt_0p5_percent": stats["pair_gt_0p5"] / total_pairs * 100.0,
        "pair_full": int(stats["pair_full"]),
        "pair_full_percent": stats["pair_full"] / total_pairs * 100.0,
        "max_real_ratio": {
            "mean": float(np.mean(max_arr)),
            "median": float(np.median(max_arr)),
            "p90": float(np.percentile(max_arr, 90)),
            "p99": float(np.percentile(max_arr, 99)),
            "max": float(np.max(max_arr)),
        },
    }


def generate_combinations(
    sampler: CombinationSampler,
    num_tabs: int,
    num_combinations: int,
    check_interval: int,
    balance_attempts: int,
    verbose: bool,
) -> List[Tuple[int, ...]]:
    if verbose:
        combinations, _ = sampler.generate_balanced_combinations(
            k=num_tabs,
            target_num_combinations=num_combinations,
            check_interval=check_interval,
            balance_attempts=balance_attempts,
        )
        return combinations

    with contextlib.redirect_stdout(io.StringIO()):
        combinations, _ = sampler.generate_balanced_combinations(
            k=num_tabs,
            target_num_combinations=num_combinations,
            check_interval=check_interval,
            balance_attempts=balance_attempts,
        )
    return combinations


def simulate_split(
    duration_index: Dict[int, List[float]],
    combinations: List[Tuple[int, ...]],
    samples_per_combo: int,
    max_overlap: float,
    progress: bool,
    desc: str,
) -> Dict:
    stats = empty_stats()
    iterator = combinations
    progress_bar = tqdm(total=len(combinations) * samples_per_combo, desc=desc, disable=not progress)

    try:
        for combo in iterator:
            for _ in range(samples_per_combo):
                durations = [random.choice(duration_index[class_id]) for class_id in combo]
                configured_ratios = [random.uniform(0.0, max_overlap) for _ in range(len(combo) - 1)]
                ratios, _ = actual_pair_ratios(durations, configured_ratios)
                update_stats(stats, ratios)
                progress_bar.update(1)
    finally:
        progress_bar.close()
    return stats


def simulate_overlap(
    duration_index: Dict[str, Dict[int, List[float]]],
    max_overlap: float,
    profile: str,
    seed: int,
    num_tabs: int,
    check_interval: int,
    balance_attempts: int,
    progress: bool,
    verbose_combinations: bool,
) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    sampler = CombinationSampler(num_classes=len(BASE_CLASSES), random_seed=seed)
    split_results: Dict[str, Dict] = {}
    raw_splits: Dict[str, Dict] = {}

    for split in ("train", "test"):
        num_combinations, samples_per_combo = PROFILE_SCALES[profile][split]
        combinations = generate_combinations(
            sampler=sampler,
            num_tabs=num_tabs,
            num_combinations=num_combinations,
            check_interval=check_interval,
            balance_attempts=balance_attempts,
            verbose=verbose_combinations,
        )
        raw_stats = simulate_split(
            duration_index=duration_index[split],
            combinations=combinations,
            samples_per_combo=samples_per_combo,
            max_overlap=max_overlap,
            progress=progress,
            desc=f"{overlap_tag(max_overlap)} {split}",
        )
        raw_splits[split] = raw_stats
        split_results[split] = finalize_stats(raw_stats)

    merged = empty_stats()
    for stats in raw_splits.values():
        merged["num_samples"] += stats["num_samples"]
        merged["total_pairs"] += stats["total_pairs"]
        merged["bin_counts"] += stats["bin_counts"]
        merged["sample_any_gt_0p5"] += stats["sample_any_gt_0p5"]
        merged["sample_any_full"] += stats["sample_any_full"]
        merged["pair_gt_0p5"] += stats["pair_gt_0p5"]
        merged["pair_full"] += stats["pair_full"]
        merged["max_ratios"].extend(stats["max_ratios"])
    split_results["all"] = finalize_stats(merged)
    return split_results


def write_text_report(path: Path, results: Dict) -> None:
    lines = []
    lines.append("=" * 80)
    lines.append("Simulated 4tab overlap statistics for revision extreme-interleaving analysis")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Statistic definition:")
    lines.append("  actual_ratio = actual adjacent overlap duration / min(D_prev, D_next)")
    lines.append("  simulation reads only single-trace durations and mirrors the generator's sampling order.")
    lines.append("  no synthesized query/support samples are written.")
    lines.append("")
    for ov, payload in results["overlaps"].items():
        lines.append("-" * 80)
        lines.append(f"[max_overlap = {ov}]")
        for split_name in ("train", "test", "all"):
            stats = payload[split_name]
            lines.append(f"  split={split_name} samples={stats['num_samples']} pairs={stats['total_pairs']}")
            lines.append(
                "    any pair >0.5: "
                f"{stats['sample_any_gt_0p5']} ({stats['sample_any_gt_0p5_percent']:.4f}%)"
            )
            lines.append(
                "    any pair full: "
                f"{stats['sample_any_full']} ({stats['sample_any_full_percent']:.4f}%)"
            )
            lines.append(
                "    pair >0.5: "
                f"{stats['pair_gt_0p5']} ({stats['pair_gt_0p5_percent']:.4f}%)"
            )
            lines.append(
                "    pair full: "
                f"{stats['pair_full']} ({stats['pair_full_percent']:.4f}%)"
            )
            lines.append("    pair-level actual_ratio distribution:")
            for label, count, pct in zip(BIN_LABELS, stats["bin_counts"], stats["bin_percent"]):
                lines.append(f"      {label:12s} {count:10d} ({pct:8.4f}%)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


from matplotlib.colors import LinearSegmentedColormap

PAPER_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "paper_teal_blue",
    ["#fbfdf8", "#dceec7", "#8dcec8", "#3480b8", "#163b73"]
)

def format_overlap_tick(tag: str) -> str:
    """
    Convert ov0p2 -> 0.2
            ov0p3 -> 0.3
    """
    if tag.startswith("ov0p"):
        return f"{int(tag[4:]) / 10:.1f}"
    return tag
def build_soft_ylgnbu():
    """
    Build a lighter version based on the original YlGnBu:
    - Preserve the overall visual style of the current figure.
    - Remove the darkest blue segment at the very top.
    """
    base = plt.cm.get_cmap("YlGnBu")
    colors = base(np.linspace(0.03, 0.88, 256))   # Avoid the deepest end so the blue is lighter.
    return LinearSegmentedColormap.from_list("soft_YlGnBu", colors)

def draw_heatmap(path: Path, results: Dict, cmap: str, title: str) -> None:
    overlap_keys = list(results["overlaps"].keys())
    matrix = np.array(
        [results["overlaps"][ov]["all"]["bin_percent"] for ov in overlap_keys],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(6, 4.6))

    # Use the custom light-blue version if soft_YlGnBu is requested.
    if cmap == "soft_YlGnBu":
        cmap_obj = build_soft_ylgnbu()
    else:
        cmap_obj = cmap

    image = ax.imshow(matrix, aspect="auto", cmap=cmap_obj)

    # x-axis.
    ax.set_xticks(np.arange(len(BIN_LABELS)))
    ax.set_xticklabels(BIN_LABELS, rotation=35, ha="right")

    # y-axis: convert ov0p2 to 0.2 / 0.3 / ...
    ax.set_yticks(np.arange(len(overlap_keys)))
    ax.set_yticklabels([format_overlap_tick(ov) for ov in overlap_keys])

    ax.set_xlabel("Actual adjacent-pair overlap ratio")
    ax.set_ylabel("Configured max_overlap")
    #ax.set_title(title)

    # Choose text color based on value intensity.
    threshold = np.nanmax(matrix) * 0.58 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value > threshold else "black"
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center",
                color=color, fontsize=10
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Pair-level percentage (%)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate actual overlap distributions without writing datasets.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profile", choices=sorted(PROFILE_SCALES), default="paper")
    parser.add_argument("--max-overlaps", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--num-tabs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-interval", type=int, default=20)
    parser.add_argument("--balance-attempts", type=int, default=20)
    parser.add_argument(
    "--cmap",
    default="soft_YlGnBu",
    help="Matplotlib colormap name. Use soft_YlGnBu for a lighter YlGnBu style."
)
    parser.add_argument("--title", default="Simulated actual overlap distribution under heavier interleaving")
    parser.add_argument("--output-prefix", default="simulated_overlap_distribution_4tab")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--verbose-combinations", action="store_true")
    args = parser.parse_args()

    if not args.source_root.exists():
        raise FileNotFoundError(f"Missing source root: {args.source_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    duration_index = load_duration_index(args.source_root, BASE_CLASSES, ("train", "test"))

    results = {
        "bin_labels": BIN_LABELS,
        "source_root": str(args.source_root),
        "profile": args.profile,
        "num_tabs": args.num_tabs,
        "seed": args.seed,
        "method": "duration-only simulation; no synthesized samples are written",
        "scale": PROFILE_SCALES[args.profile],
        "overlaps": {},
    }

    for max_overlap in args.max_overlaps:
        tag = overlap_tag(max_overlap)
        print(f"\n### Simulating {tag} (max_overlap={max_overlap})")
        results["overlaps"][tag] = simulate_overlap(
            duration_index=duration_index,
            max_overlap=max_overlap,
            profile=args.profile,
            seed=args.seed,
            num_tabs=args.num_tabs,
            check_interval=args.check_interval,
            balance_attempts=args.balance_attempts,
            progress=not args.no_progress,
            verbose_combinations=args.verbose_combinations,
        )

    json_path = args.output_dir / f"{args.output_prefix}.json"
    txt_path = args.output_dir / f"{args.output_prefix}.txt"
    png_path = args.output_dir / f"{args.output_prefix}_heatmap.png"
    pdf_path = args.output_dir / f"{args.output_prefix}_heatmap.pdf"

    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_text_report(txt_path, results)
    draw_heatmap(png_path, results, cmap=args.cmap, title=args.title)
    draw_heatmap(pdf_path, results, cmap=args.cmap, title=args.title)

    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
