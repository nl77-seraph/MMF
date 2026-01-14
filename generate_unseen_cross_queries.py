"""
生成跨区未见组合的 Query 样本（修复版）

修复点：
- 删除无用且易误导的 zones 字段
- 失败的组合不加入 combinations_seen，避免浪费组合空间
- 用 while + max_attempts 尽量凑够目标 num_queries
- 严格跨区采样时打乱 zone 顺序，避免 tab 位置固定对应 zone
- 预估唯一组合容量 capacity；不允许重复且目标>capacity 时自动降目标并告警
"""

import os
import sys
import json
import uuid
import argparse
import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.multi_tab_generator import TimeBasedMerger  # noqa: E402


def split_into_zones(num_classes: int, num_tabs: int) -> Dict[int, List[int]]:
    """与 disjoint 生成逻辑一致的均匀分区。"""
    classes_per_zone = num_classes // num_tabs
    remainder = num_classes % num_tabs
    zones: Dict[int, List[int]] = {}
    current = 0
    for zone_id in range(num_tabs):
        zone_size = classes_per_zone + (1 if zone_id < remainder else 0)
        zones[zone_id] = list(range(current, current + zone_size))
        current += zone_size
    return zones


def cross_zone_capacity(zones: Dict[int, List[int]]) -> int:
    """严格跨区唯一组合数上限：各 zone 类别数乘积。"""
    cap = 1
    for z in zones.values():
        cap *= len(z)
    return cap


class CrossZoneQueryGenerator:
    """跨区 Query 生成器。"""

    def __init__(
        self,
        source_root: str,
        output_root: str,
        num_classes: int = 60,
        overlap_range: Tuple[float, float] = (0.0, 0.4),
        seed: int = 42,
    ):
        self.source_root = source_root
        self.output_root = output_root
        self.num_classes = num_classes
        self.overlap_range = overlap_range

        self.rng = random.Random(seed)
        np.random.seed(seed)

        self.merger = TimeBasedMerger(overlap_range)

    def _load_single_label_sample(self, class_id: int, split: str) -> Dict:
        class_dir = os.path.join(self.source_root, split, str(class_id))
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"类别目录不存在：{class_dir}")

        files = [f for f in os.listdir(class_dir) if f.endswith(".pkl")]
        if not files:
            raise FileNotFoundError(f"类别{class_id}在{split}下无样本：{class_dir}")

        file_name = self.rng.choice(files)
        path = os.path.join(class_dir, file_name)

        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        return {
            "time": data["time"],
            "data": data["data"],
            "label": data["label"],
            "source_file": file_name,
            "source_path": path,
        }

    def _sample_cross_zone_combo(self, zones: Dict[int, List[int]]) -> Tuple[int, ...]:
        """严格跨区：每个 tab 来自不同 zone；打乱 zone 顺序避免 tab 位置固定对应 zone。"""
        zone_order = list(zones.keys())
        self.rng.shuffle(zone_order)
        combo = [self.rng.choice(zones[zid]) for zid in zone_order]
        return tuple(combo)

    def generate_queries(
        self,
        num_tabs: int,
        num_queries: int,
        split: str = "train",
        allow_duplicate_combos: bool = False,
    ) -> str:
        zones = split_into_zones(self.num_classes, num_tabs)
        os.makedirs(self.output_root, exist_ok=True)

        out_dir = os.path.join(self.output_root, f"{num_tabs}tab", split)
        query_dir = os.path.join(out_dir, "query_data")
        os.makedirs(query_dir, exist_ok=True)

        # 唯一组合容量估计与有效目标
        cap = cross_zone_capacity(zones)
        target = num_queries
        if not allow_duplicate_combos and num_queries > cap:
            print(f"[WARN] 目标唯一组合数 {num_queries} 超过容量 {cap}，将有效目标降为 {cap}（否则必然生成不满）。")
            target = cap

        combinations_seen = set()  # 只记录“成功产出”的组合
        query_files: List[str] = []
        class_usage = Counter()

        # 统计拆分
        failures_merge = 0
        failures_validate = 0
        collisions = 0

        print(f"\n{'='*60}")
        print(f"生成跨区 Query 样本：{num_tabs}-tab | split={split}")
        print(f"{'='*60}")
        print(f"  - 目标样本数: {num_queries} | 有效目标: {target}")
        print(f"  - 允许重复组合: {'是' if allow_duplicate_combos else '否'}")
        print(f"  - 区域数: {num_tabs} | 严格跨区: 是（zone顺序随机）")
        print(f"  - 唯一组合容量(严格跨区): {cap}")
        print(f"  - 输出目录: {out_dir}")

        # 尽量凑够 target，避免固定循环导致“失败=少产出”
        # max_attempts 需要比 target 大不少，尤其在接近容量上限时碰撞会很多
        if allow_duplicate_combos:
            max_attempts = max(5000, target * 20)
        else:
            max_attempts = max(10000, target * 80)

        attempts = 0
        pbar = tqdm(total=target, desc="生成query")
        while len(query_files) < target and attempts < max_attempts:
            attempts += 1

            combo = self._sample_cross_zone_combo(zones)

            # 唯一性：只拒绝“成功产出过”的 combo
            if (not allow_duplicate_combos) and (combo in combinations_seen):
                collisions += 1
                continue

            # 合成
            try:
                traces = [self._load_single_label_sample(cls, split) for cls in combo]
                merged = self.merger.merge_traces_with_overlap(traces)
            except Exception as e:  # pylint: disable=broad-except
                # 合成/加载失败
                failures_merge += 1
                # 只在偶发时打印，避免刷屏（你也可以去掉这行限制）
                if failures_merge <= 10:
                    print(f"[WARN] 合成失败(前10条打印): combo={combo}, err={e}")
                continue

            if not self.merger.validate_merge(merged, verbose=False):
                failures_validate += 1
                continue

            # 到这里才算成功：保存 + 记录 seen
            labels_str = "_".join(map(str, combo))
            fname = f"unseen_{labels_str}_{uuid.uuid4().hex[:8]}.pkl"
            fpath = os.path.join(query_dir, fname)

            import pickle

            with open(fpath, "wb") as f:
                pickle.dump(
                    {
                        "time": merged["time"],
                        "data": merged["data"],
                        "labels": merged["labels"],
                        "metadata": merged["metadata"],
                        # "zones": ...  # 已删除：你们不会用且之前容易误导
                    },
                    f,
                )

            query_files.append(fname)
            combinations_seen.add(combo)
            for cls in combo:
                class_usage[cls] += 1

            pbar.update(1)

        pbar.close()

        # 保存索引与统计
        index_path = os.path.join(out_dir, f"unseen_queries_{split}.json")
        with open(index_path, "w") as f:
            json.dump(query_files, f, indent=2)

        stats = {
            "num_tabs": num_tabs,
            "split": split,
            "requested_queries": num_queries,
            "effective_target": target,
            "actual_queries": len(query_files),
            "allow_duplicate_combos": allow_duplicate_combos,
            "unique_combos": len(combinations_seen),
            "capacity_unique_combos": cap,
            "attempts": attempts,
            "max_attempts": max_attempts,
            "collisions": collisions,
            "failures_merge": failures_merge,
            "failures_validate": failures_validate,
            "failures_total": int(collisions + failures_merge + failures_validate),
            "class_usage": {int(k): int(v) for k, v in class_usage.items()},
            "zone_split": {int(z): v for z, v in zones.items()},
        }

        stats_path = os.path.join(out_dir, f"statistics_{split}.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n[OK] 完成 {split}: 实际样本 {len(query_files)} / 有效目标 {target}（用户请求 {num_queries}）")
        if len(query_files) < target:
            print(f"[WARN] 未达到有效目标：可能因 max_attempts 限制或合成/校验失败率过高。")
        print(f"  - attempts: {attempts} / max_attempts: {max_attempts}")
        print(f"  - collisions(唯一性碰撞): {collisions}")
        print(f"  - failures_merge: {failures_merge} | failures_validate: {failures_validate}")
        print(f"  - 索引: {index_path}")
        print(f"  - 统计: {stats_path}")

        return out_dir


def main():
    parser = argparse.ArgumentParser(description="生成跨区未见组合 Query 数据集（修复版）")
    parser.add_argument("--tabs", type=int, required=True, choices=[2, 3, 4, 5], help="tab 数量")
    parser.add_argument("--num-queries", type=int, default=5000, help="生成的 Query 数量")
    parser.add_argument("--input", type=str, default="../datasets/MMFOW", help="单标签数据根目录（需含 train/test）")
    parser.add_argument(
        "--output",
        type=str,
        default="../datasets/unseen_cross_queries",
        help="输出根目录（会在下级创建 {tabs}tab/train|test）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="生成哪个 split")
    parser.add_argument(
        "--allow-duplicate-combos",
        action="store_true",
        help="允许重复组合（默认唯一组合；若目标超过容量会自动降为容量）",
    )
    args = parser.parse_args()

    gen = CrossZoneQueryGenerator(
        source_root=args.input,
        output_root=args.output,
        num_classes=60,
        overlap_range=(0.0, 0.4),
        seed=args.seed,
    )
    gen.generate_queries(
        num_tabs=args.tabs,
        num_queries=args.num_queries,
        split=args.split,
        allow_duplicate_combos=args.allow_duplicate_combos,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\n" + "=" * 60)
        print("跨区未见组合 Query 生成器（修复版）")
        print("=" * 60)
        print("\n示例：生成 4-tab train 跨区 query 5000 条")
        print("python generate_unseen_cross_queries.py --tabs 4 --num-queries 5000 \\")
        print("    --input /root/datasets/MMFOW --output /root/datasets/unseen_cross_queries --split train")
        print("\n示例：生成 4-tab test 跨区 query 2000 条（可重复组合）")
        print("python generate_unseen_cross_queries.py --tabs 4 --num-queries 2000 --split test --allow-duplicate-combos")
    else:
        main()
