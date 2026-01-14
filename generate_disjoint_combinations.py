"""
区域分离组合数据集生成脚本
训练集：只生成同区域内的组合（如 [0,0,0,0], [1,1,1,1], [2,2,2,2], [3,3,3,3]）
测试集：只生成全跨区的组合（如 [0,1,2,3]）
用于测试模型对未见过组合模式的泛化能力

完全独立的实现，不修改原有的 MultiTabDatasetGenerator
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import json
import random
import numpy as np
import shutil
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict
from tqdm import tqdm
import uuid

from data.multi_tab_generator import TimeBasedMerger


class DisjointZoneCombinationSampler:
    """
    区域分离的组合采样器
    支持训练集（同区域）和测试集（跨区域）的组合生成
    """
    
    def __init__(self, num_classes: int = 60, random_seed: int = 42):
        """
        Args:
            num_classes: 类别总数（默认60，对应0-59）
            random_seed: 随机种子
        """
        self.num_classes = num_classes
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def split_into_zones(self, num_tabs: int) -> Dict[int, List[int]]:
        """
        将类别分成num_tabs个区域
        
        Args:
            num_tabs: 区域数量（也是tab数量）
            
        Returns:
            zones: {zone_id: [class_ids]}
        """
        classes_per_zone = self.num_classes // num_tabs
        remainder = self.num_classes % num_tabs
        
        zones = {}
        current_idx = 0
        
        for zone_id in range(num_tabs):
            # 如果有余数，前面的区域多分配一个类别
            zone_size = classes_per_zone + (1 if zone_id < remainder else 0)
            zones[zone_id] = list(range(current_idx, current_idx + zone_size))
            current_idx += zone_size
        
        return zones
    
    def generate_train_combinations(
        self, 
        num_tabs: int,
        num_combinations_per_zone: int,
        max_std_ratio: float = 0.15
    ) -> Tuple[List[Tuple[int]], Dict]:
        """
        生成训练集组合：所有tab来自同一个区域，且每个组合中类别不重复
        例如 4-tab: [11,12,13,14], [8,3,9,0] 等（类别不重复的有序排列）
        最大组合数 = A(n,k) = n!/(n-k)! 其中n是区域内类别数，k是tab数
        
        Args:
            num_tabs: tab数量
            num_combinations_per_zone: 每个区域期望的组合数（如果超过A(n,k)则使用A(n,k)）
            max_std_ratio: 区域内类别均衡的最大标准差比例
            
        Returns:
            combinations: 组合列表
            statistics: 统计信息
        """
        print(f"\n{'='*60}")
        print(f"生成训练集组合 (同区域模式，类别不重复)")
        print(f"{'='*60}")
        print(f"  - Tab数: {num_tabs}")
        print(f"  - 区域数: {num_tabs}")
        print(f"  - 每区域期望组合数: {num_combinations_per_zone}")
        
        # 分区
        zones = self.split_into_zones(num_tabs)
        
        print(f"\n区域划分:")
        for zone_id, classes in zones.items():
            # 计算该区域的最大组合数 A(n,k)
            n = len(classes)
            k = num_tabs
            if k > n:
                max_combos = 0
                print(f"  Zone {zone_id}: 类别 {classes[0]}-{classes[-1]} ({n}个类别) - 无法生成{k}-tab组合！")
            else:
                # A(n,k) = n!/(n-k)!
                max_combos = 1
                for i in range(n, n-k, -1):
                    max_combos *= i
                print(f"  Zone {zone_id}: 类别 {classes[0]}-{classes[-1]} ({n}个类别) - 最大组合数 A({n},{k})={max_combos}")
        
        print(f"  - 说明: 组合中类别不重复，视为有序排列")
        
        all_combinations = []
        class_counts = Counter()
        zone_stats = {}
        
        # 对每个区域生成组合
        for zone_id, zone_classes in zones.items():
            n = len(zone_classes)
            k = num_tabs
            
            # 计算该区域的最大组合数 A(n,k)
            if k > n:
                print(f"\n[SKIP] Zone {zone_id}: tab数({k}) > 类别数({n})，无法生成不重复组合")
                zone_stats[zone_id] = {
                    'num_combinations': 0,
                    'class_range': [zone_classes[0], zone_classes[-1]],
                    'error': 'insufficient_classes'
                }
                continue
            
            max_combos = 1
            for i in range(n, n-k, -1):
                max_combos *= i
            
            # 调整目标组合数
            actual_target = min(num_combinations_per_zone, max_combos)
            
            if actual_target < num_combinations_per_zone:
                print(f"\n[WARNING] Zone {zone_id}: 期望{num_combinations_per_zone}个组合，但最大只能生成{max_combos}个")
                print(f"          将生成{actual_target}个组合")
            
            print(f"\n生成 Zone {zone_id} 的组合 (目标: {actual_target}个)...")
            zone_combinations = set()
            zone_class_counts = Counter()
            
            attempts = 0
            max_attempts = actual_target * 1000
            
            while len(zone_combinations) < actual_target and attempts < max_attempts:
                # 从当前区域随机选择num_tabs个类别（不可重复）
                selected = np.random.choice(zone_classes, size=num_tabs, replace=False)
                combo = tuple(selected.tolist())
                
                if combo not in zone_combinations:
                    zone_combinations.add(combo)
                    for cls in combo:
                        zone_class_counts[cls] += 1
                        class_counts[cls] += 1
                
                attempts += 1
                
                # 定期检查均衡性
                if len(zone_combinations) > 0 and len(zone_combinations) % 100 == 0:
                    counts = [zone_class_counts[c] for c in zone_classes]
                    mean_count = np.mean(counts) if counts else 0
                    std_count = np.std(counts) if counts else 0
                    std_ratio = std_count / mean_count if mean_count > 0 else 0
                    
                    # 如果不均衡，优先选择使用次数少的类别
                    if std_ratio > max_std_ratio:
                        # 找出使用次数最少的类别
                        underrepresented = sorted(zone_classes, key=lambda c: zone_class_counts[c])[:max(num_tabs, len(zone_classes)//2)]
                        
                        # 强制生成包含这些类别的组合
                        for _ in range(10):
                            # 优先从欠代表类别中选择，但确保不重复
                            if len(underrepresented) >= num_tabs:
                                # 全部从欠代表类别中选
                                selected = np.random.choice(underrepresented, size=num_tabs, replace=False)
                            else:
                                # 混合选择，确保不重复
                                n_under = len(underrepresented)
                                selected_under = underrepresented  # 全部欠代表类别
                                
                                # 从其他类别中选择
                                other_classes = [c for c in zone_classes if c not in selected_under]
                                n_need = num_tabs - n_under
                                
                                if len(other_classes) >= n_need:
                                    selected_other = np.random.choice(other_classes, size=n_need, replace=False)
                                    selected = np.concatenate([selected_under, selected_other])
                                else:
                                    # 类别数不足，跳过均衡补偿
                                    break
                            
                            np.random.shuffle(selected)
                            combo = tuple(selected.tolist())
                            
                            if combo not in zone_combinations:
                                zone_combinations.add(combo)
                                for cls in combo:
                                    zone_class_counts[cls] += 1
                                    class_counts[cls] += 1
                                break
            
            zone_combinations_list = list(zone_combinations)
            all_combinations.extend(zone_combinations_list)
            
            # 统计区域信息
            counts = [zone_class_counts[c] for c in zone_classes]
            zone_stats[zone_id] = {
                'num_combinations': len(zone_combinations_list),
                'target_combinations': num_combinations_per_zone,
                'max_possible_combinations': max_combos,
                'class_range': [zone_classes[0], zone_classes[-1]],
                'num_classes': len(zone_classes),
                'mean_frequency': float(np.mean(counts)) if counts else 0,
                'std_frequency': float(np.std(counts)) if counts else 0,
                'std_mean_ratio': float(np.std(counts) / np.mean(counts)) if np.mean(counts) > 0 else 0,
                'min_frequency': int(np.min(counts)) if counts else 0,
                'max_frequency': int(np.max(counts)) if counts else 0
            }
            
            print(f"  [OK] Zone {zone_id}: 生成{len(zone_combinations_list)}个组合 (最大{max_combos}), "
                  f"均衡度={zone_stats[zone_id]['std_mean_ratio']:.4f}")
        
        # 全局统计
        counts = [class_counts[i] for i in range(self.num_classes)]
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        statistics = {
            'mode': 'train_same_zone',
            'num_tabs': num_tabs,
            'num_zones': num_tabs,
            'num_combinations': len(all_combinations),
            'zones': zone_stats,
            'global_class_distribution': {int(k): int(v) for k, v in class_counts.items()},
            'global_mean_frequency': float(mean_count),
            'global_std_frequency': float(std_count),
            'global_std_mean_ratio': float(std_count / mean_count) if mean_count > 0 else 0
        }
        
        print(f"\n[OK] 训练集组合生成完成:")
        print(f"  - 总组合数: {len(all_combinations)}")
        print(f"  - 期望组合数: {num_tabs * num_combinations_per_zone}")
        print(f"  - 全局均衡度: std/mean={statistics['global_std_mean_ratio']:.4f}")
        print(f"  - 示例组合 (类别不重复): {all_combinations[:3]}")
        
        # 验证组合中没有重复类别
        has_duplicates = sum(1 for combo in all_combinations if len(combo) != len(set(combo)))
        if has_duplicates > 0:
            print(f"  [WARNING] 发现{has_duplicates}个包含重复类别的组合！")
        else:
            print(f"  [OK] 所有组合均无重复类别")
        
        return all_combinations, statistics
    
    def generate_test_combinations(
        self,
        num_tabs: int,
        num_combinations: int,
        allow_duplicate_zones: bool = False
    ) -> Tuple[List[Tuple[int]], Dict]:
        """
        生成测试集组合：每个tab来自不同区域（全跨区）
        例如 4-tab: [0,1,2,3] 模式，每个位置的tab来自不同区域
        
        Args:
            num_tabs: tab数量
            num_combinations: 要生成的组合数
            allow_duplicate_zones: 是否允许多个tab来自同一区域（False=严格跨区）
            
        Returns:
            combinations: 组合列表
            statistics: 统计信息
        """
        print(f"\n{'='*60}")
        print(f"生成测试集组合 (全跨区模式)")
        print(f"{'='*60}")
        print(f"  - Tab数: {num_tabs}")
        print(f"  - 区域数: {num_tabs}")
        print(f"  - 目标组合数: {num_combinations}")
        print(f"  - 严格跨区: {'是' if not allow_duplicate_zones else '否'}")
        
        # 分区
        zones = self.split_into_zones(num_tabs)
        
        print(f"\n区域划分:")
        for zone_id, classes in zones.items():
            print(f"  Zone {zone_id}: 类别 {classes[0]}-{classes[-1]} ({len(classes)}个类别)")
        
        combinations = set()
        class_counts = Counter()
        zone_usage_at_position = defaultdict(lambda: defaultdict(int))
        
        attempts = 0
        max_attempts = num_combinations * 10000
        
        print(f"\n开始生成跨区组合...")
        
        while len(combinations) < num_combinations and attempts < max_attempts:
            combo = []
            
            if not allow_duplicate_zones:
                # 严格模式：每个tab必须来自不同区域
                # 打乱区域顺序，然后从每个区域选一个类别
                zone_order = list(range(num_tabs))
                random.shuffle(zone_order)
                
                for zone_id in zone_order:
                    selected_class = random.choice(zones[zone_id])
                    combo.append(selected_class)
            else:
                # 宽松模式：确保至少包含2个以上不同区域
                for position in range(num_tabs):
                    zone_id = random.randint(0, num_tabs - 1)
                    selected_class = random.choice(zones[zone_id])
                    combo.append(selected_class)
                
                # 检查是否有足够的区域多样性
                used_zones = set()
                for cls in combo:
                    for zone_id, zone_classes in zones.items():
                        if cls in zone_classes:
                            used_zones.add(zone_id)
                            break
                
                if len(used_zones) < max(2, num_tabs // 2):
                    attempts += 1
                    continue
            
            combo_tuple = tuple(combo)
            
            if combo_tuple not in combinations:
                combinations.add(combo_tuple)
                
                # 统计每个位置使用的区域
                for position, cls in enumerate(combo):
                    for zone_id, zone_classes in zones.items():
                        if cls in zone_classes:
                            zone_usage_at_position[position][zone_id] += 1
                            break
                    class_counts[cls] += 1
            
            attempts += 1
            
            # 进度显示
            if len(combinations) % 500 == 0 and len(combinations) > 0:
                print(f"  进度: {len(combinations)}/{num_combinations}")
        
        combinations_list = list(combinations)
        
        # 验证跨区性质
        cross_zone_verification = []
        for combo in combinations_list[:10]:  # 验证前10个
            zones_in_combo = []
            for cls in combo:
                for zone_id, zone_classes in zones.items():
                    if cls in zone_classes:
                        zones_in_combo.append(zone_id)
                        break
            cross_zone_verification.append({
                'combination': combo,
                'zones': zones_in_combo,
                'unique_zones': len(set(zones_in_combo))
            })
        
        # 统计
        counts = [class_counts[i] for i in range(self.num_classes)]
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        # 统计每个位置的区域分布
        position_zone_stats = {}
        for position in range(num_tabs):
            position_zone_stats[position] = dict(zone_usage_at_position[position])
        
        statistics = {
            'mode': 'test_cross_zone',
            'num_tabs': num_tabs,
            'num_zones': num_tabs,
            'num_combinations': len(combinations_list),
            'strict_cross_zone': not allow_duplicate_zones,
            'attempts_used': attempts,
            'position_zone_distribution': position_zone_stats,
            'global_class_distribution': {int(k): int(v) for k, v in class_counts.items()},
            'global_mean_frequency': float(mean_count),
            'global_std_frequency': float(std_count),
            'global_std_mean_ratio': float(std_count / mean_count) if mean_count > 0 else 0,
            'verification_samples': cross_zone_verification
        }
        
        print(f"\n[OK] 测试集组合生成完成:")
        print(f"  - 总组合数: {len(combinations_list)}")
        print(f"  - 全局均衡度: std/mean={statistics['global_std_mean_ratio']:.4f}")
        print(f"  - 示例组合与区域分布:")
        for item in cross_zone_verification[:3]:
            print(f"      {item['combination']} -> Zones {item['zones']} (跨{item['unique_zones']}个区)")
        
        # 验证组合中没有重复类别
        has_duplicates = sum(1 for combo in combinations_list if len(combo) != len(set(combo)))
        if has_duplicates > 0:
            print(f"  [WARNING] 发现{has_duplicates}个包含重复类别的组合！")
        else:
            print(f"  [OK] 所有组合均无重复类别")
        
        return combinations_list, statistics


class DisjointZoneDatasetGenerator:
    """
    区域分离的数据集生成器
    复用原有的 TimeBasedMerger 和数据加载逻辑
    """
    
    def __init__(
        self,
        source_root: str,
        output_root: str,
        num_classes: int = 60,
        overlap_range: Tuple[float, float] = (0.0, 0.2),
        random_seed: int = 42
    ):
        self.source_root = source_root
        self.output_root = output_root
        self.num_classes = num_classes
        self.overlap_range = overlap_range
        self.random_seed = random_seed
        
        # 初始化组件
        self.sampler = DisjointZoneCombinationSampler(num_classes, random_seed)
        self.merger = TimeBasedMerger(overlap_range)
        
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def load_single_label_sample(self, class_id: int, split: str) -> Dict:
        """
        加载单标签样本（复用原有逻辑）
        """
        class_dir = os.path.join(self.source_root, split, str(class_id))
        
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"类别目录不存在: {class_dir}")
        
        files = [f for f in os.listdir(class_dir) if f.endswith('.pkl')]
        
        if len(files) == 0:
            raise ValueError(f"类别{class_id}在{split}中没有样本")
        
        selected_file = random.choice(files)
        file_path = os.path.join(class_dir, selected_file)
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        return {
            'time': data['time'],
            'data': data['data'],
            'label': data['label'],
            'source_file': selected_file,
            'source_path': file_path
        }
    
    def generate_disjoint_dataset(
        self,
        num_tabs: int,
        train_combinations_per_zone: int,
        test_combinations: int,
        samples_per_combo: int,
        dataset_name: str,
        add_ow_class: bool = False
    ) -> Dict[str, str]:
        """
        生成区域分离的训练集和测试集
        
        Args:
            num_tabs: tab数量
            train_combinations_per_zone: 每个区域生成的训练组合数
            test_combinations: 测试集组合数
            samples_per_combo: 每个组合生成的样本数
            dataset_name: 数据集名称
            add_ow_class: 是否添加OW类别
            
        Returns:
            output_paths: {'train': path, 'test': path}
        """
        print(f"\n{'='*70}")
        print(f"生成区域分离数据集: {dataset_name}")
        print(f"{'='*70}")
        print(f"配置:")
        print(f"  - Tab数: {num_tabs}")
        print(f"  - 训练集: 每区域{train_combinations_per_zone}个组合 × {num_tabs}区域 = {train_combinations_per_zone * num_tabs}个组合")
        print(f"  - 测试集: {test_combinations}个组合 (全跨区)")
        print(f"  - 每组样本数: {samples_per_combo}")
        print(f"  - 训练集总样本: {train_combinations_per_zone * num_tabs * samples_per_combo}")
        print(f"  - 测试集总样本: {test_combinations * samples_per_combo}")
        print(f"  - 添加OW类别: {'是' if add_ow_class else '否'}")
        
        output_paths = {}
        
        # 生成训练集
        print(f"\n{'='*70}")
        print(f"[1/2] 生成训练集（同区域组合）")
        print(f"{'='*70}")
        
        train_combinations, train_combo_stats = self.sampler.generate_train_combinations(
            num_tabs=num_tabs,
            num_combinations_per_zone=train_combinations_per_zone
        )
        
        train_output = self._generate_samples(
            combinations=train_combinations,
            num_tabs=num_tabs,
            samples_per_combo=samples_per_combo,
            split='train',
            dataset_name=dataset_name,
            combo_stats=train_combo_stats,
            add_ow_class=add_ow_class
        )
        
        output_paths['train'] = train_output
        
        # 生成测试集
        print(f"\n{'='*70}")
        print(f"[2/2] 生成测试集（全跨区组合）")
        print(f"{'='*70}")
        
        test_combinations, test_combo_stats = self.sampler.generate_test_combinations(
            num_tabs=num_tabs,
            num_combinations=test_combinations,
            allow_duplicate_zones=False
        )
        
        test_output = self._generate_samples(
            combinations=test_combinations,
            num_tabs=num_tabs,
            samples_per_combo=samples_per_combo,
            split='test',
            dataset_name=dataset_name,
            combo_stats=test_combo_stats,
            add_ow_class=add_ow_class
        )
        
        output_paths['test'] = test_output
        
        # 保存全局配置
        config_file = os.path.join(self.output_root, dataset_name, 'dataset_config.json')
        with open(config_file, 'w') as f:
            json.dump({
                'dataset_name': dataset_name,
                'num_tabs': num_tabs,
                'mode': 'disjoint_zone',
                'train': {
                    'combinations_per_zone': train_combinations_per_zone,
                    'total_combinations': len(train_combinations),
                    'samples_per_combo': samples_per_combo,
                    'total_samples': len(train_combinations) * samples_per_combo
                },
                'test': {
                    'combinations': test_combinations,
                    'total_combinations': len(test_combinations),
                    'samples_per_combo': samples_per_combo,
                    'total_samples': len(test_combinations) * samples_per_combo
                },
                'overlap_range': self.overlap_range,
                'random_seed': self.random_seed,
                'add_ow_class': add_ow_class
            }, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"[OK] 区域分离数据集生成完成!")
        print(f"{'='*70}")
        print(f"  - 训练集: {train_output}")
        print(f"  - 测试集: {test_output}")
        print(f"  - 配置文件: {config_file}")
        
        return output_paths
    
    def _generate_samples(
        self,
        combinations: List[Tuple[int]],
        num_tabs: int,
        samples_per_combo: int,
        split: str,
        dataset_name: str,
        combo_stats: Dict,
        add_ow_class: bool = False
    ) -> str:
        """
        根据组合生成样本（核心生成逻辑，复用原有merger）
        """
        # 创建输出目录
        output_dir = os.path.join(self.output_root, dataset_name, split)
        query_dir = os.path.join(output_dir, "query_data")
        support_dir = os.path.join(output_dir, "support_data")
        
        os.makedirs(query_dir, exist_ok=True)
        os.makedirs(support_dir, exist_ok=True)
        
        # 如果启用OW模式
        if add_ow_class:
            print(f"\n[OW模式] 在每个组合的随机位置添加类别95...")
            combinations_with_ow = []
            for combo in combinations:
                combo_list = list(combo)
                insert_pos = random.randint(0, len(combo_list))
                combo_list.insert(insert_pos, 95)
                combinations_with_ow.append(tuple(combo_list))
            combinations = combinations_with_ow
            print(f"[OK] 已为{len(combinations)}个组合添加OW类别95")
        
        # 保存组合列表
        combinations_file = os.path.join(self.output_root, dataset_name, f"combinations_{split}.json")
        with open(combinations_file, 'w') as f:
            json.dump({
                'combinations': [[int(x) for x in c] for c in combinations],
                'statistics': combo_stats,
                'ow_enabled': add_ow_class
            }, f, indent=2)
        print(f"[OK] 组合列表已保存: {combinations_file}")
        
        # 生成样本
        print(f"\n开始生成{split}样本...")
        query_filenames = []
        class_sample_usage = defaultdict(set)
        generation_stats = {
            'total_samples': 0,
            'total_length': 0,
            'lengths': [],
            'durations': [],
            'failed_samples': 0
        }
        
        for combo_idx, combo in enumerate(tqdm(combinations, desc=f"生成{split}样本")):
            for sample_idx in range(samples_per_combo):
                try:
                    # 加载各类别的单标签样本
                    traces = []
                    source_files = []
                    
                    for class_id in combo:
                        sample = self.load_single_label_sample(class_id, split)
                        traces.append(sample)
                        source_files.append((class_id, sample['source_file'], sample['source_path']))
                    
                    # 合并trace
                    merged_trace = self.merger.merge_traces_with_overlap(traces)
                    
                    # 验证合并结果
                    if not self.merger.validate_merge(merged_trace, verbose=False):
                        print(f"[WARN] 样本验证失败，跳过: combo={combo}, sample={sample_idx}")
                        generation_stats['failed_samples'] += 1
                        continue
                    
                    # 生成文件名
                    labels_str = "_".join(map(str, combo))
                    random_id = uuid.uuid4().hex[:8]
                    filename = f"{labels_str}_{random_id}.pkl"
                    
                    # 保存多标签样本
                    query_path = os.path.join(query_dir, filename)
                    with open(query_path, 'wb') as f:
                        pickle.dump({
                            'time': merged_trace['time'],
                            'data': merged_trace['data'],
                            'labels': merged_trace['labels'],
                            'metadata': merged_trace['metadata']
                        }, f)
                    
                    query_filenames.append(filename)
                    
                    # 复制单标签样本到support_data
                    for class_id, source_file, source_path in source_files:
                        if source_file not in class_sample_usage[class_id]:
                            class_support_dir = os.path.join(support_dir, str(class_id))
                            os.makedirs(class_support_dir, exist_ok=True)
                            
                            dest_path = os.path.join(class_support_dir, source_file)
                            if not os.path.exists(dest_path):
                                shutil.copy2(source_path, dest_path)
                            
                            class_sample_usage[class_id].add(source_file)
                    
                    # 统计信息
                    generation_stats['total_samples'] += 1
                    generation_stats['total_length'] += len(merged_trace['data'])
                    generation_stats['lengths'].append(len(merged_trace['data']))
                    generation_stats['durations'].append(merged_trace['metadata']['merged_duration'])
                    
                except Exception as e:
                    print(f"\n[ERROR] 生成样本失败: combo={combo}, sample={sample_idx}")
                    print(f"   错误: {e}")
                    generation_stats['failed_samples'] += 1
                    continue
        
        # 保存query文件名列表
        query_json_path = os.path.join(output_dir, f"{dataset_name}_{split}.json")
        with open(query_json_path, 'w') as f:
            json.dump(query_filenames, f, indent=2)
        print(f"\n[OK] Query索引已保存: {query_json_path}")
        
        # 生成统计报告
        self._save_statistics(
            dataset_name, split, num_tabs, combo_stats,
            generation_stats, class_sample_usage
        )
        
        print(f"\n[OK] {split}集生成完成:")
        print(f"  - 总样本数: {generation_stats['total_samples']}")
        print(f"  - 失败样本: {generation_stats['failed_samples']}")
        print(f"  - 平均长度: {np.mean(generation_stats['lengths']):.0f}")
        
        return output_dir
    
    def _save_statistics(
        self,
        dataset_name: str,
        split: str,
        num_tabs: int,
        combo_stats: Dict,
        generation_stats: Dict,
        class_sample_usage: Dict
    ):
        """保存统计报告"""
        lengths = generation_stats['lengths']
        durations = generation_stats['durations']
        
        statistics = {
            'dataset_name': dataset_name,
            'split': split,
            'mode': 'disjoint_zone',
            'num_tabs': num_tabs,
            'overlap_config': {
                'max_overlap_ratio': float(self.overlap_range[1]),
                'min_overlap_ratio': float(self.overlap_range[0])
            },
            'random_seed': int(self.random_seed),
            'combinations': combo_stats,
            'samples': {
                'total_samples': int(generation_stats['total_samples']),
                'failed_samples': int(generation_stats['failed_samples']),
                'avg_length': float(np.mean(lengths)) if lengths else 0,
                'std_length': float(np.std(lengths)) if lengths else 0,
                'min_length': int(np.min(lengths)) if lengths else 0,
                'max_length': int(np.max(lengths)) if lengths else 0,
                'avg_duration': float(np.mean(durations)) if durations else 0,
                'std_duration': float(np.std(durations)) if durations else 0
            },
            'support_data': {
                'num_classes_used': int(len(class_sample_usage)),
                'total_unique_samples': int(sum(len(v) for v in class_sample_usage.values())),
                'samples_per_class': {str(k): int(len(v)) for k, v in class_sample_usage.items()}
            }
        }
        
        stats_file = os.path.join(self.output_root, dataset_name, f"statistics_{split}.json")
        with open(stats_file, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        print(f"[OK] 统计报告已保存: {stats_file}")


# 预设配置
DISJOINT_CONFIGS = {
    'small': {
        '2tab': {
            'train_per_zone': 500,
            'test_total': 500,
            'samples_per_combo': 10
        },
        '3tab': {
            'train_per_zone': 1000,
            'test_total': 1500,
            'samples_per_combo': 5
        },
        '4tab': {
            'train_per_zone': 1500,
            'test_total': 3000,
            'samples_per_combo': 3
        },
        '5tab': {
            'train_per_zone': 2000,
            'test_total': 5000,
            'samples_per_combo': 2
        }
    },
    'medium': {
        '2tab': {
            'train_per_zone': 1000,
            'test_total': 1000,
            'samples_per_combo': 20
        },
        '3tab': {
            'train_per_zone': 5000,
            'test_total': 5000,
            'samples_per_combo': 5
        },
        '4tab': {
            'train_per_zone': 5000,
            'test_total': 5000,
            'samples_per_combo': 5
        },
        '5tab': {
            'train_per_zone': 5000,
            'test_total': 5000,
            'samples_per_combo': 5
        }
    },
    'large': {
        '2tab': {
            'train_per_zone': 1000,
            'test_total': 1000,
            'samples_per_combo': 20
        },
        '3tab': {
            'train_per_zone': 5000,
            'test_total': 2500,
            'samples_per_combo': 10
        },
        '4tab': {
            'train_per_zone': 7500,
            'test_total': 3000,
            'samples_per_combo': 7
        },
        '5tab': {
            'train_per_zone': 10000,
            'test_total': 5000,
            'samples_per_combo': 5
        }
    }
}


def main():
    parser = argparse.ArgumentParser(
        description='生成区域分离的多标签数据集（训练集=同区域，测试集=跨区域）'
    )
    
    parser.add_argument(
        '--tabs',
        type=int,
        required=True,
        choices=[2, 3, 4, 5],
        help='Tab数量（也是区域数量）'
    )
    
    parser.add_argument(
        '--scale',
        type=str,
        default='small',
        choices=['small', 'medium', 'large'],
        help='数据规模'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='../datasets/MMFOW',
        help='输入根目录（必须包含train和test子目录）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='../datasets/disjoint_zone_datasets',
        help='输出根目录'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    
    parser.add_argument(
        '--train-per-zone',
        type=int,
        help='每个区域的训练组合数（覆盖预设配置）'
    )
    
    parser.add_argument(
        '--test-total',
        type=int,
        help='测试集总组合数（覆盖预设配置）'
    )
    
    parser.add_argument(
        '--samples-per-combo',
        type=int,
        help='每个组合的样本数（覆盖预设配置）'
    )
    
    parser.add_argument(
        '--ow',
        action='store_true',
        help='添加Open World类别95'
    )
    
    args = parser.parse_args()
    
    # 获取配置
    config_key = f'{args.tabs}tab'
    config = DISJOINT_CONFIGS[args.scale][config_key]
    
    train_per_zone = args.train_per_zone if args.train_per_zone else config['train_per_zone']
    test_total = args.test_total if args.test_total else config['test_total']
    samples_per_combo = args.samples_per_combo if args.samples_per_combo else config['samples_per_combo']
    
    # 创建生成器
    generator = DisjointZoneDatasetGenerator(
        source_root=args.input,
        output_root=args.output,
        num_classes=60,
        overlap_range=(0.0, 0.4),
        random_seed=args.seed
    )
    
    # 生成数据集
    dataset_name = f'{args.tabs}tab_disjoint_zone'
    
    output_paths = generator.generate_disjoint_dataset(
        num_tabs=args.tabs,
        train_combinations_per_zone=train_per_zone,
        test_combinations=test_total,
        samples_per_combo=samples_per_combo,
        dataset_name=dataset_name,
        add_ow_class=args.ow
    )
    
    print(f"\n{'='*70}")
    print(f"全部完成！")
    print(f"{'='*70}")
    print(f"\n数据集信息:")
    print(f"  - 名称: {dataset_name}")
    print(f"  - 规模: {args.scale}")
    print(f"  - 训练集: 每区域{train_per_zone}组 × {args.tabs}区域 = {train_per_zone * args.tabs}组, "
          f"每组{samples_per_combo}样本 = {train_per_zone * args.tabs * samples_per_combo}样本")
    print(f"  - 测试集: {test_total}组 (全跨区), 每组{samples_per_combo}样本 = {test_total * samples_per_combo}样本")
    print(f"\n输出路径:")
    print(f"  - 训练集: {output_paths['train']}")
    print(f"  - 测试集: {output_paths['test']}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\n" + "="*70)
        print("区域分离组合数据集生成器")
        print("="*70)
        print("\n用途：")
        print("  测试模型对未见过组合模式的泛化能力")
        print("  - 训练集: 只包含同区域的组合（如 [0,0,0,0], [1,1,1,1]）")
        print("  - 测试集: 只包含全跨区的组合（如 [0,1,2,3]）")
        print("\n使用示例:")
        print("\n1. 生成4-tab小规模测试数据集:")
        print("   python generate_disjoint_combinations.py --tabs 4 --scale small \\")
        print("       --input /root/datasets/OW_split --output /root/datasets/disjoint_zone")
        print("\n2. 生成4-tab大规模数据集:")
        print("   python generate_disjoint_combinations.py --tabs 4 --scale large \\")
        print("       --input /root/datasets/OW_split --output /root/datasets/disjoint_zone")
        print("\n3. 自定义参数:")
        print("   python generate_disjoint_combinations.py --tabs 4 --scale medium \\")
        print("       --train-per-zone 10000 --test-total 20000 --samples-per-combo 5")
        print("\n4. 添加Open World类别:")
        print("   python generate_disjoint_combinations.py --tabs 4 --scale large --ow")
        print("\n5. 查看所有选项:")
        print("   python generate_disjoint_combinations.py --help")
        print("\n" + "="*70)
    else:
        main()

