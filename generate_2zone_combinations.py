"""
2区域组合数据集生成脚本（简化版本）
- 固定将60个类别分成2个区域：Zone0(0-29), Zone1(30-59)
- 训练集：只生成同区域内的组合
- 测试集：混合两个区域，前区选m个，后区选(tab-m)个，m∈[0,tab]
- 所有组合中类别不重复

适用于任意tab数（2-5tab），降低识别难度
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


class TwoZoneCombinationSampler:
    """
    2区域组合采样器
    固定分成2个区域，训练集同区域，测试集混合区域
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
        
        # 固定分成2个区域
        self.zone0 = list(range(0, num_classes // 2))  # 0-29
        self.zone1 = list(range(num_classes // 2, num_classes))  # 30-59
    
    def generate_train_combinations(
        self, 
        num_tabs: int,
        num_combinations_per_zone: int,
        max_std_ratio: float = 0.15
    ) -> Tuple[List[Tuple[int]], Dict]:
        """
        生成训练集组合：所有tab来自同一个区域，类别不重复
        
        Args:
            num_tabs: tab数量
            num_combinations_per_zone: 每个区域期望的组合数
            max_std_ratio: 区域内类别均衡的最大标准差比例
            
        Returns:
            combinations: 组合列表
            statistics: 统计信息
        """
        print(f"\n{'='*60}")
        print(f"生成训练集组合 (2区域同区模式，类别不重复)")
        print(f"{'='*60}")
        print(f"  - Tab数: {num_tabs}")
        print(f"  - 区域划分: Zone0(0-29), Zone1(30-59)")
        print(f"  - 每区域期望组合数: {num_combinations_per_zone}")
        
        zones = {0: self.zone0, 1: self.zone1}
        
        # 计算最大组合数
        n0, n1 = len(self.zone0), len(self.zone1)
        if num_tabs > n0 or num_tabs > n1:
            max_combos = 0
            print(f"  [ERROR] tab数({num_tabs}) > 区域类别数({n0})，无法生成不重复组合！")
        else:
            max_combos = 1
            for i in range(n0, n0 - num_tabs, -1):
                max_combos *= i
            print(f"  - 每区域最大组合数: A({n0},{num_tabs}) = {max_combos}")
        
        all_combinations = []
        class_counts = Counter()
        zone_stats = {}
        
        # 对每个区域生成组合
        for zone_id in [0, 1]:
            zone_classes = zones[zone_id]
            n = len(zone_classes)
            
            if num_tabs > n:
                print(f"\n[SKIP] Zone {zone_id}: tab数({num_tabs}) > 类别数({n})")
                continue
            
            # 计算该区域的最大组合数
            max_combos_zone = 1
            for i in range(n, n - num_tabs, -1):
                max_combos_zone *= i
            
            actual_target = min(num_combinations_per_zone, max_combos_zone)
            
            if actual_target < num_combinations_per_zone:
                print(f"\n[WARNING] Zone {zone_id}: 期望{num_combinations_per_zone}个，最大{max_combos_zone}个")
                print(f"          将生成{actual_target}个组合")
            
            print(f"\n生成 Zone {zone_id} ({zone_classes[0]}-{zone_classes[-1]}) 的组合 (目标: {actual_target}个)...")
            
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
                
                # 定期检查均衡性并补偿
                if len(zone_combinations) > 0 and len(zone_combinations) % 100 == 0:
                    counts = [zone_class_counts[c] for c in zone_classes]
                    mean_count = np.mean(counts) if counts else 0
                    std_count = np.std(counts) if counts else 0
                    std_ratio = std_count / mean_count if mean_count > 0 else 0
                    
                    if std_ratio > max_std_ratio:
                        # 找出使用次数最少的类别
                        underrepresented = sorted(zone_classes, key=lambda c: zone_class_counts[c])[:max(num_tabs, len(zone_classes)//2)]
                        
                        # 强制生成包含这些类别的组合
                        for _ in range(10):
                            if len(underrepresented) >= num_tabs:
                                selected = np.random.choice(underrepresented, size=num_tabs, replace=False)
                            else:
                                n_under = len(underrepresented)
                                selected_under = underrepresented
                                other_classes = [c for c in zone_classes if c not in selected_under]
                                n_need = num_tabs - n_under
                                
                                if len(other_classes) >= n_need:
                                    selected_other = np.random.choice(other_classes, size=n_need, replace=False)
                                    selected = np.concatenate([selected_under, selected_other])
                                else:
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
                'max_possible_combinations': max_combos_zone,
                'class_range': [zone_classes[0], zone_classes[-1]],
                'num_classes': len(zone_classes),
                'mean_frequency': float(np.mean(counts)) if counts else 0,
                'std_frequency': float(np.std(counts)) if counts else 0,
                'std_mean_ratio': float(np.std(counts) / np.mean(counts)) if np.mean(counts) > 0 else 0,
                'min_frequency': int(np.min(counts)) if counts else 0,
                'max_frequency': int(np.max(counts)) if counts else 0
            }
            
            print(f"  [OK] Zone {zone_id}: 生成{len(zone_combinations_list)}个组合, "
                  f"均衡度={zone_stats[zone_id]['std_mean_ratio']:.4f}")
        
        # 全局统计
        counts = [class_counts[i] for i in range(self.num_classes)]
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        statistics = {
            'mode': 'train_same_zone_2zones',
            'num_tabs': num_tabs,
            'num_zones': 2,
            'zone0_range': [0, 29],
            'zone1_range': [30, 59],
            'num_combinations': len(all_combinations),
            'zones': zone_stats,
            'global_class_distribution': {int(k): int(v) for k, v in class_counts.items()},
            'global_mean_frequency': float(mean_count),
            'global_std_frequency': float(std_count),
            'global_std_mean_ratio': float(std_count / mean_count) if mean_count > 0 else 0
        }
        
        print(f"\n[OK] 训练集组合生成完成:")
        print(f"  - 总组合数: {len(all_combinations)}")
        print(f"  - 期望组合数: {2 * num_combinations_per_zone}")
        print(f"  - 全局均衡度: std/mean={statistics['global_std_mean_ratio']:.4f}")
        print(f"  - 示例组合: {all_combinations[:3]}")
        
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
        num_combinations: int
    ) -> Tuple[List[Tuple[int]], Dict]:
        """
        生成测试集组合：混合两个区域
        从Zone0选m个，从Zone1选(tab-m)个，m∈[0,tab]，类别不重复
        
        Args:
            num_tabs: tab数量
            num_combinations: 要生成的组合数
            
        Returns:
            combinations: 组合列表
            statistics: 统计信息
        """
        print(f"\n{'='*60}")
        print(f"生成测试集组合 (2区域混合模式，类别不重复)")
        print(f"{'='*60}")
        print(f"  - Tab数: {num_tabs}")
        print(f"  - 区域划分: Zone0(0-29), Zone1(30-59)")
        print(f"  - 目标组合数: {num_combinations}")
        print(f"  - 混合策略: 从Zone0选m个，从Zone1选({num_tabs}-m)个，m∈[0,{num_tabs}]")
        
        combinations = set()
        class_counts = Counter()
        m_distribution = Counter()  # 记录每种m值的分布
        
        attempts = 0
        max_attempts = num_combinations * 10000
        
        print(f"\n开始生成混合组合...")
        
        while len(combinations) < num_combinations and attempts < max_attempts:
            # 随机选择m（从Zone0选几个）
            m = random.randint(0, num_tabs)
            n_from_zone1 = num_tabs - m
            
            combo = []
            
            # 从Zone0选m个
            if m > 0:
                if m > len(self.zone0):
                    attempts += 1
                    continue
                selected_zone0 = np.random.choice(self.zone0, size=m, replace=False)
                combo.extend(selected_zone0.tolist())
            
            # 从Zone1选(tab-m)个
            if n_from_zone1 > 0:
                if n_from_zone1 > len(self.zone1):
                    attempts += 1
                    continue
                selected_zone1 = np.random.choice(self.zone1, size=n_from_zone1, replace=False)
                combo.extend(selected_zone1.tolist())
            
            # 打乱顺序（有序排列）
            random.shuffle(combo)
            combo_tuple = tuple(combo)
            
            if combo_tuple not in combinations:
                combinations.add(combo_tuple)
                m_distribution[m] += 1
                
                for cls in combo:
                    class_counts[cls] += 1
            
            attempts += 1
            
            # 进度显示
            if len(combinations) % 500 == 0 and len(combinations) > 0:
                print(f"  进度: {len(combinations)}/{num_combinations}")
        
        combinations_list = list(combinations)
        
        # 验证混合性质
        mixed_verification = []
        for combo in combinations_list[:10]:  # 验证前10个
            zone0_count = sum(1 for cls in combo if cls in self.zone0)
            zone1_count = sum(1 for cls in combo if cls in self.zone1)
            mixed_verification.append({
                'combination': combo,
                'zone0_count': zone0_count,
                'zone1_count': zone1_count
            })
        
        # 统计
        counts = [class_counts[i] for i in range(self.num_classes)]
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        statistics = {
            'mode': 'test_mixed_2zones',
            'num_tabs': num_tabs,
            'num_zones': 2,
            'zone0_range': [0, 29],
            'zone1_range': [30, 59],
            'num_combinations': len(combinations_list),
            'attempts_used': attempts,
            'm_distribution': {f'zone0={k}_zone1={num_tabs-k}': int(v) for k, v in sorted(m_distribution.items())},
            'global_class_distribution': {int(k): int(v) for k, v in class_counts.items()},
            'global_mean_frequency': float(mean_count),
            'global_std_frequency': float(std_count),
            'global_std_mean_ratio': float(std_count / mean_count) if mean_count > 0 else 0,
            'verification_samples': mixed_verification
        }
        
        print(f"\n[OK] 测试集组合生成完成:")
        print(f"  - 总组合数: {len(combinations_list)}")
        print(f"  - 全局均衡度: std/mean={statistics['global_std_mean_ratio']:.4f}")
        print(f"  - m值分布:")
        for k, v in sorted(m_distribution.items()):
            print(f"      Zone0选{k}个, Zone1选{num_tabs-k}个: {v}个组合")
        print(f"  - 示例组合:")
        for item in mixed_verification[:3]:
            print(f"      {item['combination']} (Zone0: {item['zone0_count']}, Zone1: {item['zone1_count']})")
        
        # 验证组合中没有重复类别
        has_duplicates = sum(1 for combo in combinations_list if len(combo) != len(set(combo)))
        if has_duplicates > 0:
            print(f"  [WARNING] 发现{has_duplicates}个包含重复类别的组合！")
        else:
            print(f"  [OK] 所有组合均无重复类别")
        
        return combinations_list, statistics


class TwoZoneDatasetGenerator:
    """
    2区域数据集生成器
    复用原有的 TimeBasedMerger 和数据加载逻辑
    """
    
    def __init__(
        self,
        source_root: str,
        output_root: str,
        num_classes: int = 60,
        overlap_range: Tuple[float, float] = (0.0, 0.4),
        random_seed: int = 42
    ):
        self.source_root = source_root
        self.output_root = output_root
        self.num_classes = num_classes
        self.overlap_range = overlap_range
        self.random_seed = random_seed
        
        # 初始化组件
        self.sampler = TwoZoneCombinationSampler(num_classes, random_seed)
        self.merger = TimeBasedMerger(overlap_range)
        
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def load_single_label_sample(self, class_id: int, split: str) -> Dict:
        """加载单标签样本"""
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
    
    def generate_2zone_dataset(
        self,
        num_tabs: int,
        train_combinations_per_zone: int,
        test_combinations: int,
        samples_per_combo: int,
        dataset_name: str,
        add_ow_class: bool = False
    ) -> Dict[str, str]:
        """
        生成2区域分离的训练集和测试集
        
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
        print(f"生成2区域数据集: {dataset_name}")
        print(f"{'='*70}")
        print(f"配置:")
        print(f"  - Tab数: {num_tabs}")
        print(f"  - 区域划分: Zone0(0-29), Zone1(30-59)")
        print(f"  - 训练集: 每区域{train_combinations_per_zone}个组合 × 2区域 = {train_combinations_per_zone * 2}个组合")
        print(f"  - 测试集: {test_combinations}个组合 (混合区域)")
        print(f"  - 每组样本数: {samples_per_combo}")
        print(f"  - 训练集总样本: {train_combinations_per_zone * 2 * samples_per_combo}")
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
        print(f"[2/2] 生成测试集（混合区域组合）")
        print(f"{'='*70}")
        
        test_combinations, test_combo_stats = self.sampler.generate_test_combinations(
            num_tabs=num_tabs,
            num_combinations=test_combinations
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
                'mode': '2zone_disjoint',
                'zone0_range': [0, 29],
                'zone1_range': [30, 59],
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
        print(f"[OK] 2区域数据集生成完成!")
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
        """根据组合生成样本"""
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
            'mode': '2zone_disjoint',
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
TWO_ZONE_CONFIGS = {
    'small': {
        '2tab': {'train_per_zone': 500, 'test_total': 500, 'samples_per_combo': 10},
        '3tab': {'train_per_zone': 1500, 'test_total': 2000, 'samples_per_combo': 5},
        '4tab': {'train_per_zone': 2500, 'test_total': 4000, 'samples_per_combo': 3},
        '5tab': {'train_per_zone': 3500, 'test_total': 6000, 'samples_per_combo': 2}
    },
    'medium': {
        '2tab': {'train_per_zone': 1000, 'test_total': 1000, 'samples_per_combo': 20},
        '3tab': {'train_per_zone': 4000, 'test_total': 6000, 'samples_per_combo': 10},
        '4tab': {'train_per_zone': 7000, 'test_total': 12000, 'samples_per_combo': 5},
        '5tab': {'train_per_zone': 10000, 'test_total': 18000, 'samples_per_combo': 3}
    },
    'large': {
        '2tab': {'train_per_zone': 1000, 'test_total': 1000, 'samples_per_combo': 20},
        '3tab': {'train_per_zone': 6000, 'test_total': 3000, 'samples_per_combo': 10},
        '4tab': {'train_per_zone': 15000, 'test_total': 5000, 'samples_per_combo': 10},
        '5tab': {'train_per_zone': 30000, 'test_total': 10000, 'samples_per_combo': 5}
    }
}


def main():
    parser = argparse.ArgumentParser(
        description='生成2区域组合数据集（训练集=同区域，测试集=混合区域）'
    )
    
    parser.add_argument(
        '--tabs',
        type=int,
        required=True,
        choices=[2, 3, 4, 5],
        help='Tab数量'
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
        default='../datasets/2zone_datasets',
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
    config = TWO_ZONE_CONFIGS[args.scale][config_key]
    
    train_per_zone = args.train_per_zone if args.train_per_zone else config['train_per_zone']
    test_total = args.test_total if args.test_total else config['test_total']
    samples_per_combo = args.samples_per_combo if args.samples_per_combo else config['samples_per_combo']
    
    # 创建生成器
    generator = TwoZoneDatasetGenerator(
        source_root=args.input,
        output_root=args.output,
        num_classes=60,
        overlap_range=(0.0, 0.4),
        random_seed=args.seed
    )
    
    # 生成数据集
    dataset_name = f'{args.tabs}tab_2zone'
    
    output_paths = generator.generate_2zone_dataset(
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
    print(f"  - 区域划分: Zone0(0-29), Zone1(30-59)")
    print(f"  - 训练集: 每区域{train_per_zone}组 × 2区域 = {train_per_zone * 2}组, "
          f"每组{samples_per_combo}样本 = {train_per_zone * 2 * samples_per_combo}样本")
    print(f"  - 测试集: {test_total}组 (混合区域), 每组{samples_per_combo}样本 = {test_total * samples_per_combo}样本")
    print(f"\n输出路径:")
    print(f"  - 训练集: {output_paths['train']}")
    print(f"  - 测试集: {output_paths['test']}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\n" + "="*70)
        print("2区域组合数据集生成器（简化版）")
        print("="*70)
        print("\n特点：")
        print("  - 固定2个区域: Zone0(0-29), Zone1(30-59)")
        print("  - 训练集: 只包含同区域的组合")
        print("  - 测试集: 混合两个区域（从Zone0选m个，从Zone1选(tab-m)个）")
        print("  - 所有组合中类别不重复")
        print("  - 适用于任意tab数（2-5tab）")
        print("\n使用示例:")
        print("\n1. 生成4-tab小规模测试数据集:")
        print("   python generate_2zone_combinations.py --tabs 4 --scale small \\")
        print("       --input /root/datasets/OW_split --output /root/datasets/2zone")
        print("\n2. 生成4-tab大规模数据集:")
        print("   python generate_2zone_combinations.py --tabs 4 --scale large \\")
        print("       --input /root/datasets/OW_split --output /root/datasets/2zone")
        print("\n3. 生成3-tab数据集:")
        print("   python generate_2zone_combinations.py --tabs 3 --scale medium \\")
        print("       --input /root/datasets/OW_split --output /root/datasets/2zone")
        print("\n4. 自定义参数:")
        print("   python generate_2zone_combinations.py --tabs 4 --scale medium \\")
        print("       --train-per-zone 8000 --test-total 15000 --samples-per-combo 5")
        print("\n5. 添加Open World类别:")
        print("   python generate_2zone_combinations.py --tabs 4 --scale large --ow")
        print("\n6. 查看所有选项:")
        print("   python generate_2zone_combinations.py --help")
        print("\n" + "="*70)
    else:
        main()


