"""
多标签数据集生成器
基于时间戳的多trace合并，支持类别均衡的组合采样
避免组合爆炸，参数化配置
改进：保持组合顺序，增强均衡补偿
"""

import os
import pickle
import json
import random
import numpy as np
import shutil
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict
from tqdm import tqdm
import uuid


class CombinationSampler:
    """
    类别均衡的组合采样器
    避免C(n,k)组合爆炸，通过迭代采样确保类别分布均衡
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
        
    def generate_balanced_combinations(
        self, 
        k: int, 
        target_num_combinations: int,
        max_std_ratio: float = 0.10,
        max_iterations: int = 100000,
        check_interval: int = 20,  # 更频繁地检查均衡性
        balance_attempts: int = 20  # 每次不均衡时尝试的次数
    ) -> Tuple[List[Tuple[int]], Dict]:
        """
        生成类别均衡的组合（不排序，保持顺序）
        
        Args:
            k: 每个组合的类别数（tab数）
            target_num_combinations: 目标组合数量
            max_std_ratio: 最大标准差/均值比例（默认10%）
            max_iterations: 最大迭代次数
            check_interval: 均衡性检查间隔
            balance_attempts: 每次不均衡时的补偿尝试次数
            
        Returns:
            combinations: 组合列表，每个组合是k个类别的元组（保持生成顺序）
            statistics: 统计信息字典
        """
        print(f"\n开始生成{k}-tab的{target_num_combinations}个均衡组合...")
        print(f"  - 类别范围: 0-{self.num_classes-1}")
        print(f"  - 均衡约束: std/mean < {max_std_ratio}")
        print(f"  - 检查间隔: 每{check_interval}个组合")
        print(f"  - 保持组合顺序: 是")
        
        combinations = set()
        class_counts = Counter()
        
        # 迭代采样，确保类别均衡
        iteration = 0
        total_balance_attempts = 0
        
        while len(combinations) < target_num_combinations and iteration < max_iterations:
            need_balance = False
            
            # 定期检查均衡性
            if len(combinations) > 0 and len(combinations) % check_interval == 0:
                counts = [class_counts[i] for i in range(self.num_classes)]
                if any(counts):  # 避免除零
                    mean_count = np.mean(counts)
                    std_count = np.std(counts)
                    std_ratio = std_count / mean_count if mean_count > 0 else 0
                    
                    if std_ratio > max_std_ratio:
                        need_balance = True
                        # 打印当前不均衡状态
                        print(f"  检测到不均衡 @ {len(combinations)}个组合: std/mean={std_ratio:.4f}")
            
            if need_balance:
                # 执行均衡补偿
                counts = [class_counts[i] for i in range(self.num_classes)]
                mean_count = np.mean(counts) if counts else 0
                
                # 找出所有低于平均值的类别
                underrepresented = [
                    cls for cls in range(self.num_classes) 
                    if class_counts[cls] < mean_count
                ]
                
                # 尝试生成包含欠代表类别的组合
                balanced_combos_added = 0
                for _ in range(balance_attempts):
                    if len(underrepresented) >= k:
                        # 全部从欠代表类别中选择
                        selected = np.random.choice(underrepresented, size=k, replace=False)
                    elif len(underrepresented) > 0:
                        # 优先选择欠代表类别，然后补充其他类别
                        n_under = len(underrepresented)
                        selected_under = np.random.choice(underrepresented, size=min(n_under, k), replace=False)
                        
                        # 从其他类别中补充
                        other_classes = [c for c in range(self.num_classes) if c not in selected_under]
                        n_need = k - len(selected_under)
                        
                        # 优先选择出现次数较少的其他类别
                        other_classes.sort(key=lambda x: class_counts[x])
                        selected_other = np.random.choice(other_classes[:len(other_classes)//2], 
                                                        size=n_need, replace=False)
                        
                        selected = np.concatenate([selected_under, selected_other])
                        np.random.shuffle(selected)  # 打乱顺序
                    else:
                        # 如果没有欠代表类别，正常随机
                        selected = np.random.choice(self.num_classes, size=k, replace=False)
                    
                    # 不排序，保持随机顺序
                    combo = tuple(selected.tolist())
                    
                    if combo not in combinations:
                        combinations.add(combo)
                        for cls in combo:
                            class_counts[cls] += 1
                        balanced_combos_added += 1
                        
                        # 如果成功添加了足够的均衡组合，可以提前退出
                        if balanced_combos_added >= check_interval // 2:
                            break
                
                total_balance_attempts += 1
                print(f"    -> 添加了{balanced_combos_added}个均衡组合")
                
            else:
                # 正常随机生成组合（不排序）
                selected = np.random.choice(self.num_classes, size=k, replace=False)
                combo = tuple(selected.tolist())  # 保持生成的顺序
                
                if combo not in combinations:
                    combinations.add(combo)
                    for cls in combo:
                        class_counts[cls] += 1
            
            iteration += 1
            
            # 进度显示
            if len(combinations) % 500 == 0:
                counts = [class_counts[i] for i in range(self.num_classes)]
                mean_count = np.mean(counts)
                std_count = np.std(counts)
                std_ratio = std_count / mean_count if mean_count > 0 else 0
                print(f"  进度: {len(combinations)}/{target_num_combinations}, "
                      f"std/mean: {std_ratio:.4f}")
        
        # 转换为列表
        combinations_list = list(combinations)
        
        # 计算最终统计信息
        counts = [class_counts[i] for i in range(self.num_classes)]
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        std_ratio = std_count / mean_count if mean_count > 0 else 0
        
        statistics = {
            'num_combinations': len(combinations_list),
            'k': k,
            'class_distribution': {int(k): int(v) for k, v in class_counts.items()},
            'mean_frequency': float(mean_count),
            'std_frequency': float(std_count),
            'std_mean_ratio': float(std_ratio),
            'min_frequency': int(min(counts)) if counts else 0,
            'max_frequency': int(max(counts)) if counts else 0,
            'iterations_used': int(iteration),
            'total_balance_attempts': int(total_balance_attempts)
        }
        
        print(f"\n[OK] 组合生成完成:")
        print(f"  - 生成组合数: {len(combinations_list)}")
        print(f"  - 类别频率: 均值={mean_count:.2f}, 标准差={std_count:.2f}")
        print(f"  - 均衡度: std/mean={std_ratio:.4f}")
        print(f"  - 频率范围: [{min(counts)}, {max(counts)}]")
        print(f"  - 均衡补偿次数: {total_balance_attempts}")
        
        return combinations_list, statistics


class TimeBasedMerger:
    """
    基于时间戳的多trace合并器
    通过启动延迟控制重叠，不破坏trace内部结构
    
    关键：处理trace时间跨度包含的边界情况
    """
    
    def __init__(self, overlap_range: Tuple[float, float] = (0.0, 0.4)):
        """
        Args:
            overlap_range: 重叠比例范围 [min, max]
        """
        self.overlap_range = overlap_range
        
    def merge_traces_with_overlap(
        self, 
        traces: List[Dict],
        overlap_ratios: Optional[List[float]] = None
    ) -> Dict:
        """
        合并多个trace，基于时间戳和重叠比例
        
        Args:
            traces: trace列表，每个trace是{'time': array, 'data': array, 'label': int}
            overlap_ratios: 重叠比例列表，长度为len(traces)-1
                          如果为None，则随机生成
                          
        Returns:
            merged_trace: 合并后的trace字典
        """
        num_traces = len(traces)
        
        # 生成或验证重叠比例
        if overlap_ratios is None:
            overlap_ratios = [
                random.uniform(self.overlap_range[0], self.overlap_range[1])
                for _ in range(num_traces - 1)
            ]
        else:
            assert len(overlap_ratios) == num_traces - 1, \
                f"重叠比例数量应为{num_traces-1}，实际为{len(overlap_ratios)}"
        
        # 计算每个trace的总时长
        durations = [trace['time'][-1] - trace['time'][0] for trace in traces]
        
        # 计算每个trace的启动时间
        start_times = [0.0]  # 第一个trace从0开始
        
        for i in range(1, num_traces):
            # 前一个trace的启动时间 + 前一个trace总时长 × (1 - 重叠比例)
            prev_start = start_times[i-1]
            prev_duration = durations[i-1]
            overlap_ratio = overlap_ratios[i-1]
            
            # 新trace的启动时间
            new_start = prev_start + prev_duration * (1.0 - overlap_ratio)
            start_times.append(new_start)
        
        # 调整每个trace的时间戳并合并
        all_packets = []  # (时间戳, 方向, trace_id)
        
        for trace_id, trace in enumerate(traces):
            adjusted_times = trace['time'] + start_times[trace_id]
            
            for t, d in zip(adjusted_times, trace['data']):
                all_packets.append((t, d, trace_id))
        
        # 按时间戳排序
        all_packets.sort(key=lambda x: x[0])
        
        # 提取排序后的时间和数据
        merged_time = np.array([p[0] for p in all_packets])
        merged_data = np.array([p[1] for p in all_packets])
        
        # 生成标签列表（保持原始顺序）
        labels = [trace['label'] for trace in traces]
        
        # 计算统计信息
        total_duration = merged_time[-1] - merged_time[0]
        
        merged_trace = {
            'time': merged_time,
            'data': merged_data,
            'labels': labels,  # 多标签（按顺序）
            'metadata': {
                'num_traces': num_traces,
                'overlap_ratios': overlap_ratios,
                'start_times': start_times,
                'original_durations': durations,
                'merged_duration': total_duration,
                'merged_length': len(merged_data)
            }
        }
        
        return merged_trace
    
    def validate_merge(self, merged_trace: Dict, verbose: bool = False) -> bool:
        """
        验证合并结果的正确性
        
        Args:
            merged_trace: 合并后的trace
            verbose: 是否打印详细信息
            
        Returns:
            is_valid: 是否有效
        """
        time = merged_trace['time']
        data = merged_trace['data']
        metadata = merged_trace['metadata']
        
        # 检查1: 时间戳单调递增
        is_monotonic = np.all(np.diff(time) >= 0)
        
        # 检查2: 数据长度一致
        length_match = len(time) == len(data) == metadata['merged_length']
        
        # 检查3: 标签数量与trace数量一致
        labels_match = len(merged_trace['labels']) == metadata['num_traces']
        
        is_valid = is_monotonic and length_match and labels_match
        
        if verbose or not is_valid:
            print(f"  验证结果:")
            print(f"    - 时间戳单调递增: {'PASS' if is_monotonic else 'FAIL'}")
            print(f"    - 数据长度一致: {'PASS' if length_match else 'FAIL'}")
            print(f"    - 标签数量正确: {'PASS' if labels_match else 'FAIL'}")
            if not is_monotonic:
                non_monotonic = np.where(np.diff(time) < 0)[0]
                print(f"    - 非单调位置: {non_monotonic[:5]}...")
        
        return is_valid


class MultiTabDatasetGenerator:
    """
    完整的多标签数据集生成器
    支持fixed-tab模式（每次生成单一tab数的数据集）
    改进：保持组合顺序，增强均衡补偿
    """
    
    def __init__(
        self,
        source_root: str = "datasets/OW_split",
        output_root: str = "datasets/multi_tab_datasets",
        num_classes: int = 60,
        overlap_range: Tuple[float, float] = (0.0, 0.4),
        random_seed: int = 42
    ):
        """
        Args:
            source_root: 单标签数据源根目录
            output_root: 多标签数据集输出根目录
            num_classes: 类别总数
            overlap_range: 重叠比例范围
            random_seed: 随机种子
        """
        self.source_root = source_root
        self.output_root = output_root
        self.num_classes = num_classes
        self.overlap_range = overlap_range
        self.random_seed = random_seed
        
        # 初始化组件
        self.sampler = CombinationSampler(num_classes, random_seed)
        self.merger = TimeBasedMerger(overlap_range)
        
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        
    def load_single_label_sample(self, class_id: int, split: str) -> Dict:
        """
        加载单标签样本
        
        Args:
            class_id: 类别ID
            split: 'train' 或 'test'
            
        Returns:
            sample: {'time': array, 'data': array, 'label': int}
        """
        class_dir = os.path.join(self.source_root, split, str(class_id))
        
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"类别目录不存在: {class_dir}")
        
        # 获取该类别的所有文件
        files = [f for f in os.listdir(class_dir) if f.endswith('.pkl')]
        
        if len(files) == 0:
            raise ValueError(f"类别{class_id}在{split}中没有样本")
        
        # 随机选择一个文件
        selected_file = random.choice(files)
        file_path = os.path.join(class_dir, selected_file)
        
        # 加载数据
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        return {
            'time': data['time'],
            'data': data['data'],
            'label': data['label'],
            'source_file': selected_file,
            'source_path': file_path
        }
    
    def generate_dataset(
        self,
        num_tabs: int,
        num_combinations: int,
        samples_per_combo: int,
        split: str = 'train',
        dataset_name: Optional[str] = None,
        check_interval: int = 20,  # 可配置的检查间隔
        balance_attempts: int = 20,  # 可配置的补偿尝试次数
        add_ow_class: bool = False  # 是否添加OW类别95
    ) -> str:
        """
        生成固定tab数的数据集
        
        Args:
            num_tabs: tab数量（2-5）
            num_combinations: 组合数量
            samples_per_combo: 每个组合生成的样本数
            split: 'train' 或 'test'
            dataset_name: 数据集名称（默认为"{num_tabs}tab"）
            check_interval: 均衡性检查间隔
            balance_attempts: 每次不均衡时的补偿尝试次数
            add_ow_class: 是否在每个组合的随机位置添加OW类别95
            
        Returns:
            output_dir: 输出目录路径
        """
        if dataset_name is None:
            dataset_name = f"{num_tabs}tab"
        
        print(f"\n{'='*60}")
        print(f"生成{dataset_name}数据集 - {split}集")
        print(f"{'='*60}")
        print(f"配置:")
        print(f"  - Tab数: {num_tabs}")
        print(f"  - 组合数: {num_combinations}")
        print(f"  - 每组样本数: {samples_per_combo}")
        print(f"  - 总样本数: {num_combinations * samples_per_combo}")
        print(f"  - 重叠比例范围: {self.overlap_range}")
        print(f"  - 保持组合顺序: 是")
        print(f"  - 添加OW类别95: {'是' if add_ow_class else '否'}")
        
        # 创建输出目录
        output_dir = os.path.join(self.output_root, dataset_name, split)
        query_dir = os.path.join(output_dir, "query_data")
        support_dir = os.path.join(output_dir, "support_data")
        
        os.makedirs(query_dir, exist_ok=True)
        os.makedirs(support_dir, exist_ok=True)
        
        # 生成类别组合
        combinations, combo_stats = self.sampler.generate_balanced_combinations(
            k=num_tabs,
            target_num_combinations=num_combinations,
            check_interval=check_interval,
            balance_attempts=balance_attempts
        )
        
        # 如果启用OW模式，在每个组合的随机位置添加类别95
        if add_ow_class:
            print(f"\n[OW模式] 在每个组合的随机位置添加类别95...")
            combinations_with_ow = []
            for combo in combinations:
                combo_list = list(combo)
                # 随机选择插入位置（0到len(combo_list)之间）
                insert_pos = random.randint(0, len(combo_list))
                combo_list.insert(insert_pos, 95)
                combinations_with_ow.append(tuple(combo_list))
            combinations = combinations_with_ow
            print(f"[OK] 已为{len(combinations)}个组合添加OW类别95")
            print(f"     示例: {combinations[0]} (类别95在位置{list(combinations[0]).index(95)})")
        
        # 保存组合列表
        combinations_file = os.path.join(
            self.output_root, dataset_name, f"combinations_{split}.json"
        )
        with open(combinations_file, 'w') as f:
            json.dump({
                'combinations': [[int(x) for x in c] for c in combinations],
                'statistics': combo_stats,
                'ow_enabled': add_ow_class
            }, f, indent=2)
        print(f"[OK] 组合列表已保存: {combinations_file}")
        
        # 生成样本
        print(f"\n开始生成样本...")
        query_filenames = []
        class_sample_usage = defaultdict(set)  # 记录每个类别使用的样本
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
                    # 加载各类别的单标签样本（按组合顺序）
                    traces = []
                    source_files = []
                    
                    for class_id in combo:  # 保持combo的原始顺序
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
                    
                    # 生成文件名: 类别1_类别2_..._随机ID.pkl（保持顺序）
                    labels_str = "_".join(map(str, combo))  # 不排序
                    random_id = uuid.uuid4().hex[:8]
                    filename = f"{labels_str}_{random_id}.pkl"
                    
                    # 保存多标签样本
                    query_path = os.path.join(query_dir, filename)
                    with open(query_path, 'wb') as f:
                        pickle.dump({
                            'time': merged_trace['time'],
                            'data': merged_trace['data'],
                            'labels': merged_trace['labels'],  # 有序的标签列表
                            'metadata': merged_trace['metadata']
                        }, f)
                    
                    query_filenames.append(filename)
                    
                    # 复制使用的单标签样本到support_data
                    for class_id, source_file, source_path in source_files:
                        # 只复制第一次使用的样本
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
        self._generate_statistics_report(
            dataset_name, split, num_tabs, combo_stats, 
            generation_stats, class_sample_usage
        )
        
        print(f"\n{'='*60}")
        print(f"[OK] {dataset_name}-{split}数据集生成完成!")
        print(f"{'='*60}")
        print(f"输出目录: {output_dir}")
        print(f"总样本数: {generation_stats['total_samples']}")
        print(f"失败样本: {generation_stats['failed_samples']}")
        
        return output_dir
    
    def _generate_statistics_report(
        self,
        dataset_name: str,
        split: str,
        num_tabs: int,
        combo_stats: Dict,
        generation_stats: Dict,
        class_sample_usage: Dict
    ):
        """生成统计报告"""
        # 计算类别在生成样本中的分布
        class_distribution = combo_stats['class_distribution']
        
        # 计算样本长度统计
        lengths = generation_stats['lengths']
        durations = generation_stats['durations']
        
        statistics = {
            'dataset_name': dataset_name,
            'split': split,
            'num_tabs': num_tabs,
            'target_length': 30000,
            'preserve_order': True,  # 标识保持了顺序
            'overlap_config': {
                'max_overlap_ratio': float(self.overlap_range[1]),
                'min_overlap_ratio': float(self.overlap_range[0])
            },
            'random_seed': int(self.random_seed),
            'combinations': {
                'num_combinations': int(combo_stats['num_combinations']),
                'class_distribution': class_distribution,
                'avg_class_frequency': float(combo_stats['mean_frequency']),
                'std_class_frequency': float(combo_stats['std_frequency']),
                'std_mean_ratio': float(combo_stats['std_mean_ratio']),
                'min_frequency': int(combo_stats['min_frequency']),
                'max_frequency': int(combo_stats['max_frequency']),
                'total_balance_attempts': int(combo_stats['total_balance_attempts'])
            },
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
        
        # 保存统计报告
        stats_file = os.path.join(
            self.output_root, dataset_name, f"statistics_{split}.json"
        )
        with open(stats_file, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        print(f"\n[统计报告]:")
        print(f"  - 样本长度: 均值={statistics['samples']['avg_length']:.0f}, "
              f"标准差={statistics['samples']['std_length']:.0f}")
        print(f"  - 样本时长: 均值={statistics['samples']['avg_duration']:.2f}s, "
              f"标准差={statistics['samples']['std_duration']:.2f}s")
        print(f"  - 使用单标签样本: {statistics['support_data']['total_unique_samples']}个")
        print(f"  - 均衡补偿次数: {statistics['combinations']['total_balance_attempts']}")
        print(f"[OK] 统计报告已保存: {stats_file}")


def test_time_based_merger():
    """测试时间戳合并算法"""
    print("\n" + "="*60)
    print("测试TimeBasedMerger - 时间戳合并算法")
    print("="*60)
    
    merger = TimeBasedMerger(overlap_range=(0.0, 0.4))
    
    # 创建3个模拟trace
    trace1 = {
        'time': np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        'data': np.array([1, -1, 1, -1, 1]),
        'label': 0
    }
    
    trace2 = {
        'time': np.array([0.0, 0.3, 0.6, 0.9]),
        'data': np.array([-1, 1, -1, 1]),
        'label': 1
    }
    
    trace3 = {
        'time': np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        'data': np.array([1, 1, -1, -1, 1, -1]),
        'label': 2
    }
    
    # 测试1: 正常重叠
    print("\n测试1: 正常重叠 (30%, 20%)")
    merged = merger.merge_traces_with_overlap(
        [trace1, trace2, trace3],
        overlap_ratios=[0.3, 0.2]
    )
    
    print(f"结果:")
    print(f"  - 原始长度: {len(trace1['data'])}, {len(trace2['data'])}, {len(trace3['data'])}")
    print(f"  - 合并长度: {len(merged['data'])}")
    print(f"  - 标签顺序: {merged['labels']}")  # 注意这里是有序的
    print(f"  - 启动时间: {merged['metadata']['start_times']}")
    print(f"  - 总时长: {merged['metadata']['merged_duration']:.3f}s")
    
    is_valid = merger.validate_merge(merged, verbose=True)
    print(f"  - 验证结果: {'PASS' if is_valid else 'FAIL'}")
    
    print("\n[OK] TimeBasedMerger测试完成!")


def test_combination_order():
    """测试组合顺序保持"""
    print("\n" + "="*60)
    print("测试组合顺序保持")
    print("="*60)
    
    sampler = CombinationSampler(num_classes=10, random_seed=42)
    
    # 生成一些组合
    combinations, stats = sampler.generate_balanced_combinations(
        k=3,
        target_num_combinations=50,
        check_interval=10,
        balance_attempts=10
    )
    
    # 检查是否存在顺序不同的组合
    print("\n前10个组合（保持生成顺序）:")
    for i, combo in enumerate(combinations[:10]):
        print(f"  {i+1}: {combo}")
    
    # 验证没有被排序
    sorted_combos = [tuple(sorted(combo)) for combo in combinations]
    identical_count = sum(1 for c1, c2 in zip(combinations, sorted_combos) if c1 == c2)
    
    print(f"\n顺序验证:")
    print(f"  - 总组合数: {len(combinations)}")
    print(f"  - 与排序版本相同的组合: {identical_count}")
    print(f"  - 保持原始顺序的比例: {(len(combinations)-identical_count)/len(combinations)*100:.1f}%")
    
    # 查找同样元素但顺序不同的组合对
    element_sets = {}
    for combo in combinations:
        element_set = frozenset(combo)
        if element_set not in element_sets:
            element_sets[element_set] = []
        element_sets[element_set].append(combo)
    
    print(f"\n相同元素但顺序不同的组合示例:")
    count = 0
    for element_set, combos in element_sets.items():
        if len(combos) > 1:
            print(f"  元素集{set(element_set)}: {combos}")
            count += 1
            if count >= 3:  # 只显示前3个示例
                break
    
    if count == 0:
        print("  （未找到相同元素但顺序不同的组合，这是正常的）")
    
    print("\n[OK] 组合顺序保持测试完成!")


if __name__ == "__main__":
    # 测试时间戳合并
    test_time_based_merger()
    
    # 测试组合顺序保持
    test_combination_order()
    
    print("\n" + "="*60)
    print("改进的多标签数据集生成器已准备就绪")
    print("="*60)
    print("\n核心改进:")
    print("  1. 移除sorted操作，保持组合顺序")
    print("  2. 更频繁的均衡检查（默认每20个组合）")
    print("  3. 增强的均衡补偿策略（每次补偿最多20个组合）")
    print("  4. 优先选择欠代表类别，持续补偿直到达到目标")
    print("\n使用示例:")
    print("  generator = MultiTabDatasetGenerator()")
    print("  generator.generate_dataset(")
    print("      num_tabs=2,")
    print("      num_combinations=100,")
    print("      samples_per_combo=5,")
    print("      split='train',")
    print("      check_interval=20,    # 可自定义")
    print("      balance_attempts=20   # 可自定义")
    print("  )")