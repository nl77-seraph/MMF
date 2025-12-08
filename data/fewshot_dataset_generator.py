"""
Few-shot数据集生成器

参考 Few-shot Object Detection via Feature Reweighting 的设计思路:
- K-shot: 每个类别有K个带标注的样本用于支持集
- Base classes: 用于base training的类别 (0-59)
- Novel classes: 用于few-shot fine-tuning的新类别 (60-n)

Few-shot阶段数据组织:
1. Support集: 
   - Base classes: 每类K个样本 (从base training的support中采样)
   - Novel classes: 每类K个样本 (从split后未用于base training的类别中采样)
   
2. Query集:
   - 合成的多标签样本
   - 每个query样本只包含1个novel class + 若干base classes
   - 每个novel class在query中出现总次数 = K
"""

import os
import pickle
import json
import random
import shutil
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
from tqdm import tqdm
import uuid
import argparse


class FewshotDatasetGenerator:
    """
    Few-shot数据集生成器
    
    生成用于few-shot fine-tuning阶段的数据集
    """
    
    def __init__(
        self,
        base_training_dir: str,  # base training的目录
        novel_source_dir: str,  # novel classes的原始数据目录
        output_root: str,       # 输出目录
        base_classes: List[int] = None,  # base类别列表，默认0-59
        novel_classes: List[int] = None,  # novel类别列表
        overlap_range: Tuple[float, float] = (0.0, 0.4),
        random_seed: int = 42,
        mixed: bool = False,
        copy_base_query: bool = False,
        num_base_per_query: int = 2
    ):
        """
        Args:
            base_training_dir: base training使用的数据目录
            novel_source_dir: novel classes的原始single-tab数据目录
            output_root: few-shot数据集输出目录
            base_classes: base类别列表，默认[0, 1, ..., 59]
            novel_classes: novel类别列表，如[60, 61, ..., 69]
            overlap_range: 合成时的重叠比例范围
            random_seed: 随机种子
        """
        self.base_training_dir = base_training_dir
        self.novel_source_dir = novel_source_dir
        self.output_root = output_root
        
        self.base_classes = base_classes if base_classes else list(range(60))
        self.novel_classes = novel_classes if novel_classes else []
        
        self.overlap_range = overlap_range
        self.random_seed = random_seed
        self.mixed = mixed
        self.copy_base_query = copy_base_query
        if self.mixed:
            self.output_root = os.path.join(self.output_root, 'mixed_tab')
        else:
            self.output_root = os.path.join(self.output_root, f'{num_base_per_query+1}_tab')
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 所有类别
        self.all_classes = sorted(self.base_classes + self.novel_classes)
        
        print(f"FewshotDatasetGenerator 初始化:")
        print(f"  - Base classes: {len(self.base_classes)}个 ({min(self.base_classes)}-{max(self.base_classes)})")
        print(f"  - Novel classes: {len(self.novel_classes)}个 ({self.novel_classes if self.novel_classes else 'None'})")
        print(f"  - Base support目录: {base_training_dir}")
        print(f"  - Novel source目录: {novel_source_dir}")
        print(f"  - 输出目录: {output_root}")
    
    def _get_class_samples(self, class_id: int, split: str = 'train') -> List[str]:
        """
        获取某个类别的所有样本文件路径
        
        Args:
            class_id: 类别ID
            split: 'train' 或 'test'
            
        Returns:
            样本文件路径列表
        """
        if class_id in self.base_classes:
            # Base class: 从base training的support目录获取
            class_dir = os.path.join(self.base_training_dir,split, 'support_data', str(class_id))
            # print(class_dir)
        else:
            # Novel class: 从原始split目录获取
            class_dir = os.path.join(self.novel_source_dir, split, str(class_id))
        
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"类别目录不存在: {class_dir}")
        
        files = [
            os.path.join(class_dir, f) 
            for f in os.listdir(class_dir) 
            if f.endswith('.pkl')
        ]
        
        return files
    
    def _load_sample(self, file_path: str) -> Dict:
        """加载单个样本"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def _sample_k_shot(
        self, 
        class_id: int, 
        k: int, 
        split: str = 'train',
        exclude_files: Set[str] = None
    ) -> List[str]:
        """
        从某个类别中采样K个样本
        
        Args:
            class_id: 类别ID
            k: 采样数量
            split: 'train' 或 'test'
            exclude_files: 需要排除的文件集合
            
        Returns:
            采样的文件路径列表
        """
        all_files = self._get_class_samples(class_id, split)
        
        if exclude_files:
            all_files = [f for f in all_files if os.path.basename(f) not in exclude_files]
        
        if len(all_files) < k:
            print(f"[WARN] 类别{class_id}样本不足: 需要{k}个，实际{len(all_files)}个")
            return all_files  # 返回所有可用样本
        
        return random.sample(all_files, k)
    
    def _merge_traces(
        self, 
        traces: List[Dict],
        overlap_ratios: Optional[List[float]] = None
    ) -> Dict:
        """
        合并多个trace（参考multi_tab_generator.py）
        
        Args:
            traces: trace列表
            overlap_ratios: 重叠比例列表
            
        Returns:
            合并后的trace
        """
        num_traces = len(traces)
        
        if overlap_ratios is None:
            overlap_ratios = [
                random.uniform(self.overlap_range[0], self.overlap_range[1])
                for _ in range(num_traces - 1)
            ]
        
        # 计算时长
        durations = [trace['time'][-1] - trace['time'][0] for trace in traces]
        
        # 计算启动时间
        start_times = [0.0]
        for i in range(1, num_traces):
            prev_start = start_times[i-1]
            prev_duration = durations[i-1]
            overlap_ratio = overlap_ratios[i-1]
            new_start = prev_start + prev_duration * (1.0 - overlap_ratio)
            start_times.append(new_start)
        
        # 合并
        all_packets = []
        for trace_id, trace in enumerate(traces):
            adjusted_times = trace['time'] + start_times[trace_id]
            for t, d in zip(adjusted_times, trace['data']):
                all_packets.append((t, d, trace_id))
        
        all_packets.sort(key=lambda x: x[0])
        
        merged_time = np.array([p[0] for p in all_packets])
        merged_data = np.array([p[1] for p in all_packets])
        labels = [trace['label'] for trace in traces]
        
        return {
            'time': merged_time,
            'data': merged_data,
            'labels': labels,
            'metadata': {
                'num_traces': num_traces,
                'overlap_ratios': overlap_ratios,
                'start_times': start_times,
                'original_durations': durations,
                'merged_duration': merged_time[-1] - merged_time[0],
                'merged_length': len(merged_data)
            }
        }
    
    def generate_fewshot_support(
        self,
        k_shot: int,
        split: str = 'train'
    ) -> Tuple[str, Dict]:
        """
        生成Few-shot的Support数据集
        
        为每个类别（base + novel）采样K个样本
        
        Args:
            k_shot: 每类样本数
            split: 'train' 或 'test'
            
        Returns:
            support_dir: support数据目录
            statistics: 统计信息
        """
        print(f"\n{'='*60}")
        print(f"生成Few-shot Support数据集 ({k_shot}-shot)")
        print(f"{'='*60}")
        
        support_dir = os.path.join(self.output_root, f'{k_shot}shot', split, 'support_data')
        os.makedirs(support_dir, exist_ok=True)
        
        statistics = {
            'k_shot': k_shot,
            'split': split,
            'base_classes': self.base_classes,
            'novel_classes': self.novel_classes,
            'samples_per_class': {}
        }
        
        all_classes = self.base_classes + self.novel_classes
        
        for class_id in tqdm(all_classes, desc="采样Support"):
            class_support_dir = os.path.join(support_dir, str(class_id))
            os.makedirs(class_support_dir, exist_ok=True)
            
            # 采样K个样本
            sampled_files = self._sample_k_shot(class_id, k_shot, split)
            
            # 复制到support目录
            for src_path in sampled_files:
                dst_path = os.path.join(class_support_dir, os.path.basename(src_path))
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
            
            statistics['samples_per_class'][str(class_id)] = len(sampled_files)
        
        # 保存统计信息
        stats_path = os.path.join(self.output_root, f'{k_shot}shot', split, 'support_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        print(f"\n[OK] Support数据生成完成:")
        print(f"  - 输出目录: {support_dir}")
        print(f"  - Base类别: {len(self.base_classes)}类 × {k_shot}样本")
        print(f"  - Novel类别: {len(self.novel_classes)}类 × {k_shot}样本")
        
        return support_dir, statistics
    
    def generate_fewshot_query(
        self,
        k_shot: int,
        num_base_per_query: int = 2,  # 每个query中base类别数量
        split: str = 'train'
    ) -> Tuple[str, Dict]:
        """
        生成Few-shot的Query数据集
        
        关键设计:
        1. 每个query只包含1个novel class + 若干base classes
        2. 每个novel class在所有query中出现的总次数 = K
        
        Args:
            k_shot: K值（每个novel类别在query中出现K次）
            num_base_per_query: 每个query中base类别的数量
            split: 'train' 或 'test'
            
        Returns:
            query_dir: query数据目录
            statistics: 统计信息
        """
        print(f"\n{'='*60}")
        print(f"生成Few-shot Query数据集 ({k_shot}-shot)")
        print(f"{'='*60}")
        print(f"配置:")
        print(f"  - 每个query包含: 1个novel + {num_base_per_query}个base")
        print(f"  - 每个novel类别出现次数: {k_shot}")
        print(f"  - 总query数: {len(self.novel_classes) * k_shot}")
        
        query_dir = os.path.join(self.output_root, f'{k_shot}shot', split, 'query_data')
        os.makedirs(query_dir, exist_ok=True)
        
        # 获取已经用于support的novel样本
        support_dir = os.path.join(self.output_root, f'{k_shot}shot', split, 'support_data')
        # pkl_files = [f for f in os.listdir(support_dir) if f.endswith('.pkl')]
        # print(len(pkl_files))
        # exit()
        used_novel_samples = defaultdict(set)
        
        for novel_class in self.novel_classes:
            novel_support_dir = os.path.join(support_dir, str(novel_class))
            if os.path.exists(novel_support_dir):
                for f in os.listdir(novel_support_dir):
                    if f.endswith('.pkl'):
                        used_novel_samples[novel_class].add(os.path.join(novel_support_dir, f))

        query_filenames = []
        novel_class_counts = defaultdict(int)  # 记录每个novel类别出现次数
        
        statistics = {
            'k_shot': k_shot,
            'split': split,
            'num_base_per_query': num_base_per_query,
            'novel_classes': self.novel_classes,
            'total_queries': 0,
            'novel_class_distribution': {}
        }
        
        # 为每个novel类别生成K个query样本
        if split == 'test':
            query_shot = 5*k_shot
        else:
            query_shot = k_shot
        for novel_class in tqdm(self.novel_classes, desc="生成Query"):
            for shot_idx in range(query_shot):
                try:
                    # 
                    #novel_files = self._get_class_samples(novel_class, split)
                    available_novel = [
                        f for f in used_novel_samples[novel_class] 
                        #if os.path.basename(f) not in used_novel_samples[novel_class]
                    ]
                    if not available_novel:
                        print(f"[WARN] Novel类别{novel_class}没有可用样本")
                        continue
                    
                    novel_file = available_novel[shot_idx%k_shot]
                    novel_sample = self._load_sample(novel_file)
                    novel_sample['label'] = novel_class
                    
                    # 2. 采样base类别样本
                    if self.mixed:
                        tabs = random.randint(1, 3)
                        selected_base_classes = random.sample(self.base_classes, tabs)
                    else:       
                        selected_base_classes = random.sample(self.base_classes, num_base_per_query)
                    base_samples = []
                    
                    for base_class in selected_base_classes:
                        base_files = self._get_class_samples(base_class, 'train')  # base使用train
                        if base_files:
                            base_file = random.choice(base_files)
                            base_sample = self._load_sample(base_file)
                            base_sample['label'] = base_class
                            base_samples.append(base_sample)
                    
                    if len(base_samples) < num_base_per_query and not self.mixed:
                        print(f"[WARN] Base样本不足，跳过")
                        continue
                    
                    # 3. 随机确定novel在组合中的位置
                    all_samples = base_samples.copy()
                    insert_pos = random.randint(0, len(all_samples))
                    all_samples.insert(insert_pos, novel_sample)
                    
                    # 4. 合并traces
                    merged = self._merge_traces(all_samples)
                    
                    # 5. 保存
                    labels_str = "_".join(map(str, merged['labels']))
                    random_id = uuid.uuid4().hex[:8]
                    filename = f"novel{novel_class}_{labels_str}_{random_id}.pkl"
                    
                    query_path = os.path.join(query_dir, filename)
                    with open(query_path, 'wb') as f:
                        pickle.dump({
                            'time': merged['time'],
                            'data': merged['data'],
                            'labels': merged['labels'],
                            'novel_class': novel_class,  # 标记哪个是novel
                            'base_classes': selected_base_classes,
                            'metadata': merged['metadata']
                        }, f)
                    
                    query_filenames.append(filename)
                    novel_class_counts[novel_class] += 1
                    
                except Exception as e:
                    print(f"[ERROR] 生成query失败: novel={novel_class}, shot={shot_idx}, error={e}")
                    continue
        if self.copy_base_query:
            base_query_dir = os.path.join(self.base_training_dir, split, 'query_data')
            # 采样K个样本
            pkl_files = [f for f in os.listdir(base_query_dir) if f.endswith('.pkl')]

            selected_files = random.sample(pkl_files, 60*k_shot)
            
            # 复制到support目录
            for pkl_name in selected_files:
                dst_path = os.path.join(base_query_dir, pkl_name)
                src_path = os.path.join(base_query_dir, pkl_name)
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    query_filenames.append(pkl_name)
        # 保存query索引
        query_json_path = os.path.join(self.output_root, f'{k_shot}shot', split, f'fewshot_query_{split}.json')
        with open(query_json_path, 'w') as f:
            json.dump(query_filenames, f, indent=2)
        
        statistics['total_queries'] = len(query_filenames)
        statistics['novel_class_distribution'] = dict(novel_class_counts)
        
        # 保存统计信息
        stats_path = os.path.join(self.output_root, f'{k_shot}shot', split, 'query_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        print(f"\n[OK] Query数据生成完成:")
        print(f"  - 输出目录: {query_dir}")
        print(f"  - 总Query数: {len(query_filenames)}")
        print(f"  - Novel类别分布: {dict(novel_class_counts)}")
        
        return query_dir, statistics
    
    def generate_fewshot_dataset(
        self,
        k_shot: int,
        num_base_per_query: int = 2,
        splits: List[str] = ['train']
    ) -> Dict:
        """
        生成完整的Few-shot数据集
        
        Args:
            k_shot: K值
            num_base_per_query: 每个query中base类别数量
            splits: 要生成的splits列表
            
        Returns:
            包含所有生成路径的字典
        """
        print(f"\n{'='*60}")
        print(f"生成完整的 {k_shot}-shot Few-shot 数据集")
        print(f"{'='*60}")
        
        results = {}
        
        for split in splits:
            print(f"\n--- 生成 {split} 集 ---")
            
            # 1. 生成Support
            support_dir, support_stats = self.generate_fewshot_support(k_shot, split)
            
            # 2. 生成Query
            query_dir, query_stats = self.generate_fewshot_query(k_shot, num_base_per_query, split)
            
            results[split] = {
                'support_dir': support_dir,
                'query_dir': query_dir,
                'support_stats': support_stats,
                'query_stats': query_stats
            }
        
        # 保存整体配置
        config = {
            'k_shot': k_shot,
            'num_base_per_query': num_base_per_query,
            'base_classes': self.base_classes,
            'novel_classes': self.novel_classes,
            'splits': splits,
            'overlap_range': self.overlap_range,
            'random_seed': self.random_seed
        }
        
        config_path = os.path.join(self.output_root, f'{k_shot}shot', 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"[OK] {k_shot}-shot 数据集生成完成!")
        print(f"{'='*60}")
        print(f"输出目录: {os.path.join(self.output_root, f'{k_shot}shot')}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='生成Few-shot数据集')
    
    parser.add_argument(
        '--base-training-dir',
        type=str,
        required=True,
        help='Base training的目录'
    )
    
    parser.add_argument(
        '--novel-source-dir',
        type=str,
        required=True,
        help='Novel classes的原始数据目录(split后的)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='输出目录'
    )
    
    parser.add_argument(
        '--base-classes',
        type=str,
        default='0-59',
        help='Base类别范围，格式: start-end 或 逗号分隔'
    )
    
    parser.add_argument(
        '--novel-classes',
        type=str,
        required=True,
        help='Novel类别，格式: start-end 或 逗号分隔'
    )
    
    parser.add_argument(
        '--k-shot',
        type=int,
        default=5,
        help='K-shot值'
    )
    
    parser.add_argument(
        '--num-base-per-query',
        type=int,
        default=2,
        help='每个query中base类别数量'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    parser.add_argument(
        '--mixed',
        action='store_true',
        help='生成mixed-tab query数据集（2-5 tab混合）'
    )
    parser.add_argument(
        '--copy-base-query',
        action='store_true',
        help='复制base query数据集'
    )
    args = parser.parse_args()
    
    # 解析类别范围
    def parse_class_range(s):
        if '-' in s and ',' not in s:
            parts = s.split('-')
            return list(range(int(parts[0]), int(parts[1]) + 1))
        else:
            return [int(x) for x in s.split(',')]
    
    base_classes = parse_class_range(args.base_classes)
    novel_classes = parse_class_range(args.novel_classes)
    
    # 创建生成器
    generator = FewshotDatasetGenerator(
        base_training_dir=args.base_training_dir,
        novel_source_dir=args.novel_source_dir,
        output_root=args.output_dir,
        base_classes=base_classes,
        novel_classes=novel_classes,
        random_seed=args.seed,
        mixed=args.mixed,
        copy_base_query=args.copy_base_query,
        num_base_per_query=args.num_base_per_query
    )
    
    # 生成数据集
    results = generator.generate_fewshot_dataset(
        k_shot=args.k_shot,
        num_base_per_query=args.num_base_per_query,
        splits=['train','test']
    )
    
    print("\n生成完成!")
    for split, paths in results.items():
        print(f"\n{split}:")
        print(f"  Support: {paths['support_dir']}")
        print(f"  Query: {paths['query_dir']}")


if __name__ == '__main__':
    # 如果没有参数，显示使用示例
    import sys
    if len(sys.argv) == 1:
        print("\n" + "="*60)
        print("Few-shot数据集生成器")
        print("="*60)
        print("\n使用示例:")
        print("\n1. 生成5-shot数据集，novel classes为60-69:")
        print("   python fewshot_dataset_generator.py \\")
        print("       --base-training-dir /path/to/base_training \\")
        print("       --novel-source-dir /path/to/CW_split_folder \\")
        print("       --output-dir /path/to/fewshot_datasets \\")
        print("       --base-classes 0-59 \\")
        print("       --novel-classes 60-69 \\")
        print("       --k-shot 5")
        print("\n2. 自定义类别列表:")
        print("   python fewshot_dataset_generator.py \\")
        print("       --novel-classes 60,65,70,75,80 \\")
        print("       --k-shot 10")
        print("\n" + "="*60)
    else:
        main()

