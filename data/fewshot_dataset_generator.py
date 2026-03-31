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
    def __init__(
        self,
        base_training_dir: str,
        novel_source_dir: str,
        output_root: str,
        base_classes: List[int] = None,
        novel_classes: List[int] = None,
        overlap_range: Tuple[float, float] = (0.0, 0.4),
        random_seed: int = 42,
        mixed: bool = False,
        copy_base_query: bool = False,
        num_base_per_query: int = 2,
        add_ow_class: bool = False
    ):
        self.base_training_dir = base_training_dir
        self.novel_source_dir = novel_source_dir
        self.output_root = output_root
        
        self.base_classes = base_classes if base_classes else list(range(60))
        self.novel_classes = novel_classes if novel_classes else []
        
        self.overlap_range = overlap_range
        self.random_seed = random_seed
        self.mixed = mixed
        self.copy_base_query = copy_base_query
        self.num_base_per_query = num_base_per_query
        self.add_ow_class = add_ow_class
        
        if self.mixed:
            self.output_root = os.path.join(self.output_root, 'mixed_tab')
        else:
            self.output_root = os.path.join(self.output_root, f'{num_base_per_query+1}_tab')
            
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        self.all_classes = sorted(self.base_classes + self.novel_classes)

    def _get_class_samples(self, class_id: int, split: str = 'train') -> List[str]:
        """获取某个类别的所有样本文件路径"""
        # 注意：这里假设所有single-tab源文件都在 novel_source_dir 下
        # 如果base类的源文件在其他地方，请修改此处
        class_dir = os.path.join(self.novel_source_dir, split, str(class_id))
        
        if not os.path.exists(class_dir):
            # 尝试回退到 base_training_dir (视你的目录结构而定，这里保留原逻辑)
            # class_dir = os.path.join(self.base_training_dir, split, str(class_id))
            pass

        if not os.path.exists(class_dir):
             return [] # 或者 raise error，视情况而定
        
        files = [
            os.path.join(class_dir, f) 
            for f in os.listdir(class_dir) 
            if f.endswith('.pkl')
        ]
        return files

    def _load_sample(self, file_path: str) -> Dict:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def _merge_traces(self, traces: List[Dict], overlap_ratios: Optional[List[float]] = None) -> Dict:
        """(保持原有的合并逻辑不变)"""
        num_traces = len(traces)
        if overlap_ratios is None:
            overlap_ratios = [
                random.uniform(self.overlap_range[0], self.overlap_range[1])
                for _ in range(num_traces - 1)
            ]
        
        durations = [trace['time'][-1] - trace['time'][0] for trace in traces]
        start_times = [0.0]
        for i in range(1, num_traces):
            prev_start = start_times[i-1]
            prev_duration = durations[i-1]
            overlap_ratio = overlap_ratios[i-1]
            new_start = prev_start + prev_duration * (1.0 - overlap_ratio)
            start_times.append(new_start)
        
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
                'start_times': start_times
            }
        }

    def _save_support_set(self, files_map: Dict[int, List[str]], output_dir: str) -> Dict:
        """通用方法：将指定的文件列表保存为Support集"""
        stats = {}
        for class_id, files in files_map.items():
            class_dir = os.path.join(output_dir, str(class_id))
            os.makedirs(class_dir, exist_ok=True)
            for src in files:
                dst = os.path.join(class_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
            stats[str(class_id)] = len(files)
        return stats

    def _synthesize_and_save_query(self, 
                                   novel_file: str, 
                                   base_files: List[str], 
                                   novel_class: int, 
                                   output_dir: str) -> str:
        """通用方法：合成并保存一个Query样本"""
        try:
            novel_sample = self._load_sample(novel_file)
            novel_sample['label'] = novel_class
            
            base_samples = []
            base_class_ids = []
            for bf in base_files:
                b_data = self._load_sample(bf)
                # 从路径或元数据推断label，这里假设调用者保证了正确性
                # 为了简单，我们需要知道base_file对应的class_id。
                # 由于文件名可能不包含class信息，最好是在外部处理好label
                # 这里我们假设base_files是随机选的，我们需要知道他们的真实label
                # 下面的逻辑假设路径结构是 .../class_id/file.pkl
                try:
                    b_label = int(os.path.basename(os.path.dirname(bf)))
                except:
                    b_label = -1 # Fallback
                b_data['label'] = b_label
                base_samples.append(b_data)
                base_class_ids.append(b_label)

            # 组合
            all_samples = base_samples.copy()
            insert_pos = random.randint(0, len(all_samples))
            all_samples.insert(insert_pos, novel_sample)
            
            # Open World: 添加类别95（如果启用）
            if self.add_ow_class:
                # 获取类别95的样本
                ow_files = self._get_class_samples(95, 'train')
                if ow_files:
                    ow_file = random.choice(ow_files)
                    ow_sample = self._load_sample(ow_file)
                    ow_sample['label'] = 95  # 临时标记，后续会移除
                    
                    # 在随机位置插入OW样本
                    ow_insert_pos = random.randint(0, len(all_samples))
                    all_samples.insert(ow_insert_pos, ow_sample)
            
            merged = self._merge_traces(all_samples)
            
            # 如果添加了OW类别，从labels中移除95（保持无标签）
            final_labels = [l for l in merged['labels'] if l != 95]
            
            labels_str = "_".join(map(str, final_labels))
            random_id = uuid.uuid4().hex[:8]
            filename = f"novel{novel_class}_{labels_str}_{random_id}.pkl"
            
            save_path = os.path.join(output_dir, filename)
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'time': merged['time'],
                    'data': merged['data'],
                    'labels': final_labels,  # 不包含类别95
                    'novel_class': novel_class,
                    'base_classes': base_class_ids,
                    'metadata': merged['metadata'],
                    'has_ow': self.add_ow_class  # 标记是否包含OW
                }, f)
            return filename
        except Exception as e:
            print(f"[Error] Synthesis failed: {e}")
            return None

    def _process_train_split(self, k_shot: int, split: str):
        """
        Train Split 生成逻辑:
        1. 选定 K 个 Novel 样本
        2. 生成 Query (使用选定的 Novel + 随机 Base)，并记录用到的 Base
        3. 构建 Support (K Novel + Used Base + 补齐 Base)
        """
        print(f"\nProcessing TRAIN split ({k_shot}-shot)...")
        
        split_root = os.path.join(self.output_root, f'{k_shot}shot', split)
        support_dir = os.path.join(split_root, 'support_data')
        query_dir = os.path.join(split_root, 'query_data')
        os.makedirs(support_dir, exist_ok=True)
        os.makedirs(query_dir, exist_ok=True)

        # 1. 为每个 Novel 类选定 K 个样本
        novel_support_candidates = {}
        for c in self.novel_classes:
            files = self._get_class_samples(c, split)
            if len(files) >= k_shot:
                novel_support_candidates[c] = random.sample(files, k_shot)
            else:
                novel_support_candidates[c] = files # 样本不足全选
        
        # 准备 Base 池
        base_pool = {}
        for c in self.base_classes:
            base_pool[c] = self._get_class_samples(c, 'train') # Base 始终用 train 数据
            
        used_base_files = defaultdict(set) # 记录 query 中用到的 base 样本
        query_filenames = []
        novel_counts = defaultdict(int)

        # 2. 生成 Query
        # Train 阶段通常严格按照 K-shot，即每个 Novel 样本作为 Query 出现 1 次 (总共 K 次)
        # 或者是 K 个 Novel 样本循环使用。根据你的描述 "每个novel class在query中出现总次数=K"
        for novel_c, novel_files in tqdm(novel_support_candidates.items(), desc="Train Queries"):
            # 确保生成 K 个 query
            # 如果 novel_files 少于 K (样本不足)，需要循环采样
            for i in range(k_shot):
                novel_f = novel_files[i % len(novel_files)]
                
                # 随机选择 Base
                if self.mixed:
                    num_base = random.randint(1, 3)
                else:
                    num_base = self.num_base_per_query
                
                selected_base_files = []
                # 随机选 base 类
                chosen_base_classes = random.sample(self.base_classes, num_base)
                for bc in chosen_base_classes:
                    if base_pool[bc]:
                        bf = random.choice(base_pool[bc])
                        selected_base_files.append(bf)
                        used_base_files[bc].add(bf)
                
                # 合成
                fname = self._synthesize_and_save_query(novel_f, selected_base_files, novel_c, query_dir)
                if fname:
                    query_filenames.append(fname)
                    novel_counts[novel_c] += 1

        # 3. 构建并保存 Support
        final_support_map = {}
        
        # Novel Support: 刚才选定的那些
        for c, files in novel_support_candidates.items():
            final_support_map[c] = files
            
        # Base Support: Used + Backfill
        for c in self.base_classes:
            current_files = list(used_base_files[c])
            # 如果不足 K 个，随机补齐
            if len(current_files) < k_shot:
                remaining = [f for f in base_pool[c] if f not in current_files]
                needed = k_shot - len(current_files)
                if len(remaining) >= needed:
                    current_files.extend(random.sample(remaining, needed))
                else:
                    current_files.extend(remaining)
            
            final_support_map[c] = current_files

        # 保存 Support 文件
        print("Saving Support Set...")
        self._save_support_set(final_support_map, support_dir)

        # 保存统计
        self._save_stats(split, k_shot, query_filenames, novel_counts, final_support_map)

    def _process_test_split(self, k_shot: int, split: str):
        """
        Test Split 生成逻辑:
        1. Support: 随机 K 个 Novel，随机 K 个 Base (标准 K-shot)
        2. Query: 大规模生成 (e.g. 2000 per class)，不局限于 Support，随机采样合成
        """
        print(f"\nProcessing TEST split ({k_shot}-shot)...")
        
        split_root = os.path.join(self.output_root, f'{k_shot}shot', split)
        support_dir = os.path.join(split_root, 'support_data')
        query_dir = os.path.join(split_root, 'query_data')
        os.makedirs(support_dir, exist_ok=True)
        os.makedirs(query_dir, exist_ok=True)

        # 1. 构建 Support (Base + Novel 都是随机采样 K 个)
        support_map = {}
        
        # Novel Support
        novel_pool = {} # 缓存所有可用文件，用于后面Query生成
        for c in self.novel_classes:
            files = self._get_class_samples(c, split)
            novel_pool[c] = files
            if len(files) >= k_shot:
                support_map[c] = random.sample(files, k_shot)
            else:
                support_map[c] = files
        
        # Base Support (Test support 里的 base 也是随机 K 个即可，无需 backfill)
        base_pool = {}
        for c in self.base_classes:
            files = self._get_class_samples(c, 'train')
            base_pool[c] = files
            if len(files) >= k_shot:
                support_map[c] = random.sample(files, k_shot)
            else:
                support_map[c] = files
        
        print("Saving Support Set...")
        self._save_support_set(support_map, support_dir)

        # 2. 生成 Query (Large Scale, Random)
        query_filenames = []
        novel_counts = defaultdict(int)
        
        # 配置测试集生成的数量
        TEST_QUERY_PER_CLASS = 3000//len(self.novel_classes) 
        
        for novel_c in tqdm(self.novel_classes, desc="Test Queries"):
            available_novel_files = novel_pool[novel_c]
            if not available_novel_files:
                continue
                
            for _ in range(TEST_QUERY_PER_CLASS):
                # 随机选一个 Novel 样本 (不局限于 support)
                novel_f = random.choice(available_novel_files)
                
                # 随机选 Base
                if self.mixed:
                    num_base = random.randint(1, 3)
                else:
                    num_base = self.num_base_per_query
                
                selected_base_files = []
                chosen_base_classes = random.sample(self.base_classes, num_base)
                for bc in chosen_base_classes:
                    if base_pool[bc]:
                        selected_base_files.append(random.choice(base_pool[bc]))
                
                # 合成
                fname = self._synthesize_and_save_query(novel_f, selected_base_files, novel_c, query_dir)
                if fname:
                    query_filenames.append(fname)
                    novel_counts[novel_c] += 1

        # 保存统计
        self._save_stats(split, k_shot, query_filenames, novel_counts, support_map)

    def _save_stats(self, split, k_shot, query_filenames, novel_counts, support_map):
        # 保存 query 列表
        json_path = os.path.join(self.output_root, f'{k_shot}shot', split, f'fewshot_query_{split}.json')
        with open(json_path, 'w') as f:
            json.dump(query_filenames, f, indent=2)
            
        # 保存详细统计
        stats = {
            'k_shot': k_shot,
            'split': split,
            'total_queries': len(query_filenames),
            'novel_counts': dict(novel_counts),
            'support_counts': {str(k): len(v) for k, v in support_map.items()}
        }
        with open(os.path.join(self.output_root, f'{k_shot}shot', split, 'stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

    def generate_fewshot_dataset(self, k_shot: int, splits: List[str] = ['train']):
        """入口函数"""
        for split in splits:
            if split == 'train':
                self._process_train_split(k_shot, split)
            elif split == 'test':
                self._process_test_split(k_shot, split)
            else:
                print(f"[Warn] Unknown split: {split}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-training-dir', type=str, required=True)
    parser.add_argument('--novel-source-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--base-classes', type=str, default='0-59')
    parser.add_argument('--novel-classes', type=str, required=True)
    parser.add_argument('--k-shot', type=int, default=5)
    parser.add_argument('--num-base-per-query', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mixed', action='store_true')
    parser.add_argument('--copy-base-query', action='store_true') # 注意：新逻辑暂未包含此功能的调用，如需要需在process中添加
    parser.add_argument('--ow', action='store_true', help='添加Open World类别95到每个组合的随机位置')
    args = parser.parse_args()
    
    def parse_range(s):
        if '-' in s:
            start, end = map(int, s.split('-'))
            return list(range(start, end + 1))
        return [int(x) for x in s.split(',')]

    generator = FewshotDatasetGenerator(
        base_training_dir=args.base_training_dir,
        novel_source_dir=args.novel_source_dir,
        output_root=args.output_dir,
        base_classes=parse_range(args.base_classes),
        novel_classes=parse_range(args.novel_classes),
        random_seed=args.seed,
        mixed=args.mixed,
        copy_base_query=args.copy_base_query,
        num_base_per_query=args.num_base_per_query,
        add_ow_class=args.ow
    )
    
    generator.generate_fewshot_dataset(args.k_shot, splits=['train', 'test'])

if __name__ == '__main__':
    main()