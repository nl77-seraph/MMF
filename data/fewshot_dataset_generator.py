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
        """Get all sample file paths for a class."""
        # This assumes all single-tab source files are under novel_source_dir.
        # Update this logic if base-class source files are stored elsewhere.
        class_dir = os.path.join(self.novel_source_dir, split, str(class_id))
        
        if not os.path.exists(class_dir):
            # Try falling back to base_training_dir if the directory layout requires it.
            # class_dir = os.path.join(self.base_training_dir, split, str(class_id))
            pass

        if not os.path.exists(class_dir):
             return [] # Or raise an error, depending on the desired behavior.
        
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
        """Keep the original merge logic unchanged."""
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
        """Save the specified file lists as the support set."""
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
        """Synthesize and save one query sample."""
        try:
            novel_sample = self._load_sample(novel_file)
            novel_sample['label'] = novel_class
            
            base_samples = []
            base_class_ids = []
            for bf in base_files:
                b_data = self._load_sample(bf)
                # Infer the label from the path or metadata; callers are expected to ensure correctness.
                # For simplicity, we need the class_id corresponding to each base_file.
                # Since filenames may not contain class information, handling labels upstream is preferred.
                # Here we assume base_files are sampled randomly and recover their true labels from the path.
                # The logic below assumes a path structure like .../class_id/file.pkl.
                try:
                    b_label = int(os.path.basename(os.path.dirname(bf)))
                except:
                    b_label = -1 # Fallback
                b_data['label'] = b_label
                base_samples.append(b_data)
                base_class_ids.append(b_label)

            # Combine samples.
            all_samples = base_samples.copy()
            insert_pos = random.randint(0, len(all_samples))
            all_samples.insert(insert_pos, novel_sample)
            
            # Open world: add class 95 if enabled.
            if self.add_ow_class:
                # Get samples from class 95.
                ow_files = self._get_class_samples(95, 'train')
                if ow_files:
                    ow_file = random.choice(ow_files)
                    ow_sample = self._load_sample(ow_file)
                    ow_sample['label'] = 95  # Temporary marker removed later.
                    
                    # Insert the OW sample at a random position.
                    ow_insert_pos = random.randint(0, len(all_samples))
                    all_samples.insert(ow_insert_pos, ow_sample)
            
            merged = self._merge_traces(all_samples)
            
            # Remove class 95 from labels if an OW class was added, keeping it unlabeled.
            final_labels = [l for l in merged['labels'] if l != 95]
            
            labels_str = "_".join(map(str, final_labels))
            random_id = uuid.uuid4().hex[:8]
            filename = f"novel{novel_class}_{labels_str}_{random_id}.pkl"
            
            save_path = os.path.join(output_dir, filename)
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'time': merged['time'],
                    'data': merged['data'],
                    'labels': final_labels,  # Excludes class 95.
                    'novel_class': novel_class,
                    'base_classes': base_class_ids,
                    'metadata': merged['metadata'],
                    'has_ow': self.add_ow_class  # Marks whether the sample contains OW.
                }, f)
            return filename
        except Exception as e:
            print(f"[Error] Synthesis failed: {e}")
            return None

    def _process_train_split(self, k_shot: int, split: str):
        """
        Train split generation logic:
        1. Select K novel samples.
        2. Generate queries using selected novel samples plus random base samples, and record the used base samples.
        3. Build support with K novel samples plus used base samples and backfilled base samples.
        """
        print(f"\nProcessing TRAIN split ({k_shot}-shot)...")
        
        split_root = os.path.join(self.output_root, f'{k_shot}shot', split)
        support_dir = os.path.join(split_root, 'support_data')
        query_dir = os.path.join(split_root, 'query_data')
        os.makedirs(support_dir, exist_ok=True)
        os.makedirs(query_dir, exist_ok=True)

        # 1. Select K samples for each novel class.
        novel_support_candidates = {}
        for c in self.novel_classes:
            files = self._get_class_samples(c, split)
            if len(files) >= k_shot:
                novel_support_candidates[c] = random.sample(files, k_shot)
            else:
                novel_support_candidates[c] = files # Use all samples if there are fewer than k_shot.
        
        # Prepare the base pool.
        base_pool = {}
        for c in self.base_classes:
            base_pool[c] = self._get_class_samples(c, 'train') # Base classes always use training data.
            
        used_base_files = defaultdict(set) # Record base samples used in queries.
        query_filenames = []
        novel_counts = defaultdict(int)

        # 2. Generate queries.
        # The training phase usually follows K-shot strictly: each novel sample appears once as a query for K total queries.
        # If fewer than K novel samples are available, cycle through them so each novel class appears K times in queries.
        for novel_c, novel_files in tqdm(novel_support_candidates.items(), desc="Train Queries"):
            # Ensure K queries are generated.
            # Cycle through novel_files if there are fewer than K samples.
            for i in range(k_shot):
                novel_f = novel_files[i % len(novel_files)]
                
                # Randomly select base samples.
                if self.mixed:
                    num_base = random.randint(1, 3)
                else:
                    num_base = self.num_base_per_query
                
                selected_base_files = []
                # Randomly select base classes.
                chosen_base_classes = random.sample(self.base_classes, num_base)
                for bc in chosen_base_classes:
                    if base_pool[bc]:
                        bf = random.choice(base_pool[bc])
                        selected_base_files.append(bf)
                        used_base_files[bc].add(bf)
                
                # Synthesize.
                fname = self._synthesize_and_save_query(novel_f, selected_base_files, novel_c, query_dir)
                if fname:
                    query_filenames.append(fname)
                    novel_counts[novel_c] += 1

        # 3. Build and save the support set.
        final_support_map = {}
        
        # Novel support: the samples selected above.
        for c, files in novel_support_candidates.items():
            final_support_map[c] = files
            
        # Base Support: Used + Backfill
        for c in self.base_classes:
            current_files = list(used_base_files[c])
            # Randomly backfill if fewer than K files are available.
            if len(current_files) < k_shot:
                remaining = [f for f in base_pool[c] if f not in current_files]
                needed = k_shot - len(current_files)
                if len(remaining) >= needed:
                    current_files.extend(random.sample(remaining, needed))
                else:
                    current_files.extend(remaining)
            
            final_support_map[c] = current_files

        # Save support files.
        print("Saving Support Set...")
        self._save_support_set(final_support_map, support_dir)

        # Save statistics.
        self._save_stats(split, k_shot, query_filenames, novel_counts, final_support_map)

    def _process_test_split(self, k_shot: int, split: str):
        """
        Test split generation logic:
        1. Support: randomly sample K novel files and K base files as standard K-shot support.
        2. Query: generate at large scale, e.g. 2000 per class, using random synthesis not limited to support samples.
        """
        print(f"\nProcessing TEST split ({k_shot}-shot)...")
        
        split_root = os.path.join(self.output_root, f'{k_shot}shot', split)
        support_dir = os.path.join(split_root, 'support_data')
        query_dir = os.path.join(split_root, 'query_data')
        os.makedirs(support_dir, exist_ok=True)
        os.makedirs(query_dir, exist_ok=True)

        # 1. Build support. Randomly sample K files for both base and novel classes.
        support_map = {}
        
        # Novel Support
        novel_pool = {} # Cache all available files for later query generation.
        for c in self.novel_classes:
            files = self._get_class_samples(c, split)
            novel_pool[c] = files
            if len(files) >= k_shot:
                support_map[c] = random.sample(files, k_shot)
            else:
                support_map[c] = files
        
        # Base support. Random K samples are sufficient for test support; no backfill needed.
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

        # 2. Generate queries at large scale with random sampling.
        query_filenames = []
        novel_counts = defaultdict(int)
        
        # Configure the number of generated test queries.
        TEST_QUERY_PER_CLASS = 3000//len(self.novel_classes) 
        
        for novel_c in tqdm(self.novel_classes, desc="Test Queries"):
            available_novel_files = novel_pool[novel_c]
            if not available_novel_files:
                continue
                
            for _ in range(TEST_QUERY_PER_CLASS):
                # Randomly select a novel sample, not limited to support.
                novel_f = random.choice(available_novel_files)
                
                # Randomly select base samples.
                if self.mixed:
                    num_base = random.randint(1, 3)
                else:
                    num_base = self.num_base_per_query
                
                selected_base_files = []
                chosen_base_classes = random.sample(self.base_classes, num_base)
                for bc in chosen_base_classes:
                    if base_pool[bc]:
                        selected_base_files.append(random.choice(base_pool[bc]))
                
                # Synthesize.
                fname = self._synthesize_and_save_query(novel_f, selected_base_files, novel_c, query_dir)
                if fname:
                    query_filenames.append(fname)
                    novel_counts[novel_c] += 1

        # Save statistics.
        self._save_stats(split, k_shot, query_filenames, novel_counts, support_map)

    def _save_stats(self, split, k_shot, query_filenames, novel_counts, support_map):
        # Save the query list.
        json_path = os.path.join(self.output_root, f'{k_shot}shot', split, f'fewshot_query_{split}.json')
        with open(json_path, 'w') as f:
            json.dump(query_filenames, f, indent=2)
            
        # Save detailed statistics.
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
        """Entry point."""
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
    parser.add_argument('--copy-base-query', action='store_true') # Note: the new logic does not call this feature yet; add it in process if needed.
    parser.add_argument('--ow', action='store_true', help='Add Open World class 95 at a random position in each combination')
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