"""
Multi-label dataset generator.
Merges multiple traces based on timestamps and supports class-balanced combination sampling.
Avoids combination explosion through parameterized configuration.
Improvement: preserves combination order and strengthens balance compensation.
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
    Class-balanced combination sampler.
    Avoids C(n,k) combination explosion and keeps class distribution balanced through iterative sampling.
    """
    
    def __init__(self, num_classes: int = 60, random_seed: int = 42):
        """
        Args:
            num_classes: Total number of classes; defaults to 60 for 0-59.
            random_seed: Random seed.
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
        check_interval: int = 20,  # Check balance more frequently.
        balance_attempts: int = 20  # Number of attempts for each imbalance correction.
    ) -> Tuple[List[Tuple[int]], Dict]:
        """
        Generate class-balanced combinations without sorting, preserving order.
        
        Args:
            k: Number of classes per combination, i.e. tab count.
            target_num_combinations: Target number of combinations.
            max_std_ratio: Maximum std/mean ratio; defaults to 10%.
            max_iterations: Maximum number of iterations.
            check_interval: Balance check interval.
            balance_attempts: Number of compensation attempts for each imbalance.
            
        Returns:
            combinations: List of combinations, each a tuple of k classes in generated order.
            statistics: Statistics dictionary.
        """
        print(f"\nGenerating {target_num_combinations} balanced combinations for {k}-tab...")
        print(f"  - Class range: 0-{self.num_classes-1}")
        print(f"  - Balance constraint: std/mean < {max_std_ratio}")
        print(f"  - Check interval: every {check_interval} combinations")
        print(f"  - Preserve combination order: yes")
        
        combinations = set()
        class_counts = Counter()
        
        # Iteratively sample to keep classes balanced.
        iteration = 0
        total_balance_attempts = 0
        
        while len(combinations) < target_num_combinations and iteration < max_iterations:
            need_balance = False
            
            # Check balance periodically.
            if len(combinations) > 0 and len(combinations) % check_interval == 0:
                counts = [class_counts[i] for i in range(self.num_classes)]
                if any(counts):  # Avoid division by zero.
                    mean_count = np.mean(counts)
                    std_count = np.std(counts)
                    std_ratio = std_count / mean_count if mean_count > 0 else 0
                    
                    if std_ratio > max_std_ratio:
                        need_balance = True
                        # Print the current imbalance status.
                        print(f"  Imbalance detected at {len(combinations)} combinations: std/mean={std_ratio:.4f}")
            
            if need_balance:
                # Apply balance compensation.
                counts = [class_counts[i] for i in range(self.num_classes)]
                mean_count = np.mean(counts) if counts else 0
                
                # Find all classes below the mean.
                underrepresented = [
                    cls for cls in range(self.num_classes) 
                    if class_counts[cls] < mean_count
                ]
                
                # Try to generate combinations that include underrepresented classes.
                balanced_combos_added = 0
                for _ in range(balance_attempts):
                    if len(underrepresented) >= k:
                        # Select all classes from underrepresented classes.
                        selected = np.random.choice(underrepresented, size=k, replace=False)
                    elif len(underrepresented) > 0:
                        # Prioritize underrepresented classes, then fill with other classes.
                        n_under = len(underrepresented)
                        selected_under = np.random.choice(underrepresented, size=min(n_under, k), replace=False)
                        
                        # Fill from other classes.
                        other_classes = [c for c in range(self.num_classes) if c not in selected_under]
                        n_need = k - len(selected_under)
                        
                        # Prioritize other classes that appear less often.
                        other_classes.sort(key=lambda x: class_counts[x])
                        selected_other = np.random.choice(other_classes[:len(other_classes)//2], 
                                                        size=n_need, replace=False)
                        
                        selected = np.concatenate([selected_under, selected_other])
                        np.random.shuffle(selected)  # Shuffle order.
                    else:
                        # Fall back to normal random sampling if no class is underrepresented.
                        selected = np.random.choice(self.num_classes, size=k, replace=False)
                    
                    # Do not sort; keep the random order.
                    combo = tuple(selected.tolist())
                    
                    if combo not in combinations:
                        combinations.add(combo)
                        for cls in combo:
                            class_counts[cls] += 1
                        balanced_combos_added += 1
                        
                        # Exit early after adding enough balanced combinations.
                        if balanced_combos_added >= check_interval // 2:
                            break
                
                total_balance_attempts += 1
                print(f"    -> Added {balanced_combos_added} balanced combinations")
                
            else:
                # Normal random combination generation without sorting.
                selected = np.random.choice(self.num_classes, size=k, replace=False)
                combo = tuple(selected.tolist())  # Preserve generated order.
                
                if combo not in combinations:
                    combinations.add(combo)
                    for cls in combo:
                        class_counts[cls] += 1
            
            iteration += 1
            
            # Show progress.
            if len(combinations) % 500 == 0:
                counts = [class_counts[i] for i in range(self.num_classes)]
                mean_count = np.mean(counts)
                std_count = np.std(counts)
                std_ratio = std_count / mean_count if mean_count > 0 else 0
                print(f"  Progress: {len(combinations)}/{target_num_combinations}, "
                      f"std/mean: {std_ratio:.4f}")
        
        # Convert to a list.
        combinations_list = list(combinations)
        
        # Compute final statistics.
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
        
        print(f"\n[OK] Combination generation complete:")
        print(f"  - Generated combinations: {len(combinations_list)}")
        print(f"  - Class frequency: mean={mean_count:.2f}, std={std_count:.2f}")
        print(f"  - Balance: std/mean={std_ratio:.4f}")
        print(f"  - Frequency range: [{min(counts)}, {max(counts)}]")
        print(f"  - Balance compensation attempts: {total_balance_attempts}")
        
        return combinations_list, statistics


class TimeBasedMerger:
    """
    Timestamp-based multi-trace merger.
    Controls overlap through start delays without breaking each trace's internal structure.
    
    Key point: handles boundary cases where trace time spans contain each other.
    """
    
    def __init__(self, overlap_range: Tuple[float, float] = (0.0, 0.4)):
        """
        Args:
            overlap_range: Overlap ratio range [min, max].
        """
        self.overlap_range = overlap_range
        
    def merge_traces_with_overlap(
        self, 
        traces: List[Dict],
        overlap_ratios: Optional[List[float]] = None
    ) -> Dict:
        """
        Merge multiple traces based on timestamps and overlap ratios.
        
        Args:
            traces: List of traces; each trace is {'time': array, 'data': array, 'label': int}.
            overlap_ratios: List of overlap ratios with length len(traces)-1.
                          If None, ratios are generated randomly.
                          
        Returns:
            merged_trace: Merged trace dictionary.
        """
        num_traces = len(traces)
        
        # Generate or validate overlap ratios.
        if overlap_ratios is None:
            overlap_ratios = [
                random.uniform(self.overlap_range[0], self.overlap_range[1])
                for _ in range(num_traces - 1)
            ]
        else:
            assert len(overlap_ratios) == num_traces - 1, \
                f"Expected {num_traces-1} overlap ratios, got {len(overlap_ratios)}"
        
        # Calculate the total duration of each trace.
        durations = [trace['time'][-1] - trace['time'][0] for trace in traces]
        
        # Calculate the start time of each trace.
        start_times = [0.0]  # The first trace starts at 0.
        
        for i in range(1, num_traces):
            # Previous start time + previous duration x (1 - overlap ratio).
            prev_start = start_times[i-1]
            prev_duration = durations[i-1]
            overlap_ratio = overlap_ratios[i-1]
            
            # Start time of the new trace.
            new_start = prev_start + prev_duration * (1.0 - overlap_ratio)
            start_times.append(new_start)
        
        # Adjust timestamps for each trace and merge.
        all_packets = []  # (timestamp, direction, trace_id)
        
        for trace_id, trace in enumerate(traces):
            adjusted_times = trace['time'] + start_times[trace_id]
            
            for t, d in zip(adjusted_times, trace['data']):
                all_packets.append((t, d, trace_id))
        
        # Sort by timestamp.
        all_packets.sort(key=lambda x: x[0])
        
        # Extract sorted time and data.
        merged_time = np.array([p[0] for p in all_packets])
        merged_data = np.array([p[1] for p in all_packets])
        
        # Generate the label list while preserving original order.
        labels = [trace['label'] for trace in traces]
        
        # Calculate statistics.
        total_duration = merged_time[-1] - merged_time[0]
        
        merged_trace = {
            'time': merged_time,
            'data': merged_data,
            'labels': labels,  # Multi-labels in order.
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
        Validate the correctness of the merge result.
        
        Args:
            merged_trace: Merged trace.
            verbose: Whether to print detailed information.
            
        Returns:
            is_valid: Whether the result is valid.
        """
        time = merged_trace['time']
        data = merged_trace['data']
        metadata = merged_trace['metadata']
        
        # Check 1: timestamps are monotonically increasing.
        is_monotonic = np.all(np.diff(time) >= 0)
        
        # Check 2: data lengths are consistent.
        length_match = len(time) == len(data) == metadata['merged_length']
        
        # Check 3: the number of labels matches the number of traces.
        labels_match = len(merged_trace['labels']) == metadata['num_traces']
        
        is_valid = is_monotonic and length_match and labels_match
        
        if verbose or not is_valid:
            print(f"  Validation result:")
            print(f"    - Timestamps monotonically increasing: {'PASS' if is_monotonic else 'FAIL'}")
            print(f"    - Data lengths match: {'PASS' if length_match else 'FAIL'}")
            print(f"    - Label count is correct: {'PASS' if labels_match else 'FAIL'}")
            if not is_monotonic:
                non_monotonic = np.where(np.diff(time) < 0)[0]
                print(f"    - Non-monotonic positions: {non_monotonic[:5]}...")
        
        return is_valid


class MultiTabDatasetGenerator:
    """
    Complete multi-label dataset generator.
    Supports fixed-tab mode, generating one dataset with a single tab count at a time.
    Improvement: preserves combination order and strengthens balance compensation.
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
            source_root: Root directory for single-label data sources.
            output_root: Root output directory for multi-label datasets.
            num_classes: Total number of classes.
            overlap_range: Overlap ratio range.
            random_seed: Random seed.
        """
        self.source_root = source_root
        self.output_root = output_root
        self.num_classes = num_classes
        self.overlap_range = overlap_range
        self.random_seed = random_seed
        
        # Initialize components.
        self.sampler = CombinationSampler(num_classes, random_seed)
        self.merger = TimeBasedMerger(overlap_range)
        
        # Set random seeds.
        random.seed(random_seed)
        np.random.seed(random_seed)
        
    def load_single_label_sample(self, class_id: int, split: str) -> Dict:
        """
        Load a single-label sample.
        
        Args:
            class_id: Class ID.
            split: 'train' or 'test'.
            
        Returns:
            sample: {'time': array, 'data': array, 'label': int}
        """
        class_dir = os.path.join(self.source_root, split, str(class_id))
        
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Class directory does not exist: {class_dir}")
        
        # Get all files for this class.
        files = [f for f in os.listdir(class_dir) if f.endswith('.pkl')]
        
        if len(files) == 0:
            raise ValueError(f"Class {class_id} has no samples in {split}")
        
        # Randomly select one file.
        selected_file = random.choice(files)
        file_path = os.path.join(class_dir, selected_file)
        
        # Load data.
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
        check_interval: int = 20,  # Configurable check interval.
        balance_attempts: int = 20,  # Configurable number of compensation attempts.
        add_ow_class: bool = False  # Whether to add OW class 95.
    ) -> str:
        """
        Generate a dataset with a fixed tab count.
        
        Args:
            num_tabs: Number of tabs, from 2 to 5.
            num_combinations: Number of combinations.
            samples_per_combo: Number of generated samples per combination.
            split: 'train' or 'test'.
            dataset_name: Dataset name; defaults to "{num_tabs}tab".
            check_interval: Balance check interval.
            balance_attempts: Number of compensation attempts for each imbalance.
            add_ow_class: Whether to add OW class 95 at a random position in each combination.
            
        Returns:
            output_dir: Output directory path.
        """
        if dataset_name is None:
            dataset_name = f"{num_tabs}tab"
        
        print(f"\n{'='*60}")
        print(f"Generating {dataset_name} dataset - {split} split")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  - Tab count: {num_tabs}")
        print(f"  - Combinations: {num_combinations}")
        print(f"  - Samples per combination: {samples_per_combo}")
        print(f"  - Total samples: {num_combinations * samples_per_combo}")
        print(f"  - Overlap ratio range: {self.overlap_range}")
        print(f"  - Preserve combination order: yes")
        print(f"  - Add OW class 95: {'yes' if add_ow_class else 'no'}")
        
        # Create output directories.
        output_dir = os.path.join(self.output_root, dataset_name, split)
        query_dir = os.path.join(output_dir, "query_data")
        support_dir = os.path.join(output_dir, "support_data")
        
        os.makedirs(query_dir, exist_ok=True)
        os.makedirs(support_dir, exist_ok=True)
        
        # Generate class combinations.
        combinations, combo_stats = self.sampler.generate_balanced_combinations(
            k=num_tabs,
            target_num_combinations=num_combinations,
            check_interval=check_interval,
            balance_attempts=balance_attempts
        )
        
        # Add class 95 at a random position in each combination if OW mode is enabled.
        if add_ow_class:
            print(f"\n[OW mode] Adding class 95 at a random position in each combination...")
            combinations_with_ow = []
            for combo in combinations:
                combo_list = list(combo)
                # Randomly select an insertion position between 0 and len(combo_list).
                insert_pos = random.randint(0, len(combo_list))
                combo_list.insert(insert_pos, 95)
                combinations_with_ow.append(tuple(combo_list))
            combinations = combinations_with_ow
            print(f"[OK] Added OW class 95 to {len(combinations)} combinations")
            print(f"     Example: {combinations[0]} (class 95 at position {list(combinations[0]).index(95)})")
        
        # Save combination list.
        combinations_file = os.path.join(
            self.output_root, dataset_name, f"combinations_{split}.json"
        )
        with open(combinations_file, 'w') as f:
            json.dump({
                'combinations': [[int(x) for x in c] for c in combinations],
                'statistics': combo_stats,
                'ow_enabled': add_ow_class
            }, f, indent=2)
        print(f"[OK] Combination list saved: {combinations_file}")
        
        # Generate samples.
        print(f"\nGenerating samples...")
        query_filenames = []
        class_sample_usage = defaultdict(set)  # Record samples used by each class.
        generation_stats = {
            'total_samples': 0,
            'total_length': 0,
            'lengths': [],
            'durations': [],
            'failed_samples': 0
        }
        
        for combo_idx, combo in enumerate(tqdm(combinations, desc=f"Generating {split} samples")):
            for sample_idx in range(samples_per_combo):
                try:
                    # Load single-label samples for each class in combination order.
                    traces = []
                    source_files = []
                    
                    for class_id in combo:  # Preserve the original combo order.
                        sample = self.load_single_label_sample(class_id, split)
                        traces.append(sample)
                        source_files.append((class_id, sample['source_file'], sample['source_path']))
                    
                    # Merge traces.
                    merged_trace = self.merger.merge_traces_with_overlap(traces)
                    
                    # Validate the merge result.
                    if not self.merger.validate_merge(merged_trace, verbose=False):
                        print(f"[WARN] Sample validation failed, skipping: combo={combo}, sample={sample_idx}")
                        generation_stats['failed_samples'] += 1
                        continue
                    
                    # Generate filename: class1_class2_..._randomID.pkl, preserving order.
                    labels_str = "_".join(map(str, combo))  # Do not sort.
                    random_id = uuid.uuid4().hex[:8]
                    filename = f"{labels_str}_{random_id}.pkl"
                    
                    # Save the multi-label sample.
                    query_path = os.path.join(query_dir, filename)
                    with open(query_path, 'wb') as f:
                        pickle.dump({
                            'time': merged_trace['time'],
                            'data': merged_trace['data'],
                            'labels': merged_trace['labels'],  # Ordered label list.
                            'metadata': merged_trace['metadata']
                        }, f)
                    
                    query_filenames.append(filename)
                    
                    # Copy used single-label samples to support_data.
                    for class_id, source_file, source_path in source_files:
                        # Copy each sample only on first use.
                        if source_file not in class_sample_usage[class_id]:
                            class_support_dir = os.path.join(support_dir, str(class_id))
                            os.makedirs(class_support_dir, exist_ok=True)
                            
                            dest_path = os.path.join(class_support_dir, source_file)
                            if not os.path.exists(dest_path):
                                shutil.copy2(source_path, dest_path)
                            
                            class_sample_usage[class_id].add(source_file)
                    
                    # Statistics.
                    generation_stats['total_samples'] += 1
                    generation_stats['total_length'] += len(merged_trace['data'])
                    generation_stats['lengths'].append(len(merged_trace['data']))
                    generation_stats['durations'].append(merged_trace['metadata']['merged_duration'])
                    
                except Exception as e:
                    print(f"\n[ERROR] Failed to generate sample: combo={combo}, sample={sample_idx}")
                    print(f"   Error: {e}")
                    generation_stats['failed_samples'] += 1
                    continue
        
        # Save the query filename list.
        query_json_path = os.path.join(output_dir, f"{dataset_name}_{split}.json")
        with open(query_json_path, 'w') as f:
            json.dump(query_filenames, f, indent=2)
        print(f"\n[OK] Query index saved: {query_json_path}")
        
        # Generate the statistics report.
        self._generate_statistics_report(
            dataset_name, split, num_tabs, combo_stats, 
            generation_stats, class_sample_usage
        )
        
        print(f"\n{'='*60}")
        print(f"[OK] {dataset_name}-{split} dataset generation complete!")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        print(f"Total samples: {generation_stats['total_samples']}")
        print(f"Failed samples: {generation_stats['failed_samples']}")
        
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
        """Generate the statistics report."""
        # Calculate class distribution in generated samples.
        class_distribution = combo_stats['class_distribution']
        
        # Calculate sample length statistics.
        lengths = generation_stats['lengths']
        durations = generation_stats['durations']
        
        statistics = {
            'dataset_name': dataset_name,
            'split': split,
            'num_tabs': num_tabs,
            'target_length': 30000,
            'preserve_order': True,  # Indicates that order is preserved.
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
        
        # Save the statistics report.
        stats_file = os.path.join(
            self.output_root, dataset_name, f"statistics_{split}.json"
        )
        with open(stats_file, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        print(f"\n[Statistics report]:")
        print(f"  - Sample length: mean={statistics['samples']['avg_length']:.0f}, "
              f"std={statistics['samples']['std_length']:.0f}")
        print(f"  - Sample duration: mean={statistics['samples']['avg_duration']:.2f}s, "
              f"std={statistics['samples']['std_duration']:.2f}s")
        print(f"  - Single-label samples used: {statistics['support_data']['total_unique_samples']}")
        print(f"  - Balance compensation attempts: {statistics['combinations']['total_balance_attempts']}")
        print(f"[OK] Statistics report saved: {stats_file}")


def test_time_based_merger():
    """Test the timestamp merge algorithm."""
    print("\n" + "="*60)
    print("Testing TimeBasedMerger - timestamp merge algorithm")
    print("="*60)
    
    merger = TimeBasedMerger(overlap_range=(0.0, 0.4))
    
    # Create three simulated traces.
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
    
    # Test 1: normal overlap.
    print("\nTest 1: normal overlap (30%, 20%)")
    merged = merger.merge_traces_with_overlap(
        [trace1, trace2, trace3],
        overlap_ratios=[0.3, 0.2]
    )
    
    print(f"Result:")
    print(f"  - Original lengths: {len(trace1['data'])}, {len(trace2['data'])}, {len(trace3['data'])}")
    print(f"  - Merged length: {len(merged['data'])}")
    print(f"  - Label order: {merged['labels']}")  # This is ordered.
    print(f"  - Start times: {merged['metadata']['start_times']}")
    print(f"  - Total duration: {merged['metadata']['merged_duration']:.3f}s")
    
    is_valid = merger.validate_merge(merged, verbose=True)
    print(f"  - Validation result: {'PASS' if is_valid else 'FAIL'}")
    
    print("\n[OK] TimeBasedMerger test complete!")


def test_combination_order():
    """Test combination order preservation."""
    print("\n" + "="*60)
    print("Testing combination order preservation")
    print("="*60)
    
    sampler = CombinationSampler(num_classes=10, random_seed=42)
    
    # Generate some combinations.
    combinations, stats = sampler.generate_balanced_combinations(
        k=3,
        target_num_combinations=50,
        check_interval=10,
        balance_attempts=10
    )
    
    # Check whether combinations with different orders exist.
    print("\nFirst 10 combinations, preserving generated order:")
    for i, combo in enumerate(combinations[:10]):
        print(f"  {i+1}: {combo}")
    
    # Verify that combinations were not sorted.
    sorted_combos = [tuple(sorted(combo)) for combo in combinations]
    identical_count = sum(1 for c1, c2 in zip(combinations, sorted_combos) if c1 == c2)
    
    print(f"\nOrder validation:")
    print(f"  - Total combinations: {len(combinations)}")
    print(f"  - Combinations identical to sorted version: {identical_count}")
    print(f"  - Ratio preserving original order: {(len(combinations)-identical_count)/len(combinations)*100:.1f}%")
    
    # Find combination pairs with the same elements but different orders.
    element_sets = {}
    for combo in combinations:
        element_set = frozenset(combo)
        if element_set not in element_sets:
            element_sets[element_set] = []
        element_sets[element_set].append(combo)
    
    print(f"\nExamples with the same elements but different orders:")
    count = 0
    for element_set, combos in element_sets.items():
        if len(combos) > 1:
            print(f"  Element set {set(element_set)}: {combos}")
            count += 1
            if count >= 3:  # Show only the first three examples.
                break
    
    if count == 0:
        print("  No combinations with the same elements but different orders were found; this is normal.")
    
    print("\n[OK] Combination order preservation test complete!")


if __name__ == "__main__":
    # Test timestamp merging.
    test_time_based_merger()
    
    # Test combination order preservation.
    test_combination_order()
    
    print("\n" + "="*60)
    print("Improved multi-label dataset generator is ready")
    print("="*60)
    print("\nCore improvements:")
    print("  1. Removed sorted operations to preserve combination order")
    print("  2. More frequent balance checks, every 20 combinations by default")
    print("  3. Stronger balance compensation, up to 20 combinations per compensation step")
    print("  4. Prioritize underrepresented classes and keep compensating until the target is reached")
    print("\nUsage example:")
    print("  generator = MultiTabDatasetGenerator()")
    print("  generator.generate_dataset(")
    print("      num_tabs=2,")
    print("      num_combinations=100,")
    print("      samples_per_combo=5,")
    print("      split='train',")
    print("      check_interval=20,    # customizable")
    print("      balance_attempts=20   # customizable")
    print("  )")