"""
Complete multi-label dataset generation script.
Supports fixed-tab datasets with 2-5 tabs.
Supports mixed-tab datasets with 2-5 tabs combined.
Parameterized configuration supports both small-scale tests and large-scale generation.
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.multi_tab_generator import MultiTabDatasetGenerator


SMALL_SCALE_CONFIG = {
    '2tab': {'num_combinations': 3000, 'samples_per_combo': 20},
    '3tab': {'num_combinations': 20000, 'samples_per_combo': 3},
    '4tab': {'num_combinations': 30000, 'samples_per_combo': 2},
    '5tab': {'num_combinations': 60000, 'samples_per_combo': 1},
    '2tab_test': {'num_combinations': 1000, 'samples_per_combo': 10},
    '3tab_test': {'num_combinations': 10000, 'samples_per_combo': 2},
    '4tab_test': {'num_combinations': 10000, 'samples_per_combo': 2},
    '5tab_test': {'num_combinations': 10000, 'samples_per_combo': 2},
}

# Medium-scale configuration.
MEDIUM_SCALE_CONFIG = {
    '2tab': {'num_combinations': 2000, 'samples_per_combo': 10},
    '3tab': {'num_combinations': 20000, 'samples_per_combo': 5},
    '4tab': {'num_combinations': 35000, 'samples_per_combo': 3},
    '5tab': {'num_combinations': 50000, 'samples_per_combo': 3},
    '2tab_test': {'num_combinations': 1000, 'samples_per_combo': 10},
    '3tab_test': {'num_combinations': 10000, 'samples_per_combo': 5},
    '4tab_test': {'num_combinations': 17500, 'samples_per_combo': 3},
    '5tab_test': {'num_combinations': 25000, 'samples_per_combo': 3},

}

# Large-scale configuration for paper experiments.
LARGE_SCALE_CONFIG = {
    '2tab': {'num_combinations': 2000, 'samples_per_combo': 20},
    '3tab': {'num_combinations': 20000, 'samples_per_combo': 10},
    '4tab': {'num_combinations': 35000, 'samples_per_combo': 7},
    '5tab': {'num_combinations': 50000, 'samples_per_combo': 5},
    '2tab_test': {'num_combinations': 1000, 'samples_per_combo': 10},
    '3tab_test': {'num_combinations': 10000, 'samples_per_combo': 5},
    '4tab_test': {'num_combinations': 17500, 'samples_per_combo': 3},
    '5tab_test': {'num_combinations': 25000, 'samples_per_combo': 3},
}

# Mixed-tab configuration based on the large-scale setup, with combinations per tab divided by 4.
# Keep the total sample count around 200k.
MIXED_TAB_CONFIG = {
    'small': {
        '2tab': {'num_combinations': 3000, 'samples_per_combo': 4},  
        '3tab': {'num_combinations': 10000, 'samples_per_combo': 2}, 
        '4tab': {'num_combinations': 15000, 'samples_per_combo': 2}, 
        '5tab': {'num_combinations': 20000, 'samples_per_combo': 2}, 
        # Total: around 100k.
        '2tab_test': {'num_combinations': 1500, 'samples_per_combo': 4},
        '3tab_test': {'num_combinations': 5000, 'samples_per_combo': 2},
        '4tab_test': {'num_combinations': 7500, 'samples_per_combo': 2},
        '5tab_test': {'num_combinations': 10000, 'samples_per_combo': 2}, 
    },
    'medium': {
        '2tab': {'num_combinations': 3000, 'samples_per_combo': 6},
        '3tab': {'num_combinations': 10000, 'samples_per_combo': 3},
        '4tab': {'num_combinations': 15000, 'samples_per_combo': 3},
        '5tab': {'num_combinations': 20000, 'samples_per_combo': 3},
        '2tab_test': {'num_combinations': 500, 'samples_per_combo': 10},
        '3tab_test': {'num_combinations': 2500, 'samples_per_combo': 3},
        '4tab_test': {'num_combinations': 2500, 'samples_per_combo': 3},
        '5tab_test': {'num_combinations': 5000, 'samples_per_combo': 3},
    },
    'large': {
        '2tab': {'num_combinations': 1000, 'samples_per_combo': 20},
        '3tab': {'num_combinations': 10000, 'samples_per_combo': 5},
        '4tab': {'num_combinations': 15000, 'samples_per_combo': 5}, 
        '5tab': {'num_combinations': 20000, 'samples_per_combo': 5}, 
        # Total: close to 200k.
        '2tab_test': {'num_combinations': 500, 'samples_per_combo': 10}, 
        '3tab_test': {'num_combinations': 2500, 'samples_per_combo': 3}, 
        '4tab_test': {'num_combinations': 2500, 'samples_per_combo': 3}, 
        '5tab_test': {'num_combinations': 5000, 'samples_per_combo': 3}, 

    }
}

def generate_mixed_tab_dataset(
    generator,
    scale='small',
    output_root='datasets/multi_tab_datasets',
    check_interval=20,
    balance_attempts=20,
    add_ow_class=False
):
    """
    Generate a mixed-tab dataset with 2-5 tabs combined in one dataset.
    
    Args:
        generator: MultiTabDatasetGenerator instance.
        scale: 'small', 'medium', 'large'
        output_root: Output root directory.
        check_interval: Balance check interval.
        balance_attempts: Number of compensation attempts for each imbalance.
        add_ow_class: Whether to add OW class 95 at a random position in each combination.
    """
    config = MIXED_TAB_CONFIG[scale]
    
    print("\n" + "="*60)
    print(f"Mixed-Tab - {scale.upper()}")
    print("="*60)
    
    # Calculate total sample count.
    total_samples_train = 0
    total_samples_test = 0
    for tab_key, tab_config in config.items():
        samples = tab_config['num_combinations'] * tab_config['samples_per_combo']
        total_samples_train += samples
        total_samples_test += samples
        print(f"{tab_key}: {tab_config['num_combinations']} × {tab_config['samples_per_combo']} = {samples}")
    
    print(f"\n:")
    print(f"   - {total_samples_train}")
    print(f"   - {total_samples_test}")
    print(f"   - {total_samples_train + total_samples_test}")
    
    results = {}
    
    # Generate each split separately.
    for split in ['train', 'test']:
        print(f"\n{'='*60}")
        print(f"Mixed-Tab {split}")
        print(f"{'='*60}")
        
        # Create the mixed_tab output directory.
        mixed_output_dir = os.path.join(output_root, 'mixed_tab', split)
        query_dir = os.path.join(mixed_output_dir, "query_data")
        support_dir = os.path.join(mixed_output_dir, "support_data")
        
        os.makedirs(query_dir, exist_ok=True)
        os.makedirs(support_dir, exist_ok=True)
        
        all_query_filenames = []
        mixed_statistics = {
            'dataset_name': 'mixed_tab',
            'split': split,
            'scale': scale,
            'tab_distribution': {},
            'total_samples': 0,
            'total_combinations': 0
        }
        
        # Generate data for each tab count.
        for num_tabs in [3, 4, 5]:
            tab_key = f'{num_tabs}tab'
            if split == 'test':
                tab_key = f'{num_tabs}tab_test'
            tab_config = config[tab_key]
            
            print(f"\n{num_tabs}-tab:")
            print(f"   - {tab_config['num_combinations']}")
            print(f"   - {tab_config['samples_per_combo']}")
            
            # Temporarily generate into each tab-specific directory.
            temp_output = generator.generate_dataset(
                num_tabs=num_tabs,
                num_combinations=tab_config['num_combinations'],
                samples_per_combo=tab_config['samples_per_combo'],
                split=split,
                dataset_name=f'mixed_tab_{num_tabs}tab_temp',
                check_interval=check_interval,
                balance_attempts=balance_attempts,
                add_ow_class=add_ow_class
            )
            
            # Move generated files to the mixed_tab directory.
            temp_query_dir = os.path.join(temp_output, "query_data")
            temp_support_dir = os.path.join(temp_output, "support_data")
            
            # Move query files and record them.
            import shutil
            import json
            
            tab_query_filenames = []
            for filename in os.listdir(temp_query_dir):
                if filename.endswith('.pkl'):
                    # Add the tab marker to the filename.
                    new_filename = f"{filename}"
                    src = os.path.join(temp_query_dir, filename)
                    dst = os.path.join(query_dir, new_filename)
                    shutil.move(src, dst)
                    tab_query_filenames.append(new_filename)
                    all_query_filenames.append(new_filename)
            
            # Merge support files.
            for class_id in os.listdir(temp_support_dir):
                src_class_dir = os.path.join(temp_support_dir, class_id)
                dst_class_dir = os.path.join(support_dir, class_id)
                
                if os.path.isdir(src_class_dir):
                    os.makedirs(dst_class_dir, exist_ok=True)
                    
                    for filename in os.listdir(src_class_dir):
                        src_file = os.path.join(src_class_dir, filename)
                        dst_file = os.path.join(dst_class_dir, filename)
                        
                        # Copy only if the file does not already exist.
                        if not os.path.exists(dst_file):
                            shutil.copy2(src_file, dst_file)
            
            # Update statistics.
            mixed_statistics['tab_distribution'][tab_key] = {
                'num_samples': len(tab_query_filenames),
                'num_combinations': tab_config['num_combinations'],
                'samples_per_combo': tab_config['samples_per_combo']
            }
            mixed_statistics['total_samples'] += len(tab_query_filenames)
            mixed_statistics['total_combinations'] += tab_config['num_combinations']
            
            # Clean up the temporary directory.
            temp_parent = os.path.dirname(temp_output)
            if os.path.exists(temp_parent):
                shutil.rmtree(temp_parent)
            
            print(f"  [OK] {num_tabs}-tab: {len(tab_query_filenames)}mixed_tab")
        
        # Save the mixed_tab query filename list.
        query_json_path = os.path.join(mixed_output_dir, f"mixed_tab_{split}.json")
        with open(query_json_path, 'w') as f:
            json.dump(all_query_filenames, f, indent=2)
        
        # Save statistics.
        stats_json_path = os.path.join(mixed_output_dir, f"statistics_{split}.json")
        with open(stats_json_path, 'w') as f:
            json.dump(mixed_statistics, f, indent=2)
        
        print(f"\n[OK] Mixed-Tab {split}")
        print(f"   - {mixed_statistics['total_samples']}")
        print(f"  - Query: {query_json_path}")
        print(f"   - {stats_json_path}")
        
        results[split] = mixed_output_dir
    
    return results


def generate_datasets(
    num_tabs_list=[2, 3, 4, 5],
    scale='small',
    custom_config=None,
    output_root='datasets/multi_tab_datasets',
    source_root ='../datasets/MMFOW',
    random_seed=42,
    mixed_tabs=False,
    check_interval=20,
    balance_attempts=20,
    add_ow_class=False
):
    """
    Generate multi-label datasets.
    
    Args:
        num_tabs_list: List of tab counts to generate, e.g. [2, 3] for 2-tab and 3-tab.
        scale: 'small', 'medium', 'large'
        custom_config: Custom config, e.g. {'2tab': {'num_combinations': 10, 'samples_per_combo': 5}}.
        output_root: Output root directory.
        random_seed: Random seed.
        mixed_tabs: Whether to generate a mixed-tab dataset.
        check_interval: Balance check interval.
        balance_attempts: Number of compensation attempts for each imbalance.
        add_ow_class: Whether to add OW class 95 at a random position in each combination.
    """
    # Create generator.
    generator = MultiTabDatasetGenerator(
        source_root=source_root,
        output_root=output_root,
        num_classes=60,
        overlap_range=(0.0, 0.4),
        random_seed=random_seed
    )
    
    # Mixed-tab mode.
    if mixed_tabs:
        results = generate_mixed_tab_dataset(
            generator=generator,
            scale=scale,
            output_root=output_root,
            check_interval=check_interval,
            balance_attempts=balance_attempts,
            add_ow_class=add_ow_class
        )
        
        print("\n" + "="*60)
        print("[OK] Mixed-Tab!")
        print("="*60)
        print(f"\n:")
        print(f"  - mixed_tab:")
        print(f"      train: {results['train']}")
        print(f"      test: {results['test']}")
        
        return results
    
    # Original fixed-tab generation logic.
    # Select configuration.
    if custom_config:
        config = custom_config
    elif scale == 'small':
        config = SMALL_SCALE_CONFIG
    elif scale == 'medium':
        config = MEDIUM_SCALE_CONFIG
    elif scale == 'large':
        config = LARGE_SCALE_CONFIG
    else:
        raise ValueError(f"Unknown scale: {scale}")
    
    print("\n" + "="*60)
    print(f" - {scale.upper()}")
    print("="*60)
    print(f"tab: {num_tabs_list}")
    print(f": {output_root}")
    print(f": {random_seed}")
    
    # Generate datasets for each tab count.
    results = {}
    
    for num_tabs in num_tabs_list:
        dataset_key = f'{num_tabs}tab'
        test_dataset_key = f'{num_tabs}tab_test'
        
        if dataset_key not in config:
            print(f"\n[SKIP] {dataset_key}")
            continue
        
        num_combinations = config[dataset_key]['num_combinations']
        samples_per_combo = config[dataset_key]['samples_per_combo']
        test_num_combinations = config[test_dataset_key]['num_combinations']
        test_samples_per_combo = config[test_dataset_key]['samples_per_combo']
        print(f"\n{'='*60}")
        print(f"{num_tabs}-tab")
        print(f"{'='*60}")
        print(f": {num_combinations}")
        print(f": {samples_per_combo}")
        print(f": {num_combinations * samples_per_combo}")
        
        # Generate training set.
        print(f"\n[1/2] ...")
        train_output = generator.generate_dataset(
            num_tabs=num_tabs,
            num_combinations=num_combinations,
            samples_per_combo=samples_per_combo,
            split='train',
            dataset_name=f'{num_tabs}tab',
            check_interval=check_interval,
            balance_attempts=balance_attempts,
            add_ow_class=add_ow_class
        )
        
        # Generate test set.
        print(f"\n[2/2] ...")
        test_output = generator.generate_dataset(
            num_tabs=num_tabs,
            num_combinations=test_num_combinations,
            samples_per_combo=test_samples_per_combo,
            split='test',
            dataset_name=f'{num_tabs}tab',
            check_interval=check_interval,
            balance_attempts=balance_attempts,
            add_ow_class=add_ow_class
        )
        
        results[dataset_key] = {
            'train': train_output,
            'test': test_output
        }
    
    # Summary.
    print("\n" + "="*60)
    print("[OK] !")
    print("="*60)
    print(f"\n:")
    for key, paths in results.items():
        print(f"  - {key}:")
        print(f"      train: {paths['train']}")
        print(f"      test: {paths['test']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate multi-label Website Fingerprinting datasets')
    
    parser.add_argument(
        '--tabs',
        nargs='+',
        type=int,
        default=[2],
        choices=[2, 3, 4, 5],
        help='Tab counts to generate, e.g. --tabs 2 3 4. Ignored in mixed_tabs mode.'
    )
    
    parser.add_argument(
        '--scale',
        type=str,
        default='medium',
        choices=['small', 'medium', 'large'],
        help='Data scale: small for testing, medium, or large for paper-scale runs'
    )

    parser.add_argument(
        '--input',
        type=str,
        default='../datasets/MMFOW',
        help='Input root directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../datasets/complex_multi_tab_datasets',
        help='Output root directory'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    # Custom configuration arguments.
    parser.add_argument(
        '--num-combinations',
        type=int,
        help='Custom number of combinations. Overrides scale config; not used for mixed_tabs.'
    )
    
    parser.add_argument(
        '--samples-per-combo',
        type=int,
        help='Custom number of samples per combination. Overrides scale config; not used for mixed_tabs.'
    )
    
    parser.add_argument(
        '--mixed_tabs',
        action='store_true',
        help='Generate a mixed-tab dataset with 2-5 tabs combined'
    )
    
    # Balance arguments.
    parser.add_argument(
        '--check-interval',
        type=int,
        default=20,
        help='Balance check interval; default is 20'
    )
    
    parser.add_argument(
        '--balance-attempts',
        type=int,
        default=20,
        help='Number of compensation attempts for each imbalance; default is 20'
    )
    
    parser.add_argument(
        '--ow',
        action='store_true',
        help='Add Open World class 95 at a random position in each combination'
    )
    
    args = parser.parse_args()
    
    # Mixed-tab mode.
    if args.mixed_tabs:
        if args.num_combinations or args.samples_per_combo:
            print("[WARNING] mixed_tabs--num-combinations--samples-per-combo")
        
        generate_datasets(
            num_tabs_list=[2, 3, 4, 5],  # mixed_tabs always generates all tab counts.
            scale=args.scale,
            output_root=args.output,
            source_root = args.input,
            random_seed=args.seed,
            mixed_tabs=True,
            check_interval=args.check_interval,
            balance_attempts=args.balance_attempts,
            add_ow_class=args.ow
        )
        return
    
    # Fixed-tab mode.
    # Build custom config.
    custom_config = None
    if args.num_combinations or args.samples_per_combo:
        custom_config = {}
        for tab in args.tabs:
            custom_config[f'{tab}tab'] = {
                'num_combinations': args.num_combinations or SMALL_SCALE_CONFIG[f'{tab}tab']['num_combinations'],
                'samples_per_combo': args.samples_per_combo or SMALL_SCALE_CONFIG[f'{tab}tab']['samples_per_combo']
            }
    
    # Generate datasets.
    generate_datasets(
        num_tabs_list=args.tabs,
        scale=args.scale,
        custom_config=custom_config,
        output_root=args.output,
        source_root = args.input,
        random_seed=args.seed,
        mixed_tabs=False,
        check_interval=args.check_interval,
        balance_attempts=args.balance_attempts,
        add_ow_class=args.ow
    )


if __name__ == "__main__":
    # Show usage if no command-line arguments are provided.
    if len(sys.argv) == 1:
        print("\n" + "="*60)
        print("")
        print("="*60)
        print("\n:")
        print("\nFixed-Tab")
        print("1. 2-tab10×5")
        print("   python generate_multitab_datasets.py --tabs 2 --scale small")
        print("\n2. 2-tab3-tab")
        print("   python generate_multitab_datasets.py --tabs 2 3 --scale medium")
        print("\n3. tab")
        print("   python generate_multitab_datasets.py --tabs 2 3 4 5 --scale large")
        print("\n4.")
        print("   python generate_multitab_datasets.py --tabs 2 --num-combinations 20 --samples-per-combo 10")
        print("\nMixed-Tab")
        print("5. mixed-tab")
        print("   python generate_multitab_datasets.py --mixed_tabs --scale small")
        print("\n6. mixed-tab20")
        print("   python generate_multitab_datasets.py --mixed_tabs --scale large")
        print("\n7. Mixed-tab with")
        print("   python generate_multitab_datasets.py --mixed_tabs --scale medium --check-interval 30 --balance-attempts 25")
        print("\nOpen World")
        print("8. OW")
        print("   python generate_multitab_datasets.py --tabs 2 3 --scale small --ow")
        print("\n9. Mixed-tab + OW")
        print("   python generate_multitab_datasets.py --mixed_tabs --scale medium --ow")
        print("\n10.")
        print("   python generate_multitab_datasets.py --help")
        print("\n" + "="*60)
        print("\n...")
        print(": 2-tab, 10×5\n")
        
        # Run a small-scale test by default.
        # generate_datasets(
        #     num_tabs_list=[2],
        #     scale='small',
        #     output_root='datasets/multi_tab_datasets',
        #     random_seed=42
        # )
    else:
        main()