"""
完整的多标签数据集生成脚本
支持生成2-5 tab的fixed-tab数据集
支持生成mixed-tab数据集（2-5 tab混合）
参数化配置，可进行小规模测试或大规模生成
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

# 中等规模配置
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

# 大规模配置：论文实验规模
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

# Mixed-Tab配置：基于大规模配置，每个tab的组合数除以4
# 总样本数控制在20万左右
MIXED_TAB_CONFIG = {
    'small': {
        '2tab': {'num_combinations': 3000, 'samples_per_combo': 4},  
        '3tab': {'num_combinations': 10000, 'samples_per_combo': 2}, 
        '4tab': {'num_combinations': 15000, 'samples_per_combo': 2}, 
        '5tab': {'num_combinations': 20000, 'samples_per_combo': 2}, 
        # Total: (10万左右)
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
        # Total: (接近20万)
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
    生成mixed-tab数据集（2-5 tab混合在一个数据集中）
    
    Args:
        generator: MultiTabDatasetGenerator实例
        scale: 'small', 'medium', 'large'
        output_root: 输出根目录
        check_interval: 均衡性检查间隔
        balance_attempts: 每次不均衡时的补偿尝试次数
        add_ow_class: 是否添加OW类别95到每个组合的随机位置
    """
    config = MIXED_TAB_CONFIG[scale]
    
    print("\n" + "="*60)
    print(f"生成Mixed-Tab数据集 - {scale.upper()}规模")
    print("="*60)
    
    # 计算总样本数
    total_samples_train = 0
    total_samples_test = 0
    for tab_key, tab_config in config.items():
        samples = tab_config['num_combinations'] * tab_config['samples_per_combo']
        total_samples_train += samples
        total_samples_test += samples
        print(f"{tab_key}: {tab_config['num_combinations']}组 × {tab_config['samples_per_combo']}样本 = {samples}样本")
    
    print(f"\n总样本数:")
    print(f"  - 训练集: {total_samples_train}")
    print(f"  - 测试集: {total_samples_test}")
    print(f"  - 合计: {total_samples_train + total_samples_test}")
    
    results = {}
    
    # 对每个split分别生成
    for split in ['train', 'test']:
        print(f"\n{'='*60}")
        print(f"生成Mixed-Tab {split}集")
        print(f"{'='*60}")
        
        # 创建mixed_tab输出目录
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
        
        # 生成每个tab数的数据
        for num_tabs in [2, 3, 4, 5]:
            tab_key = f'{num_tabs}tab'
            if split == 'test':
                tab_key = f'{num_tabs}tab_test'
            tab_config = config[tab_key]
            
            print(f"\n生成{num_tabs}-tab部分:")
            print(f"  - 组合数: {tab_config['num_combinations']}")
            print(f"  - 每组样本数: {tab_config['samples_per_combo']}")
            
            # 临时生成到各自的tab目录
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
            
            # 移动生成的文件到mixed_tab目录
            temp_query_dir = os.path.join(temp_output, "query_data")
            temp_support_dir = os.path.join(temp_output, "support_data")
            
            # 移动query文件并记录
            import shutil
            import json
            
            tab_query_filenames = []
            for filename in os.listdir(temp_query_dir):
                if filename.endswith('.pkl'):
                    # 添加tab标识到文件名
                    new_filename = f"{filename}"
                    src = os.path.join(temp_query_dir, filename)
                    dst = os.path.join(query_dir, new_filename)
                    shutil.move(src, dst)
                    tab_query_filenames.append(new_filename)
                    all_query_filenames.append(new_filename)
            
            # 合并support文件
            for class_id in os.listdir(temp_support_dir):
                src_class_dir = os.path.join(temp_support_dir, class_id)
                dst_class_dir = os.path.join(support_dir, class_id)
                
                if os.path.isdir(src_class_dir):
                    os.makedirs(dst_class_dir, exist_ok=True)
                    
                    for filename in os.listdir(src_class_dir):
                        src_file = os.path.join(src_class_dir, filename)
                        dst_file = os.path.join(dst_class_dir, filename)
                        
                        # 如果文件不存在则复制
                        if not os.path.exists(dst_file):
                            shutil.copy2(src_file, dst_file)
            
            # 更新统计信息
            mixed_statistics['tab_distribution'][tab_key] = {
                'num_samples': len(tab_query_filenames),
                'num_combinations': tab_config['num_combinations'],
                'samples_per_combo': tab_config['samples_per_combo']
            }
            mixed_statistics['total_samples'] += len(tab_query_filenames)
            mixed_statistics['total_combinations'] += tab_config['num_combinations']
            
            # 清理临时目录
            temp_parent = os.path.dirname(temp_output)
            if os.path.exists(temp_parent):
                shutil.rmtree(temp_parent)
            
            print(f"  [OK] {num_tabs}-tab: {len(tab_query_filenames)}个样本已添加到mixed_tab")
        
        # 保存mixed_tab的query文件名列表
        query_json_path = os.path.join(mixed_output_dir, f"mixed_tab_{split}.json")
        with open(query_json_path, 'w') as f:
            json.dump(all_query_filenames, f, indent=2)
        
        # 保存统计信息
        stats_json_path = os.path.join(mixed_output_dir, f"statistics_{split}.json")
        with open(stats_json_path, 'w') as f:
            json.dump(mixed_statistics, f, indent=2)
        
        print(f"\n[OK] Mixed-Tab {split}集生成完成")
        print(f"  - 总样本数: {mixed_statistics['total_samples']}")
        print(f"  - Query索引: {query_json_path}")
        print(f"  - 统计信息: {stats_json_path}")
        
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
    生成多标签数据集
    
    Args:
        num_tabs_list: 要生成的tab数列表，如[2, 3]表示生成2-tab和3-tab
        scale: 'small', 'medium', 'large'
        custom_config: 自定义配置，格式如{'2tab': {'num_combinations': 10, 'samples_per_combo': 5}}
        output_root: 输出根目录
        random_seed: 随机种子
        mixed_tabs: 是否生成mixed-tab数据集
        check_interval: 均衡性检查间隔
        balance_attempts: 每次不均衡时的补偿尝试次数
        add_ow_class: 是否添加OW类别95到每个组合的随机位置
    """
    # 创建生成器
    generator = MultiTabDatasetGenerator(
        source_root=source_root,
        output_root=output_root,
        num_classes=60,
        overlap_range=(0.0, 0.4),
        random_seed=random_seed
    )
    
    # 如果是mixed-tab模式
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
        print("[OK] Mixed-Tab数据集生成完成!")
        print("="*60)
        print(f"\n生成的数据集:")
        print(f"  - mixed_tab:")
        print(f"      train: {results['train']}")
        print(f"      test: {results['test']}")
        
        return results
    
    # 原有的fixed-tab生成逻辑
    # 选择配置
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
    print(f"多标签数据集生成 - {scale.upper()}规模")
    print("="*60)
    print(f"生成tab数: {num_tabs_list}")
    print(f"输出目录: {output_root}")
    print(f"随机种子: {random_seed}")
    
    # 生成各个tab数的数据集
    results = {}
    
    for num_tabs in num_tabs_list:
        dataset_key = f'{num_tabs}tab'
        test_dataset_key = f'{num_tabs}tab_test'
        
        if dataset_key not in config:
            print(f"\n[SKIP] 配置中未找到{dataset_key}，跳过")
            continue
        
        num_combinations = config[dataset_key]['num_combinations']
        samples_per_combo = config[dataset_key]['samples_per_combo']
        test_num_combinations = config[test_dataset_key]['num_combinations']
        test_samples_per_combo = config[test_dataset_key]['samples_per_combo']
        print(f"\n{'='*60}")
        print(f"生成{num_tabs}-tab数据集")
        print(f"{'='*60}")
        print(f"组合数: {num_combinations}")
        print(f"每组样本数: {samples_per_combo}")
        print(f"总样本数: {num_combinations * samples_per_combo}")
        
        # 生成训练集
        print(f"\n[1/2] 生成训练集...")
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
        
        # 生成测试集
        print(f"\n[2/2] 生成测试集...")
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
    
    # 总结
    print("\n" + "="*60)
    print("[OK] 所有数据集生成完成!")
    print("="*60)
    print(f"\n生成的数据集:")
    for key, paths in results.items():
        print(f"  - {key}:")
        print(f"      train: {paths['train']}")
        print(f"      test: {paths['test']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='生成多标签Website Fingerprinting数据集')
    
    parser.add_argument(
        '--tabs',
        nargs='+',
        type=int,
        default=[2],
        choices=[2, 3, 4, 5],
        help='要生成的tab数，如：--tabs 2 3 4（mixed_tabs模式下忽略此参数）'
    )
    
    parser.add_argument(
        '--scale',
        type=str,
        default='small',
        choices=['small', 'medium', 'large'],
        help='数据规模：small(测试), medium(中等), large(论文)'
    )

    parser.add_argument(
        '--input',
        type=str,
        default='../datasets/MMFOW',
        help='输入根目录'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../datasets/complex_multi_tab_datasets',
        help='输出根目录'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    
    # 自定义配置参数
    parser.add_argument(
        '--num-combinations',
        type=int,
        help='自定义组合数（覆盖scale配置，不适用于mixed_tabs）'
    )
    
    parser.add_argument(
        '--samples-per-combo',
        type=int,
        help='自定义每组样本数（覆盖scale配置，不适用于mixed_tabs）'
    )
    
    parser.add_argument(
        '--mixed_tabs',
        action='store_true',
        help='生成mixed-tab数据集（2-5 tab混合）'
    )
    
    # 均衡性参数
    parser.add_argument(
        '--check-interval',
        type=int,
        default=20,
        help='均衡性检查间隔（默认20）'
    )
    
    parser.add_argument(
        '--balance-attempts',
        type=int,
        default=20,
        help='每次不均衡时的补偿尝试次数（默认20）'
    )
    
    parser.add_argument(
        '--ow',
        action='store_true',
        help='添加Open World类别95到每个组合的随机位置'
    )
    
    args = parser.parse_args()
    
    # Mixed-tab模式
    if args.mixed_tabs:
        if args.num_combinations or args.samples_per_combo:
            print("[WARNING] mixed_tabs模式下忽略--num-combinations和--samples-per-combo参数")
        
        generate_datasets(
            num_tabs_list=[2, 3, 4, 5],  # mixed_tabs总是生成所有tab
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
    
    # Fixed-tab模式
    # 构建自定义配置
    custom_config = None
    if args.num_combinations or args.samples_per_combo:
        custom_config = {}
        for tab in args.tabs:
            custom_config[f'{tab}tab'] = {
                'num_combinations': args.num_combinations or SMALL_SCALE_CONFIG[f'{tab}tab']['num_combinations'],
                'samples_per_combo': args.samples_per_combo or SMALL_SCALE_CONFIG[f'{tab}tab']['samples_per_combo']
            }
    
    # 生成数据集
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
    # 如果没有命令行参数，显示使用说明
    if len(sys.argv) == 1:
        print("\n" + "="*60)
        print("多标签数据集生成脚本")
        print("="*60)
        print("\n使用示例:")
        print("\n【Fixed-Tab模式】")
        print("1. 小规模测试（生成2-tab，10组×5样本）：")
        print("   python generate_multitab_datasets.py --tabs 2 --scale small")
        print("\n2. 生成2-tab和3-tab（中等规模）：")
        print("   python generate_multitab_datasets.py --tabs 2 3 --scale medium")
        print("\n3. 生成所有tab数（大规模）：")
        print("   python generate_multitab_datasets.py --tabs 2 3 4 5 --scale large")
        print("\n4. 自定义参数：")
        print("   python generate_multitab_datasets.py --tabs 2 --num-combinations 20 --samples-per-combo 10")
        print("\n【Mixed-Tab模式】")
        print("5. 生成mixed-tab数据集（小规模测试）：")
        print("   python generate_multitab_datasets.py --mixed_tabs --scale small")
        print("\n6. 生成mixed-tab数据集（大规模，约20万样本）：")
        print("   python generate_multitab_datasets.py --mixed_tabs --scale large")
        print("\n7. Mixed-tab with均衡参数：")
        print("   python generate_multitab_datasets.py --mixed_tabs --scale medium --check-interval 30 --balance-attempts 25")
        print("\n【Open World模式】")
        print("8. 生成带OW类别的数据集：")
        print("   python generate_multitab_datasets.py --tabs 2 3 --scale small --ow")
        print("\n9. Mixed-tab + OW：")
        print("   python generate_multitab_datasets.py --mixed_tabs --scale medium --ow")
        print("\n10. 查看所有选项：")
        print("   python generate_multitab_datasets.py --help")
        print("\n" + "="*60)
        print("\n当前将使用默认参数运行小规模测试...")
        print("生成: 2-tab, 10组×5样本\n")
        
        # 默认运行小规模测试
        # generate_datasets(
        #     num_tabs_list=[2],
        #     scale='small',
        #     output_root='datasets/multi_tab_datasets',
        #     random_seed=42
        # )
    else:
        main()