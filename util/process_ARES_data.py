"""
文件: process_ARES_data.py
功能: 处理ARES OW数据集，并将其转换为meta-finger项目需要的格式
使用: python process_ARES_data.py [--input INPUT_FILE] [--output OUTPUT_DIR] [--remap] [--skip-unmonitored]
"""

import os
import numpy as np
import shutil
import pickle
import random
import argparse
from collections import Counter
from tqdm import tqdm
import json
import itertools
import gc

def create_directory_structure(base_dir):
    """
    创建目录结构，并清空已存在的目录
    
    参数:
        base_dir: 基础目录路径
    """
    # 创建主目录
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"创建主目录: {base_dir}")
    
    # 创建train和test目录及其子目录
    train_dir = os.path.join(base_dir, "train/single_tab")
    test_dir = os.path.join(base_dir, "test/single_tab")
    
    # 清空或创建train目录
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
        print(f"清空目录: {train_dir}")
    os.makedirs(train_dir)
    print(f"创建目录: {train_dir}")
    
    # 清空或创建test目录
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print(f"清空目录: {test_dir}")
    os.makedirs(test_dir)
    print(f"创建目录: {test_dir}")
    
    return train_dir, test_dir

def identify_unmonitored_class(y, threshold=30.0):
    """
    识别未监控类别（通常是样本数量最多的类别）
    
    参数:
        y: 类别标签数组
        threshold: 判断阈值，表示占总样本的百分比
        
    返回:
        未监控类别的标签
    """
    # 计算每个类别的样本数量
    class_counts = Counter(y)
    # 找出样本数量最多的类别
    most_common_class, count = class_counts.most_common(1)[0]
    
    total_samples = len(y)
    percentage = (count / total_samples) * 100
    
    print(f"检测到可能的未监控类别: {most_common_class}, 样本数量: {count}, 占总样本比例: {percentage:.2f}%")
    
    # 如果这个类别的样本占比显著高于其他类别，则认为它是未监控类别
    if percentage > threshold:  # 使用可配置的阈值
        print(f"确认类别 {most_common_class} 为未监控类别")
        return most_common_class
    else:
        print("无法确定未监控类别，将使用原始标签")
        return None

def split_data(X, y, train_ratio=0.8, seed=42):
    """
    将数据按照指定比例划分为训练集和测试集
    
    参数:
        X: 特征数据
        y: 标签数据
        train_ratio: 训练集比例
        seed: 随机种子
        
    返回:
        训练集特征, 训练集标签, 测试集特征, 测试集标签
    """
    # 设置随机种子以确保结果可重现
    np.random.seed(seed)
    
    # 获取数据集大小
    n_samples = len(y)
    
    # 创建索引数组并打乱
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # 计算划分点
    split_point = int(n_samples * train_ratio)
    
    # 划分训练集和测试集
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    print(f"数据集划分完成:")
    print(f"  - 训练集: {len(X_train)} 样本")
    print(f"  - 测试集: {len(X_test)} 样本")
    
    return X_train, y_train, X_test, y_test

def save_samples_by_class(X, y, output_dir, skip_unmonitored=False, unmonitored_class=None, batch_size=1000):
    """
    将样本按照类别保存到相应的目录中，使用批处理以减少内存使用
    
    参数:
        X: 特征数据
        y: 标签数据
        output_dir: 输出目录
        skip_unmonitored: 是否跳过未监控类别
        unmonitored_class: 未监控类别的标签
        batch_size: 批处理大小
    """
    # 获取所有类别
    unique_classes = np.unique(y)
    
    # 统计每个类别的样本数量
    class_counts = Counter(y)
    total_saved = 0
    skipped = 0
    
    print(f"开始保存样本到 {output_dir}:")
    
    # 按类别处理样本
    for class_label in unique_classes:
        # 如果指定了跳过未监控类别，且当前类别是未监控类别，则跳过
        if skip_unmonitored and class_label == unmonitored_class:
            skipped += class_counts[class_label]
            print(f"  - 跳过未监控类别 {class_label}, {class_counts[class_label]} 个样本")
            continue
        
        # 创建类别目录
        class_dir = os.path.join(output_dir, str(class_label))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        # 获取当前类别的样本索引
        class_indices = np.where(y == class_label)[0]
        
        # 分批处理以减少内存使用
        num_batches = (len(class_indices) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(class_indices))
            batch_indices = class_indices[start_idx:end_idx]
            
            batch_str = f"批次 {batch_idx+1}/{num_batches}"
            
            # 保存该批次的样本
            for i, idx in enumerate(tqdm(batch_indices, 
                                         desc=f"保存类别 {class_label} 的样本 ({batch_str})")):
                sample_idx = start_idx + i
                
                # 为每个样本创建伪时间序列（与特征长度相同）
                # 为了节省内存，只为当前处理的样本创建时间序列
                feature_length = len(X[idx])
                fake_time = np.arange(feature_length) * 0.001  # 间隔0.001秒
                
                # 创建样本字典
                sample_dict = {
                    'time': fake_time,     # 伪时间序列
                    'data': X[idx],        # 特征数据
                    'label': class_label   # 标签
                }
                
                # 保存样本
                sample_path = os.path.join(class_dir, f"sample_{sample_idx}.pkl")
                with open(sample_path, 'wb') as f:
                    pickle.dump(sample_dict, f)
            
            # 显式释放内存
            del fake_time
            import gc
            gc.collect()
        
        total_saved += len(class_indices)
    
    print(f"保存完成，共保存 {total_saved} 个样本，跳过 {skipped} 个样本")

def remap_unmonitored_class(y, unmonitored_class, target_label=None):
    """
    将未监控类别重映射到指定标签
    
    参数:
        y: 类别标签数组
        unmonitored_class: 未监控类别的原始标签
        target_label: 目标标签（如果为None，则使用最大类别索引+1）
        
    返回:
        重映射后的标签数组
    """
    if unmonitored_class is None:
        print("未指定未监控类别，不进行重映射")
        return y
    
    # 创建标签的副本
    y_remapped = y.copy()
    
    # 如果目标标签未指定，则使用最大类别索引+1
    if target_label is None:
        max_class_idx = np.max([c for c in np.unique(y) if c != unmonitored_class])
        target_label = max_class_idx + 1
        print(f"未指定目标标签，将使用最大类别索引+1: {target_label}")
    
    # 重映射未监控类别
    mask = (y == unmonitored_class)
    y_remapped[mask] = target_label
    
    # 统计修改数量
    num_changed = np.sum(mask)
    print(f"将 {num_changed} 个样本从类别 {unmonitored_class} 重映射为 {target_label}")
    
    return y_remapped

def process_ARES_data(input_file, output_base_dir, remap_unmonitored=False, skip_unmonitored=False, 
                     train_ratio=0.8, seed=42, batch_size=500, unmonitored_threshold=30.0):
    """
    处理ARES数据集，并按照meta-finger项目的要求组织
    
    参数:
        input_file: 输入数据文件路径
        output_base_dir: 输出基础目录路径
        remap_unmonitored: 是否将未监控类别重映射为num_classes+1
        skip_unmonitored: 是否跳过未监控类别（不保存）
        train_ratio: 训练集比例
        seed: 随机种子
        batch_size: 保存样本时的批处理大小
        unmonitored_threshold: 判断未监控类别的阈值
    """
    print(f"开始处理ARES数据集: {input_file}")
    
    # 创建目录结构
    train_dir, test_dir = create_directory_structure(output_base_dir)
    
    # 加载数据
    print(f"加载数据文件: {input_file}")
    try:
        # 这里不是一次性加载全部数据，而是先仅加载元数据
        data = np.load(input_file)
        # 获取数据形状
        X_shape = data["X"].shape
        y_shape = data["y"].shape
        
        # 直接获取标签进行统计，因为标签较小
        y = data["y"]
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 数据基本信息
    n_samples, n_features = X_shape
    n_classes = len(np.unique(y))
    
    print(f"数据加载完成:")
    print(f"  - 样本数量: {n_samples}")
    print(f"  - 特征数量: {n_features}")
    print(f"  - 类别数量: {n_classes}")
    
    # 检查标签是否连续
    if n_classes != y.max() + 1:
        print("警告: 标签不连续，可能需要进行重映射")
    
    # 识别未监控类别
    unmonitored_class = identify_unmonitored_class(y, threshold=unmonitored_threshold)
    
    # 如果需要，重映射未监控类别
    remapped_label = None
    if remap_unmonitored and unmonitored_class is not None:
        # 将未监控类别映射到num_classes+1
        y = remap_unmonitored_class(y, unmonitored_class, target_label=None)
        # 获取实际映射的标签（用于后续处理）
        remapped_label = np.max(y)
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 生成训练集和测试集索引
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # 计算划分点
    split_point = int(n_samples * train_ratio)
    
    # 划分训练集和测试集索引
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    print(f"数据集划分完成:")
    print(f"  - 训练集: {len(train_indices)} 样本")
    print(f"  - 测试集: {len(test_indices)} 样本")
    
    # 按类别组织索引
    train_indices_by_class = {}
    test_indices_by_class = {}
    
    for idx in train_indices:
        label = y[idx]
        if label not in train_indices_by_class:
            train_indices_by_class[label] = []
        train_indices_by_class[label].append(idx)
    
    for idx in test_indices:
        label = y[idx]
        if label not in test_indices_by_class:
            test_indices_by_class[label] = []
        test_indices_by_class[label].append(idx)
    
    # 保存训练集
    print("\n保存训练集:")
    save_samples_by_indices(data, train_indices_by_class, train_dir, 
                          skip_unmonitored=skip_unmonitored, 
                          unmonitored_class=remapped_label if remap_unmonitored else unmonitored_class,
                          batch_size=batch_size)
    
    # 保存测试集
    print("\n保存测试集:")
    save_samples_by_indices(data, test_indices_by_class, test_dir, 
                          skip_unmonitored=skip_unmonitored, 
                          unmonitored_class=remapped_label if remap_unmonitored else unmonitored_class,
                          batch_size=batch_size)
    
    print("\n数据处理完成!")

def save_samples_by_indices(data, indices_by_class, output_dir, skip_unmonitored=False, unmonitored_class=None, batch_size=500):
    """
    按类别和索引保存样本，流式读取数据以节省内存
    
    参数:
        data: npz数据对象
        indices_by_class: 按类别组织的索引字典
        output_dir: 输出目录
        skip_unmonitored: 是否跳过未监控类别
        unmonitored_class: 未监控类别的标签
        batch_size: 批处理大小
    """
    total_saved = 0
    skipped = 0
    
    print(f"开始保存样本到 {output_dir}:")
    
    # 按类别处理样本
    for class_label, indices in indices_by_class.items():
        # 如果指定了跳过未监控类别，且当前类别是未监控类别，则跳过
        if skip_unmonitored and class_label == unmonitored_class:
            skipped += len(indices)
            print(f"  - 跳过未监控类别 {class_label}, {len(indices)} 个样本")
            continue
        
        # 创建类别目录
        class_dir = os.path.join(output_dir, str(class_label))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        # 分批处理索引以减少内存使用
        num_batches = (len(indices) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            batch_str = f"批次 {batch_idx+1}/{num_batches}"
            
            # 只加载当前批次的数据
            batch_data = data["X"][batch_indices]
            
            # 保存该批次的样本
            for i, (idx, features) in enumerate(zip(batch_indices, batch_data)):
                sample_idx = start_idx + i
                
                # 为每个样本创建伪时间序列（与特征长度相同）
                feature_length = len(features)
                fake_time = np.arange(feature_length) * 0.001  # 间隔0.001秒
                
                # 创建样本字典
                sample_dict = {
                    'time': fake_time,     # 伪时间序列
                    'data': features,      # 特征数据
                    'label': class_label   # 标签
                }
                
                # 保存样本
                sample_path = os.path.join(class_dir, f"sample_{sample_idx}.pkl")
                with open(sample_path, 'wb') as f:
                    pickle.dump(sample_dict, f)
                
                # 更新进度
                if (i + 1) % 100 == 0 or i + 1 == len(batch_indices):
                    print(f"\r保存类别 {class_label} 的样本 ({batch_str}): {i+1}/{len(batch_indices)}", end="")
            
            print()  # 换行
            
            # 显式释放内存
            del batch_data, fake_time
            import gc
            gc.collect()
        
        total_saved += len(indices)
    
    print(f"保存完成，共保存 {total_saved} 个样本，跳过 {skipped} 个样本")

def rm_zero(sequence):
    """
    移除序列末尾的零填充，确保不会返回空数组
    
    参数:
        sequence: 输入序列
        
    返回:
        处理后的序列，至少保留一个元素
    """
    if len(sequence) == 0:
        return sequence  # 如果序列为空，则直接返回
        
    # 找到最后一个非零元素的位置
    index = 1  # 默认至少保留一个元素
    for i in range(len(sequence) - 1, -1, -1):
        if sequence[i] != 0:
            index = i + 1
            break
    
    # 确保至少保留100个元素（或整个序列如果更短）
    index = max(index, min(100, len(sequence)))
    
    return sequence[:index]

def merge_with_durationscale(times, datas, ratio):
    """
    基于duration方式合成多标签数据
    
    参数:
        times: 时间序列数组
        datas: 数据序列数组
        ratio: 重叠比例
    
    返回:
        merged_time: 合并后的时间序列
        merged_data: 合并后的数据序列
        None: 如果合并失败
    """
    # 检查输入序列是否为空
    for i, (time, data) in enumerate(zip(times, datas)):
        if len(time) == 0 or len(data) == 0:
            return None
    
    time = times[0]
    data = datas[0]
    
    # 检查序列是否为空
    if len(time) == 0 or len(data) == 0:
        return None
    
    split_index = 0
    
    # 根据重叠比例找到分割点
    split_time = np.max(time) * (1 - ratio)
    for i, packet_time in enumerate(time):
        if packet_time >= split_time:
            split_index = i
            break
    
    # 如果分割点是0，则设置为序列长度的一半
    if split_index == 0:
        split_index = len(time) // 2
    
    # 添加第一个页面的前半部分到合并序列
    merged_time = list(time[:split_index])
    merged_data = list(data[:split_index])
    
    # 剩余部分用于与下一个页面重叠
    res_time = time[split_index:]
    res_data = data[split_index:]
    
    # 如果剩余部分为空，则认为无法合并
    if len(res_time) == 0 or len(res_data) == 0:
        return None
    
    # 处理剩余的页面
    for time, data in list(zip(times, datas))[1:]:
        # 检查序列是否为空
        if len(time) == 0 or len(data) == 0:
            return None
            
        # 调整时间序列，使其与前一个页面的剩余部分重叠
        base_time = res_time[0]  # 使用剩余部分的第一个时间点作为基准
        time = [(t + base_time) for t in time]
        
        index1 = index2 = 0
        
        # 合并两个序列，按时间顺序排序
        while index1 < len(res_time) and index2 < len(time):
            if res_time[index1] <= time[index2]:
                merged_time.append(res_time[index1])
                merged_data.append(res_data[index1])
                index1 += 1
            else:
                merged_time.append(time[index2])
                merged_data.append(data[index2])
                index2 += 1
        
        # 如果第一个序列没有完全遍历，继续添加剩余部分
        while index1 < len(res_time):
            merged_time.append(res_time[index1])
            merged_data.append(res_data[index1])
            index1 += 1
        
        # 如果第二个序列没有遍历完，继续添加剩余部分
        if index2 < len(time):
            remaining_time = time[index2:]
            remaining_data = data[index2:]
            
            # 根据重叠比例找到第二个页面的分割点
            if len(remaining_time) > 0:
                split_time = remaining_time[-1] * (1 - ratio)
                
                # 寻找分割索引
                split_index = len(remaining_time) - 1  # 默认为最后一个点
                for i, packet_time in enumerate(remaining_time):
                    if packet_time >= split_time:
                        split_index = i
                        break
                
                # 添加第二个页面的中间部分到合并序列
                merged_time.extend(remaining_time[:split_index])
                merged_data.extend(remaining_data[:split_index])
                
                # 更新剩余部分用于下一次合并
                res_time = remaining_time[split_index:]
                res_data = remaining_data[split_index:]
            else:
                # 如果没有剩余部分，认为无法继续合并
                return None
        else:
            # 如果第二个序列已经全部合并，无法继续进行下一轮合并
            return None
    
    # 添加最后一个页面的剩余部分
    merged_time.extend(res_time)
    merged_data.extend(res_data)
    
    # 确保合并后的序列不为空
    if len(merged_time) == 0 or len(merged_data) == 0:
        return None
    
    return merged_time, merged_data

def load_sample(sample_path):
    """
    加载样本数据
    
    参数:
        sample_path: 样本文件路径
    
    返回:
        sample_data: 样本数据
    """
    try:
        with open(sample_path, 'rb') as f:
            sample_data = pickle.load(f)
        return sample_data
    except Exception as e:
        print(f"加载样本 {sample_path} 失败: {e}")
        return None

def create_multi_tab_dataset(single_tab_dir, output_dir, num_tabs=3, include_unmonitored=False, 
                             overlap_ratio=0.1, samples_per_combination=30, max_sequence_length=30000, seed=42):
    """
    从单标签数据创建多标签数据集，使用类别组合方式
    
    参数:
        single_tab_dir: 单标签数据目录
        output_dir: 输出目录
        num_tabs: 每个样本的标签数量
        include_unmonitored: 是否包含未监控类别
        overlap_ratio: 重叠比例
        samples_per_combination: 每个类别组合生成的样本数量
        max_sequence_length: 最大序列长度
        seed: 随机种子
    """
    print(f"开始生成{num_tabs}tab数据集，{'包含' if include_unmonitored else '不包含'}未监控类别，重叠比例: {overlap_ratio}")
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载所有可用类别
    categories = []
    unmonitored_class = None
    
    for item in os.listdir(single_tab_dir):
        if os.path.isdir(os.path.join(single_tab_dir, item)) and item.isdigit():
            cat_id = int(item)
            # 确定未监控类别（假设是最大的类别ID）
            if unmonitored_class is None or cat_id > unmonitored_class:
                unmonitored_class = cat_id
            categories.append(cat_id)
    
    categories.sort()
    
    # 如果不包含未监控类别，则从类别列表中移除
    monitored_categories = categories.copy()
    if not include_unmonitored and unmonitored_class in categories:
        monitored_categories.remove(unmonitored_class)
        print(f"找到{len(monitored_categories)}个监控类别和1个未监控类别({unmonitored_class})")
    else:
        print(f"找到{len(categories)}个类别: {categories}")
    
    # 检查是否有足够的类别用于生成多标签数据
    if len(monitored_categories) < 2:
        print(f"错误: 可用类别数量不足，至少需要2个不同的类别")
        return
    
    # 创建样本索引
    category_samples = {}
    for category in categories:
        category_dir = os.path.join(single_tab_dir, str(category))
        if os.path.exists(category_dir):
            sample_files = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.pkl')]
            if sample_files:
                category_samples[category] = sample_files
                print(f"类别{category}：找到{len(sample_files)}个样本")
            else:
                print(f"警告: 类别{category}没有找到有效样本")
    
    # 检查是否所有类别都有样本
    for category in list(categories):
        if category not in category_samples or not category_samples[category]:
            print(f"警告: 移除没有样本的类别 {category}")
            if category in categories:
                categories.remove(category)
            if category in monitored_categories:
                monitored_categories.remove(category)
    
    # 再次检查是否有足够的类别
    if len(monitored_categories) < num_tabs:
        print(f"错误: 可用类别数量({len(monitored_categories)})小于所需标签数量({num_tabs})")
        adjusted_num_tabs = len(monitored_categories)
        if adjusted_num_tabs >= 2:
            print(f"调整标签数量为: {adjusted_num_tabs}")
            num_tabs = adjusted_num_tabs
        else:
            print("无法生成多标签数据")
            return
    
    # 生成所有可能的类别组合
    available_categories = monitored_categories
    if include_unmonitored and unmonitored_class in category_samples:
        available_categories = monitored_categories + [unmonitored_class]
    
    print(f"生成所有{num_tabs}个类别的组合...")
    # 使用迭代器而不是一次性生成所有组合以减少内存使用
    combinations_iterator = itertools.combinations(available_categories, num_tabs)
    
    # 计算组合总数
    from math import comb
    total_combinations = comb(len(available_categories), num_tabs)
    print(f"共有{total_combinations}种可能的类别组合")
    
    # 控制生成总样本数量，避免生成太多数据
    max_total_samples = 10000  # 最大总样本数量
    
    # 如果预计总样本数太大，则调整每个组合的样本数量
    estimated_total_samples = total_combinations * samples_per_combination
    if estimated_total_samples > max_total_samples:
        adjusted_samples_per_combination = max(1, max_total_samples // total_combinations)
        print(f"警告: 预计总样本数({estimated_total_samples})超过限制({max_total_samples})，调整每个组合的样本数量为{adjusted_samples_per_combination}")
        samples_per_combination = adjusted_samples_per_combination
    
    # 保存所有样本的列表
    merged_times = []
    merged_datas = []
    merged_labels = []
    
    # 限制类别组合数量，避免组合爆炸
    max_combinations = 1000
    combinations_to_use = []
    for i, combination in enumerate(combinations_iterator):
        if i >= max_combinations:
            break
        combinations_to_use.append(combination)
    
    # 如果组合数量太多，随机选择一部分
    if len(combinations_to_use) > max_combinations:
        random.shuffle(combinations_to_use)
        combinations_to_use = combinations_to_use[:max_combinations]
        print(f"组合数量太多，随机选择了{max_combinations}个组合")
    
    # 处理每个类别组合
    combinations_created = 0
    total_tries = 0
    failed_tries = 0
    
    # 创建进度条
    pbar = tqdm(total=len(combinations_to_use) * samples_per_combination, desc="生成多标签样本")
    
    for combination in combinations_to_use:
        # 检查所有类别是否都有样本
        valid_combination = True
        for category in combination:
            if category not in category_samples or not category_samples[category]:
                print(f"组合{combination}中的类别{category}没有样本，跳过")
                valid_combination = False
                break
        
        if not valid_combination:
            continue
        
        samples_for_combination = 0
        attempts_for_combination = 0
        max_attempts_per_combination = samples_per_combination * 3  # 每个组合的最大尝试次数
        
        while samples_for_combination < samples_per_combination and attempts_for_combination < max_attempts_per_combination:
            attempts_for_combination += 1
            total_tries += 1
            
            try:
                # 为每个类别选择随机样本
                sample_paths = []
                for category in combination:
                    sample_paths.append(random.choice(category_samples[category]))
                
                # 加载所有样本数据
                all_times = []
                all_datas = []
                valid_samples = True
                
                for path in sample_paths:
                    try:
                        with open(path, 'rb') as f:
                            sample = pickle.load(f)
                        
                        # 提取数据
                        time_data = sample['time']
                        feature_data = sample['data']
                        
                        # 移除末尾的零填充
                        nonzero_indices = np.where(feature_data != 0)[0]
                        if len(nonzero_indices) > 0:
                            effective_length = nonzero_indices[-1] + 1
                            # 确保至少保留100个元素（或整个序列如果更短）
                            effective_length = max(effective_length, min(100, len(feature_data)))
                        else:
                            effective_length = min(100, len(feature_data))
                        
                        # 截取有效部分
                        time_data = time_data[:effective_length]
                        feature_data = feature_data[:effective_length]
                        
                        all_times.append(time_data)
                        all_datas.append(feature_data)
                    except Exception as e:
                        print(f"加载样本时出错 {path}: {e}")
                        valid_samples = False
                        break
                
                if not valid_samples or len(all_times) == 0:
                    failed_tries += 1
                    continue
                
                # 直接拼接样本
                try:
                    # 初始化合并后的数据和时间
                    merged_time = all_times[0].copy()
                    merged_data = all_datas[0].copy()
                    last_time = merged_time[-1]
                    
                    # 拼接剩余的样本
                    for i in range(1, len(all_times)):
                        curr_time = all_times[i].copy()
                        curr_data = all_datas[i].copy()
                        
                        # 计算重叠量
                        overlap_len = int(len(curr_time) * overlap_ratio)
                        if overlap_len >= len(curr_time):
                            overlap_len = max(0, len(curr_time) - 1)  # 确保至少保留一个元素
                        
                        # 跳过前面的重叠部分
                        curr_time = curr_time[overlap_len:].copy()
                        curr_data = curr_data[overlap_len:].copy()
                        
                        if len(curr_time) == 0:
                            # 如果跳过重叠部分后没有数据，则跳过这个样本
                            continue
                        
                        # 调整时间，使其连续
                        time_interval = 0.001  # 默认时间间隔
                        if len(curr_time) > 1:
                            # 使用原始时间序列的间隔
                            time_interval = curr_time[1] - curr_time[0]
                        
                        # 调整当前时间序列，使其从上一个时间序列的末尾开始
                        curr_time_adjusted = last_time + time_interval + np.arange(len(curr_time)) * time_interval
                        
                        # 拼接时间和数据
                        merged_time = np.concatenate([merged_time, curr_time_adjusted])
                        merged_data = np.concatenate([merged_data, curr_data])
                        
                        # 更新最后的时间点
                        last_time = merged_time[-1]
                    
                    # 检查合并后的数据长度是否合理
                    if len(merged_time) > 0 and len(merged_data) > 0:
                        # 保存合并结果
                        merged_times.append(merged_time)
                        merged_datas.append(merged_data)
                        merged_labels.append(list(combination))  # 转换为列表
                        
                        combinations_created += 1
                        samples_for_combination += 1
                        pbar.update(1)
                    else:
                        failed_tries += 1
                except Exception as e:
                    failed_tries += 1
                    print(f"合成样本时出错: {e}")
            except Exception as e:
                failed_tries += 1
                print(f"组合样本时出错: {e}")
            
            # 每100次尝试打印一次进度
            if total_tries % 100 == 0:
                print(f"进度: {combinations_created}样本已生成，尝试次数: {total_tries}，失败次数: {failed_tries}")
        
        print(f"类别组合 {combination} 已生成 {samples_for_combination}/{samples_per_combination} 样本，尝试次数: {attempts_for_combination}")
        
        # 清理内存
        gc.collect()
    
    pbar.close()
    
    print(f"共生成{combinations_created}个多标签样本，尝试次数: {total_tries}，失败次数: {failed_tries}")
    
    # 检查是否有足够的样本
    if combinations_created == 0:
        print("错误: 没有成功生成任何样本")
        return
    
    # 将数据转换为numpy数组格式
    # 由于序列长度不一，需要进行填充至指定的最大长度
    print(f"填充数据至最大长度: {max_sequence_length}")
    
    # 填充数据
    padded_times = []
    padded_datas = []
    
    for time, data in zip(merged_times, merged_datas):
        padded_time = np.zeros(max_sequence_length)
        padded_data = np.zeros(max_sequence_length)
        
        # 确保不超过最大长度
        actual_length = min(len(time), max_sequence_length)
        
        padded_time[:actual_length] = time[:actual_length]
        padded_data[:actual_length] = data[:actual_length]
        
        padded_times.append(padded_time)
        padded_datas.append(padded_data)
    
    # 转换为numpy数组
    padded_times = np.array(padded_times)
    padded_datas = np.array(padded_datas)
    labels = np.array(merged_labels)
    
    # 文件名前缀：根据是否包含未监控类别决定
    prefix = "OW" if include_unmonitored else "CW"
    
    # 保存为npz格式
    output_file = os.path.join(output_dir, f"{prefix}_{num_tabs}tab_overlap_{int(overlap_ratio*100)}.npz")
    np.savez(output_file, 
             time=padded_times, 
             data=padded_datas, 
             label=labels)
    
    print(f"数据已保存至: {output_file}")
    print(f"数据形状: time={padded_times.shape}, data={padded_datas.shape}, label={labels.shape}")
    
    # 保存统计信息
    stats = {
        "num_samples": len(merged_times),
        "num_tabs": num_tabs,
        "overlap_ratio": overlap_ratio,
        "max_sequence_length": int(max_sequence_length),
        "monitored_categories": [int(c) for c in monitored_categories],
        "unmonitored_class": int(unmonitored_class) if include_unmonitored else None,
        "shape": {
            "time": padded_times.shape,
            "data": padded_datas.shape,
            "label": labels.shape
        }
    }
    
    with open(os.path.join(output_dir, f"{prefix}_{num_tabs}tab_overlap_{int(overlap_ratio*100)}_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"统计信息已保存至: {os.path.join(output_dir, f'{prefix}_{num_tabs}tab_overlap_{int(overlap_ratio*100)}_stats.json')}")

def generate_multi_tab_data(data_dir, output_dir, num_tabs=3, samples_per_combination=30, include_unmonitored=False):
    """
    生成多标签数据集
    
    参数:
        data_dir: 基础数据目录
        output_dir: 输出目录
        num_tabs: 标签数量
        samples_per_combination: 每个类别组合生成的样本数量
        include_unmonitored: 是否包含未监控类别
    """
    import itertools
    
    # 源数据目录（单标签数据）
    train_single_tab_dir = os.path.join(data_dir, "train/single_tab")
    test_single_tab_dir = os.path.join(data_dir, "test/single_tab")
    
    # 目标数据目录（多标签数据）
    train_multi_tab_dir = os.path.join(data_dir, "train/multi_tab")
    test_multi_tab_dir = os.path.join(data_dir, "test/multi_tab")
    
    # 确保目标目录存在
    if not os.path.exists(train_multi_tab_dir):
        os.makedirs(train_multi_tab_dir)
    if not os.path.exists(test_multi_tab_dir):
        os.makedirs(test_multi_tab_dir)
    
    # 生成训练集多标签数据
    print("\n生成训练集多标签数据:")
    create_multi_tab_dataset(
        train_single_tab_dir, 
        train_multi_tab_dir,
        num_tabs=num_tabs,
        include_unmonitored=include_unmonitored,
        overlap_ratio=0.1, 
        samples_per_combination=samples_per_combination,
        max_sequence_length=num_tabs * 10000,
        seed=42
    )
    
    # 生成测试集多标签数据
    print("\n生成测试集多标签数据:")
    create_multi_tab_dataset(
        test_single_tab_dir, 
        test_multi_tab_dir,
        num_tabs=num_tabs,
        include_unmonitored=include_unmonitored,
        overlap_ratio=0.1, 
        samples_per_combination=max(5, samples_per_combination // 5),  # 测试集样本数量减少
        max_sequence_length=num_tabs * 10000,
        seed=43  # 使用不同的种子
    )
    
    print("\n多标签数据生成完成!")

def parse_args():
    """
    解析命令行参数
    
    返回:
        命令行参数对象
    """
    parser = argparse.ArgumentParser(description='处理ARES OW数据集，并转换为meta-finger项目格式')
    
    parser.add_argument('--input', type=str, 
                        default="/home/ubuntu22/multi-tab-work/Datasets/OW.npz/OW.npz",
                        help='输入数据文件路径')
    
    parser.add_argument('--output', type=str, 
                        default="/home/ubuntu22/multi-tab-work/meta-finger/data/ARES_OW",
                        help='输出基础目录路径')
    
    parser.add_argument('--remap', action='store_true',
                        help='是否将未监控类别重映射为num_classes+1')
    
    parser.add_argument('--skip-unmonitored', action='store_true',
                        help='是否跳过未监控类别（不保存）')
    
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='训练集比例（默认0.8）')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认42）')
    
    parser.add_argument('--batch-size', type=int, default=500,
                        help='保存样本时的批处理大小（默认500）')
                        
    parser.add_argument('--unmonitored-threshold', type=float, default=30.0,
                        help='判断未监控类别的阈值，表示占总样本的百分比（默认30.0）')
    
    parser.add_argument('--generate-multi', action='store_true',
                        help='是否生成多标签数据集')
    
    parser.add_argument('--num-tabs', type=int, default=3,
                        help='多标签数据中的标签数量（默认3）')
    
    parser.add_argument('--samples-per-combination', type=int, default=30,
                        help='每个类别组合生成的多标签样本数量（默认30）')
    
    parser.add_argument('--include-unmonitored', action='store_true',
                        help='是否在多标签数据中包含未监控类别（OW模式）')
    
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 打印设置
    print("处理设置:")
    print(f"  - 输入文件: {args.input}")
    print(f"  - 输出目录: {args.output}")
    print(f"  - 重映射未监控类别: {args.remap}")
    print(f"  - 跳过未监控类别: {args.skip_unmonitored}")
    print(f"  - 训练集比例: {args.train_ratio}")
    print(f"  - 随机种子: {args.seed}")
    print(f"  - 批处理大小: {args.batch_size}")
    print(f"  - 未监控类别阈值: {args.unmonitored_threshold}%")
    
    if args.generate_multi:
        print(f"  - 生成多标签数据: 是")
        print(f"  - 标签数量: {args.num_tabs}")
        print(f"  - 每类样本数: {args.samples_per_combination}")
        print(f"  - 包含未监控类别: {args.include_unmonitored}")
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input) and not args.generate_multi:
        print(f"错误: 输入文件不存在: {args.input}")
        exit(1)
    
    try:
        # 处理单标签数据
        if not args.generate_multi:
            process_ARES_data(
                input_file=args.input,
                output_base_dir=args.output,
                remap_unmonitored=args.remap,
                skip_unmonitored=args.skip_unmonitored,
                train_ratio=args.train_ratio,
                seed=args.seed,
                batch_size=args.batch_size,
                unmonitored_threshold=args.unmonitored_threshold
            )
        
        # 生成多标签数据
        if args.generate_multi:
            generate_multi_tab_data(
                data_dir=args.output,
                output_dir=args.output,
                num_tabs=args.num_tabs,
                samples_per_combination=args.samples_per_combination,
                include_unmonitored=args.include_unmonitored
            )
            
    except Exception as e:
        print(f"处理数据时发生错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 