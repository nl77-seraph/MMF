"""
文件: process_data.py
功能: 处理TMWF项目中的单标签数据，并将其转换为meta-finger项目需要的格式
作者: Claude
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
import json
import shutil
import random
import copy

def load_raw_data(file_path):
    """
    加载TMWF项目中的原始数据
    
    参数:
        file_path: 原始数据文件路径
        
    返回:
        raw_times: 时间序列数据
        raw_datas: 方向序列数据
        raw_labels: 标签数据
    """
    if file_path.split('.')[-1] == 'npz':    # 处理BAPM格式的数据集
        raw_times = np.load(file_path)['time']
        raw_datas = np.load(file_path)['data']
        raw_labels = np.load(file_path)['label']
    else:
        with open(file_path, 'rb') as f:
            raw_dict = pickle.load(f)
        raw_times = raw_dict['time']
        raw_datas = raw_dict['data']
        raw_labels = raw_dict['label']
    return raw_times, raw_datas, raw_labels

def rm_zero(sequence):
    """
    移除序列末尾的零填充
    
    参数:
        sequence: 输入序列
        
    返回:
        处理后的序列
    """
    index = len(sequence)
    for i in range(len(sequence) - 1, -1, -1):
        if sequence[i] != 0:
            index = i + 1
            break
    return sequence[:index]

def process_single_tab_data(input_path, output_dir, max_samples_per_class=None):
    """
    处理单标签数据并按类别分类存储
    
    参数:
        input_path: TMWF项目中的单标签数据文件路径
        output_dir: 输出目录
        max_samples_per_class: 每个类别最大样本数量，None表示不限制
    """
    print(f"开始处理文件: {input_path}")
    
    # 加载原始数据
    try:
        raw_times, raw_datas, raw_labels = load_raw_data(input_path)
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    total_num = len(raw_labels)
    print(f"成功加载数据，共 {total_num} 个样本")
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 统计各类别样本数量
    class_counts = {}
    for label in raw_labels:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    print(f"数据包含 {len(class_counts)} 个类别")
    
    # 创建类别目录和处理样本
    samples_processed = 0
    samples_by_class = {}
    
    for i in tqdm(range(total_num), desc="处理样本"):
        label = int(raw_labels[i])
        
        # 跳过标签为-1的样本（未监控样本）
        if label < 0:
            continue
            
        # 创建类别目录
        class_dir = os.path.join(output_dir, str(label))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        # 检查当前类别的样本数量是否已达到限制
        if label not in samples_by_class:
            samples_by_class[label] = 0
            
        if max_samples_per_class is not None and samples_by_class[label] >= max_samples_per_class:
            continue
            
        # 处理当前样本
        current_time = raw_times[i]
        current_data = raw_datas[i]
        
        # 移除末尾零填充
        if isinstance(current_time, np.ndarray) and current_time.size > 0:
            current_time = rm_zero(current_time)
        if isinstance(current_data, np.ndarray) and current_data.size > 0:
            current_data = rm_zero(current_data)
            
        # 创建样本字典
        sample_dict = {
            'time': current_time,
            'data': current_data,
            'label': label
        }
        
        # 保存样本
        sample_path = os.path.join(class_dir, f"sample_{samples_by_class[label]}.pkl")
        with open(sample_path, 'wb') as f:
            pickle.dump(sample_dict, f)
            
        samples_by_class[label] += 1
        samples_processed += 1
    
    # 统计结果
    print(f"处理完成，共处理 {samples_processed} 个样本")
    print("各类别样本数量:")
    for class_id, count in sorted(samples_by_class.items()):
        print(f"类别 {class_id}: {count} 个样本")
    
    # 保存统计信息
    stats = {
        "total_samples": samples_processed,
        "class_distribution": {str(k): v for k, v in samples_by_class.items()}
    }
    
    with open(os.path.join(output_dir, "stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"统计信息已保存至 {os.path.join(output_dir, 'stats.json')}")

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
    """
    time = times[0]
    data = datas[0]
    split_index = 0
    
    # 根据重叠比例找到分割点
    split_time = np.max(time) * (1 - ratio)
    for i, packet_time in enumerate(time):
        if packet_time >= split_time:
            split_index = i
            break
    
    # 添加第一个页面的前半部分到合并序列
    merged_time = list(time[:split_index])
    merged_data = list(data[:split_index])
    
    # 剩余部分用于与下一个页面重叠
    res_time = time[split_index:]
    res_data = data[split_index:]
    
    # 处理剩余的页面
    for time, data in list(zip(times, datas))[1:]:
        # 调整时间序列，使其与前一个页面的剩余部分重叠
        time = [(t + split_time) for t in time]
        
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
        
        # 如果第一个序列没有完全遍历，或第二个序列已完全遍历，则合并失败
        if index1 < len(res_time) or index2 == len(time):
            return None
        
        # 根据重叠比例找到第二个页面的分割点
        split_time = np.max(time) * (1 - ratio)
        
        # 如果第二个页面的分割点在已遍历部分之后，验证是否有足够的干净部分
        if split_time <= time[index2]:
            return None
        
        # 寻找分割索引
        for i, packet_time in enumerate(time):
            if packet_time >= split_time:
                split_index = i
                break
        
        # 添加第二个页面的中间部分到合并序列
        if index2 < len(time):
            merged_time.extend(time[index2:split_index])
            merged_data.extend(data[index2:split_index])
        
        # 更新剩余部分用于下一次合并
        res_time = time[split_index:]
        res_data = data[split_index:]
    
    # 添加最后一个页面的剩余部分
    merged_time.extend(res_time)
    merged_data.extend(res_data)
    
    return merged_time, merged_data

def create_multi_tab_dataset(single_tab_dir, output_dir, num_tabs=2, overlap_ratio=0.1, samples_per_combination=10):
    """
    从单标签数据创建多标签数据集
    
    参数:
        single_tab_dir: 单标签数据目录
        output_dir: 输出目录
        num_tabs: 每个样本的标签数量
        overlap_ratio: 重叠比例
        samples_per_combination: 每个标签组合生成的样本数量
    """
    print(f"开始生成{num_tabs}标签数据集，重叠比例: {overlap_ratio}")
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载所有可用类别
    categories = []
    for item in os.listdir(single_tab_dir):
        if os.path.isdir(os.path.join(single_tab_dir, item)) and item.isdigit():
            categories.append(int(item))
    
    categories.sort()
    print(f"找到{len(categories)}个类别: {categories}")
    
    # 创建样本索引
    category_samples = {}
    for category in categories:
        category_dir = os.path.join(single_tab_dir, str(category))
        sample_files = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.pkl')]
        category_samples[category] = sample_files
        print(f"类别{category}：找到{len(sample_files)}个样本")
    
    # 保存所有样本的列表
    merged_times = []
    merged_datas = []
    merged_labels = []
    
    # 随机组合不同类别生成多标签样本
    combinations_created = 0
    max_tries = samples_per_combination * 10  # 最大尝试次数
    total_tries = 0
    
    pbar = tqdm(total=samples_per_combination * len(categories), desc="生成多标签样本")
    
    # 确保每个类别都作为主类别至少出现samples_per_combination次
    for main_category in categories:
        combinations_for_category = 0
        category_tries = 0
        
        while combinations_for_category < samples_per_combination and category_tries < max_tries:
            category_tries += 1
            total_tries += 1
            
            # 选择主类别样本
            main_sample_path = random.choice(category_samples[main_category])
            
            # 随机选择其他类别
            other_categories = random.sample([c for c in categories if c != main_category], k=num_tabs-1)
            all_categories = [main_category] + other_categories
            
            # 加载所有样本
            sample_paths = [main_sample_path]
            for category in other_categories:
                sample_paths.append(random.choice(category_samples[category]))
            
            times = []
            datas = []
            
            # 加载样本数据
            for path in sample_paths:
                with open(path, 'rb') as f:
                    sample = pickle.load(f)
                times.append(sample['time'])
                datas.append(sample['data'])
            
            # 合并样本
            merged_result = merge_with_durationscale(times, datas, overlap_ratio)
            if merged_result is not None:
                merged_time, merged_data = merged_result
                
                # 保存合并结果
                merged_times.append(merged_time)
                merged_datas.append(merged_data)
                merged_labels.append(all_categories)
                
                combinations_created += 1
                combinations_for_category += 1
                pbar.update(1)
    
    pbar.close()
    
    print(f"共生成{combinations_created}个多标签样本，尝试次数: {total_tries}")
    
    # 将数据转换为numpy数组格式
    # 由于序列长度不一，需要进行填充
    max_length = max(len(time) for time in merged_times)
    print(f"最大序列长度: {max_length}")
    
    # 填充数据
    padded_times = []
    padded_datas = []
    
    for time, data in zip(merged_times, merged_datas):
        padded_time = np.zeros(max_length)
        padded_data = np.zeros(max_length)
        
        padded_time[:len(time)] = time
        padded_data[:len(data)] = data
        
        padded_times.append(padded_time)
        padded_datas.append(padded_data)
    
    # 转换为numpy数组
    padded_times = np.array(padded_times)
    padded_datas = np.array(padded_datas)
    labels = np.array(merged_labels)
    
    # 保存为npz格式
    output_file = os.path.join(output_dir, f"{num_tabs}tab_overlap_{int(overlap_ratio*100)}.npz")
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
        "max_sequence_length": int(max_length),
        "categories_used": [int(c) for c in categories],
        "shape": {
            "time": padded_times.shape,
            "data": padded_datas.shape,
            "label": labels.shape
        }
    }
    
    with open(os.path.join(output_dir, f"{num_tabs}tab_overlap_{int(overlap_ratio*100)}_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)

def main():
    """
    主函数，处理TMWF项目中的单标签数据
    """
    # 设置路径（使用绝对路径）
    tmwf_dir = "/home/ubuntu22/multi-tab-work/TMWF/dataset/chrome single tab"
    meta_finger_data_dir = "/home/ubuntu22/multi-tab-work/meta-finger/data"
    
    # 确保meta-finger/data目录存在
    if not os.path.exists(meta_finger_data_dir):
        os.makedirs(meta_finger_data_dir)
        print(f"创建目录: {meta_finger_data_dir}")
    
    # 创建train和test目录
    train_dir = os.path.join(meta_finger_data_dir, "train/single_tab")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        print(f"创建目录: {train_dir}")
        
    test_dir = os.path.join(meta_finger_data_dir, "test/single_tab")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"创建目录: {test_dir}")
    
    # 处理训练集
    train_file = os.path.join(tmwf_dir, "train")
    if os.path.exists(train_file):
        process_single_tab_data(train_file, train_dir)
    else:
        print(f"警告: 训练集文件不存在 {train_file}")
    
    # 处理测试集
    test_file = os.path.join(tmwf_dir, "test")
    if os.path.exists(test_file):
        process_single_tab_data(test_file, test_dir)
    else:
        print(f"警告: 测试集文件不存在 {test_file}")
    
    print("数据处理完成!")

def generate_multi_tab_data():
    """
    生成多标签数据集
    """
    meta_finger_data_dir = "/home/ubuntu22/multi-tab-work/meta-finger/data"
    
    # 源数据目录（单标签数据）
    train_single_tab_dir = os.path.join(meta_finger_data_dir, "train/single_tab")
    test_single_tab_dir = os.path.join(meta_finger_data_dir, "test/single_tab")
    
    # 目标数据目录（多标签数据）
    train_multi_tab_dir = os.path.join(meta_finger_data_dir, "train/multi_tab")
    test_multi_tab_dir = os.path.join(meta_finger_data_dir, "test/multi_tab")
    
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
        num_tabs=2,
        overlap_ratio=0.1, 
        samples_per_combination=5
    )
    
    # 生成测试集多标签数据
    print("\n生成测试集多标签数据:")
    create_multi_tab_dataset(
        test_single_tab_dir, 
        test_multi_tab_dir,
        num_tabs=2,
        overlap_ratio=0.1, 
        samples_per_combination=2
    )
    
    print("\n多标签数据生成完成!")

if __name__ == "__main__":
    # main()  # 处理单标签数据
    generate_multi_tab_data()  # 生成多标签数据 