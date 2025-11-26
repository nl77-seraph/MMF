"""
处理OW.npz文件，将数据重新组织为每个标签一个文件夹，每条数据一个pkl文件
"""

import numpy as np
import pickle
import os
from tqdm import tqdm
import argparse

def process_npz(npz_path, output_dir):
    """

    
    Args:
        npz_path: npz文件路径
        output_dir: 输出目录路径
    """
    # 加载npz文件
    print(f"正在加载 {npz_path}...")
    data = np.load(npz_path)
    X = data['X']
    y = data['y']
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"数据示例: {X[0][:4]}")  # 显示第一条数据的前4个元素
    
    # 获取唯一标签
    unique_labels = np.unique(y)
    print(f"唯一标签数量: {len(unique_labels)}")
    print(f"标签范围: {unique_labels.min()} - {unique_labels.max()}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个标签创建文件夹
    for label in unique_labels:
        label_dir = os.path.join(output_dir, str(int(label)))
        os.makedirs(label_dir, exist_ok=True)
    
    # 处理每条数据
    print("\n开始处理数据...")
    for idx in tqdm(range(len(X)), desc="处理进度"):
        # 获取当前数据
        trace = X[idx]
        label = y[idx]
        
        # 找到非零元素（去除填充的0）
        non_zero_mask = trace != 0
        if np.any(non_zero_mask):
            trace = trace[non_zero_mask]
        else:
            # 如果全是零，跳过这条数据
            continue
        
        # 提取时间戳和方向序列
        X_dir = np.sign(trace)  # 方向序列（正负信息）
        X_time = np.abs(trace)  # 时间戳信息（绝对值）
        
        # 创建数据字典，格式与你的读取代码兼容
        data_dict = {
            'time': X_time,      # 时间戳数组
            'data': X_dir,       # 方向序列数组
            'label': int(label)  # 标签
        }
        
        # 保存为pkl文件
        label_dir = os.path.join(output_dir, str(int(label)))
        file_name = f"trace_{idx}.pkl"
        file_path = os.path.join(label_dir, file_name)
        
        with open(file_path, 'wb') as f:
            pickle.dump(data_dict, f)
    
    print(f"\n处理完成！共处理 {len(X)} 条数据")
    
    # 统计每个标签的数据数量
    print("\n每个标签的数据统计:")
    for label in unique_labels:
        label_dir = os.path.join(output_dir, str(int(label)))
        if os.path.exists(label_dir):
            count = len([f for f in os.listdir(label_dir) if f.endswith('.pkl')])
            print(f"标签 {int(label)}: {count} 条数据")

def verify_pkl_files(output_dir, num_samples=3):
    """
    验证生成的pkl文件格式
    
    Args:
        output_dir: 输出目录路径
        num_samples: 要验证的样本数量
    """
    print("\n验证生成的pkl文件...")
    
    # 获取所有标签文件夹
    label_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    for label_dir in label_dirs[:min(num_samples, len(label_dirs))]:
        label_path = os.path.join(output_dir, label_dir)
        pkl_files = [f for f in os.listdir(label_path) if f.endswith('.pkl')]
        
        if pkl_files:
            # 读取第一个pkl文件进行验证
            test_file = os.path.join(label_path, pkl_files[0])
            print(f"\n验证文件: {test_file}")
            
            # 使用你提供的读取方式
            with open(test_file, 'rb') as f:
                raw_dict = pickle.load(f)
            
            raw_times = raw_dict['time']
            raw_datas = raw_dict['data']
            raw_labels = raw_dict['label']
            
            print(f"  标签: {raw_labels}")
            print(f"  时间戳数量: {len(raw_times)}")
            print(f"  方向序列数量: {len(raw_datas)}")
            print(f"  时间戳示例: {raw_times[:5] if len(raw_times) >= 5 else raw_times}")
            print(f"  方向序列示例: {raw_datas[:5] if len(raw_datas) >= 5 else raw_datas}")
            print(f"  数据长度一致性检查: {len(raw_times) == len(raw_datas)}")

def main():
    parser = argparse.ArgumentParser(description='处理OW.npz数据文件')
    parser.add_argument('--npz_path', type=str, default='/root/datasets/OW.npz', 
                        help='npz文件路径')
    parser.add_argument('--output_dir', type=str, default='/root/datasets/OW', 
                        help='输出目录路径')
    parser.add_argument('--verify', action='store_true', 
                        help='是否验证生成的文件')
    
    args = parser.parse_args()
    
    # 处理数据
    process_npz(args.npz_path, args.output_dir)
    
    # 验证文件
    if args.verify:
        verify_pkl_files(args.output_dir)

if __name__ == "__main__":
    main()