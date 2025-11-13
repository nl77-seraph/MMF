"""
验证支持集数据完整性的脚本
检查pickle文件格式和数据内容
"""

import os
import pickle
import numpy as np
from tqdm import tqdm

def verify_pickle_file(file_path):
    """验证单个pickle文件"""
    try:
        with open(file_path, 'rb') as f:
            sample = pickle.load(f)
        
        # 检查必要字段
        if 'data' not in sample or 'time' not in sample:
            return False, f"缺少必要字段: {list(sample.keys())}"
        
        # 检查数据类型和长度
        data = sample['data']
        time = sample['time']
        
        if len(data) != len(time):
            return False, f"数据长度不匹配: data={len(data)}, time={len(time)}"
        
        if len(data) == 0:
            return False, "数据为空"
            
        return True, f"数据长度: {len(data)}"
        
    except Exception as e:
        return False, f"读取错误: {str(e)}"

def verify_support_data(support_data_dir):
    """验证整个支持集数据目录"""
    print(f"开始验证支持集数据: {support_data_dir}")
    
    # 统计信息
    total_files = 0
    valid_files = 0
    error_files = []
    
    # 检查每个类别
    for class_id in range(60):  # 0-59类别
        class_dir = os.path.join(support_data_dir, str(class_id))
        
        if not os.path.exists(class_dir):
            print(f"❌ 类别{class_id}目录不存在")
            continue
        
        # 检查该类别的所有文件
        pkl_files = [f for f in os.listdir(class_dir) if f.endswith('.pkl')]
        
        if len(pkl_files) != 50:
            print(f"⚠️  类别{class_id}样本数量异常: {len(pkl_files)} (期望50)")
        
        print(f"验证类别{class_id}: {len(pkl_files)}个文件")
        
        for pkl_file in tqdm(pkl_files, desc=f"类别{class_id}", leave=False):
            file_path = os.path.join(class_dir, pkl_file)
            total_files += 1
            
            is_valid, message = verify_pickle_file(file_path)
            
            if is_valid:
                valid_files += 1
            else:
                error_files.append((file_path, message))
                if len(error_files) <= 5:  # 只显示前5个错误
                    print(f"  ❌ {pkl_file}: {message}")
    
    # 输出统计结果
    print(f"\n📊 验证结果统计:")
    print(f"  - 总文件数: {total_files}")
    print(f"  - 有效文件: {valid_files}")
    print(f"  - 错误文件: {len(error_files)}")
    print(f"  - 成功率: {valid_files/total_files*100:.2f}%")
    
    if error_files:
        print(f"\n❌ 错误文件列表 (显示前10个):")
        for i, (file_path, error) in enumerate(error_files[:10]):
            print(f"  {i+1}. {file_path}: {error}")
    
    return total_files, valid_files, error_files

def sample_data_inspection(support_data_dir):
    """抽样检查数据内容"""
    print(f"\n🔍 抽样检查数据内容:")
    
    # 随机选择几个文件进行详细检查
    sample_files = [
        "datasets/3tab_exp/base_train/support_data/0/sample_0.pkl",
        "datasets/3tab_exp/base_train/support_data/30/sample_15.pkl",
        "datasets/3tab_exp/base_train/support_data/59/sample_25.pkl"
    ]
    
    for file_path in sample_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    sample = pickle.load(f)
                
                data = sample['data']
                time = sample['time']
                
                print(f"\n📁 {file_path}:")
                print(f"  - 数据长度: {len(data)}")
                print(f"  - 数据范围: [{np.min(data):.3f}, {np.max(data):.3f}]")
                print(f"  - 时间范围: [{np.min(time):.3f}, {np.max(time):.3f}]")
                print(f"  - 数据类型: {type(data)}, {type(time)}")
                print(f"  - 前5个数据值: {data[:5]}")
                print(f"  - 前5个时间值: {time[:5]}")
                
            except Exception as e:
                print(f"  ❌ 读取失败: {e}")

if __name__ == "__main__":
    support_data_dir = "datasets/3tab_exp/base_train/support_data"
    
    # 验证数据完整性
    total_files, valid_files, error_files = verify_support_data(support_data_dir)
    
    # 抽样检查数据内容
    sample_data_inspection("./")
    
    # 最终报告
    print(f"\n🎯 验证完成！")
    if len(error_files) == 0:
        print("✅ 所有数据文件验证通过，可以进行下一步操作")
    else:
        print(f"⚠️  发现{len(error_files)}个问题文件，建议检查和修复") 