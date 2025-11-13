"""
3标签多标签数据生成脚本
基于process_ARES_data.py中的merge_with_durationscale函数
实现流式处理和多线程优化
"""

import os
import pickle
import random
import json
import numpy as np
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import time
import uuid

# 复制process_ARES_data.py中的merge_with_durationscale函数
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

def load_pickle_sample(file_path):
    """加载pickle样本文件"""
    try:
        with open(file_path, 'rb') as f:
            sample = pickle.load(f)
        return sample['time'], sample['data']
    except Exception as e:
        print(f"加载样本失败 {file_path}: {e}")
        return None, None

def merge_3tab_sequences(sample_paths, overlap_ratio=0.1):
    """
    合成3标签序列，确保标签顺序与数据合成顺序对应
    
    参数:
        sample_paths: 按标签顺序排列的样本路径列表
        overlap_ratio: 重叠比例
    
    返回:
        merged_time, merged_data: 合成后的时间和数据序列
        None, None: 如果合成失败
    """
    # 按标签顺序加载样本
    times, datas = [], []
    
    for path in sample_paths:  # 确保路径顺序对应标签顺序
        time_seq, data_seq = load_pickle_sample(path)
        if time_seq is None or data_seq is None:
            return None, None
        times.append(time_seq)
        datas.append(data_seq)
    
    # 使用duration方式合成
    result = merge_with_durationscale(times, datas, ratio=overlap_ratio)
    
    if result is None:
        return None, None
    
    merged_time, merged_data = result
    return np.array(merged_time), np.array(merged_data)

class Tab3DataGenerator:
    """3标签数据生成器"""
    
    def __init__(self, support_data_dir, output_dir, overlap_ratio=0.1, samples_per_combination=5):
        self.support_data_dir = support_data_dir
        self.output_dir = output_dir
        self.overlap_ratio = overlap_ratio
        self.samples_per_combination = samples_per_combination
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 建立样本文件索引
        self.class_samples = {}
        self._build_sample_index()
        
        # JSON索引数据
        self.train_index = []
        self.val_index = []
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 统计信息
        self.total_generated = 0
        self.failed_attempts = 0
        
    def _build_sample_index(self):
        """建立每个类别的样本文件索引"""
        print("🔍 建立样本文件索引...")
        
        for class_id in range(60):  # 0-59类别
            class_dir = os.path.join(self.support_data_dir, str(class_id))
            
            if os.path.exists(class_dir):
                pkl_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.pkl')]
                if pkl_files:
                    self.class_samples[class_id] = pkl_files
                    print(f"  类别{class_id}: {len(pkl_files)}个样本")
                else:
                    print(f"  ⚠️  类别{class_id}: 没有找到pkl文件")
            else:
                print(f"  ❌ 类别{class_id}: 目录不存在")
        
        print(f"✅ 样本索引建立完成，共{len(self.class_samples)}个有效类别")
    
    def generate_combinations(self):
        """生成所有3类别组合"""
        available_classes = list(self.class_samples.keys())
        if len(available_classes) < 3:
            raise ValueError(f"可用类别数量不足: {len(available_classes)} < 3")
        
        combinations_list = list(combinations(available_classes, 3))
        print(f"📋 生成了{len(combinations_list)}种3类别组合")
        return combinations_list
    
    def process_single_combination(self, combination, sample_id):
        """处理单个组合的单次采样"""
        try:
            # 按顺序选择样本路径（确保标签顺序与数据顺序对应）
            sample_paths = []
            for class_id in combination:  # combination已经是排序的
                available_samples = self.class_samples[class_id]
                selected_sample = random.choice(available_samples)
                sample_paths.append(selected_sample)
            
            # 合成数据
            merged_time, merged_data = merge_3tab_sequences(sample_paths, self.overlap_ratio)
            
            if merged_time is None or merged_data is None:
                return None
            
            # 生成文件名：类别1_类别2_类别3_随机ID.pkl
            labels_str = "_".join(map(str, combination))
            random_id = str(uuid.uuid4())[:8]
            filename = f"{labels_str}_{random_id}.pkl"
            
            # 保存数据
            sample_data = {
                'time': merged_time,
                'data': merged_data,
                'labels': list(combination),  # 保持顺序
                'source_files': sample_paths
            }
            
            file_path = os.path.join(self.output_dir, filename)
            with open(file_path, 'wb') as f:
                pickle.dump(sample_data, f)
            
            # 创建索引条目
            index_entry = {
                'filename': filename,
                'labels': list(combination),
                'data_length': len(merged_data),
                'time_range': [float(merged_time[0]), float(merged_time[-1])]
            }
            
            # 按4:1分配训练和验证
            if sample_id < 4:  # 前4个作为训练集
                dataset_type = 'train'
            else:  # 第5个作为验证集
                dataset_type = 'val'
            
            return index_entry, dataset_type
            
        except Exception as e:
            print(f"处理组合{combination}失败: {e}")
            return None
    
    def process_combination_batch(self, combination):
        """处理单个组合的所有采样（5次）"""
        results = []
        
        for sample_id in range(self.samples_per_combination):
            result = self.process_single_combination(combination, sample_id)
            if result is not None:
                results.append(result)
            else:
                with self.lock:
                    self.failed_attempts += 1
        
        # 更新索引
        with self.lock:
            for index_entry, dataset_type in results:
                if dataset_type == 'train':
                    self.train_index.append(index_entry)
                else:
                    self.val_index.append(index_entry)
            
            self.total_generated += len(results)
        
        return len(results)
    
    def generate_all_data(self, num_threads=4):
        """生成所有多标签数据"""
        print(f"🚀 开始生成3标签多标签数据...")
        print(f"  - 重叠比例: {self.overlap_ratio}")
        print(f"  - 每组合样本数: {self.samples_per_combination}")
        print(f"  - 线程数: {num_threads}")
        
        # 生成所有组合
        combinations_list = self.generate_combinations()
        total_expected = len(combinations_list) * self.samples_per_combination
        
        print(f"  - 预期生成样本总数: {total_expected}")
        
        # 多线程并行处理
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 提交所有任务
            futures = []
            for combination in combinations_list:
                future = executor.submit(self.process_combination_batch, combination)
                futures.append((combination, future))
            
            # 收集结果并显示进度
            with tqdm(total=len(combinations_list), desc="处理组合") as pbar:
                for combination, future in futures:
                    try:
                        generated_count = future.result(timeout=60)  # 60秒超时
                        pbar.set_postfix({
                            '已生成': self.total_generated,
                            '失败': self.failed_attempts,
                            '当前组合': f"{combination[0]}-{combination[1]}-{combination[2]}"
                        })
                        pbar.update(1)
                    except Exception as e:
                        print(f"组合{combination}处理超时或出错: {e}")
                        pbar.update(1)
        
        # 保存JSON索引
        self.save_json_indices()
        
        # 生成统计报告
        self.generate_report()
    
    def save_json_indices(self):
        """保存JSON索引文件"""
        print(f"\n💾 保存JSON索引文件...")
        
        # 提取文件名列表（兼容现有数据加载器格式）
        train_filenames = [entry['filename'] for entry in self.train_index]
        val_filenames = [entry['filename'] for entry in self.val_index]
        
        # 保存训练集索引
        train_json_path = os.path.join(os.path.dirname(self.output_dir), "3tab_train.json")
        with open(train_json_path, 'w', encoding='utf-8') as f:
            json.dump(train_filenames, f, ensure_ascii=False, indent=2)
        
        # 保存验证集索引
        val_json_path = os.path.join(os.path.dirname(self.output_dir), "3tab_val.json")
        with open(val_json_path, 'w', encoding='utf-8') as f:
            json.dump(val_filenames, f, ensure_ascii=False, indent=2)
        
        # 同时保存详细的元数据索引（用于分析）
        detailed_train_path = os.path.join(os.path.dirname(self.output_dir), "3tab_train_detailed.json")
        with open(detailed_train_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_index, f, ensure_ascii=False, indent=2)
            
        detailed_val_path = os.path.join(os.path.dirname(self.output_dir), "3tab_val_detailed.json")
        with open(detailed_val_path, 'w', encoding='utf-8') as f:
            json.dump(self.val_index, f, ensure_ascii=False, indent=2)
        
        print(f"  ✅ 训练集索引: {train_json_path} ({len(train_filenames)}条)")
        print(f"  ✅ 验证集索引: {val_json_path} ({len(val_filenames)}条)")
        print(f"  📋 详细元数据: {detailed_train_path}, {detailed_val_path}")
    
    def generate_report(self):
        """生成统计报告"""
        print(f"\n📊 数据生成统计报告:")
        print(f"  - 总生成样本: {self.total_generated}")
        print(f"  - 训练集样本: {len(self.train_index)}")
        print(f"  - 验证集样本: {len(self.val_index)}")
        print(f"  - 失败尝试: {self.failed_attempts}")
        print(f"  - 成功率: {self.total_generated/(self.total_generated+self.failed_attempts)*100:.2f}%")
        
        # 检查数据分布
        if len(self.train_index) > 0:
            train_lengths = [entry['data_length'] for entry in self.train_index]
            print(f"  - 训练集数据长度: {np.min(train_lengths)}-{np.max(train_lengths)} (平均{np.mean(train_lengths):.0f})")
        
        if len(self.val_index) > 0:
            val_lengths = [entry['data_length'] for entry in self.val_index]
            print(f"  - 验证集数据长度: {np.min(val_lengths)}-{np.max(val_lengths)} (平均{np.mean(val_lengths):.0f})")

def main():
    """主函数"""
    # 配置参数
    support_data_dir = "datasets/3tab_exp/base_train/support_data"
    output_dir = "datasets/3tab_exp/base_train/query_data"
    overlap_ratio = 0.1
    samples_per_combination = 5
    num_threads = 4
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    print("🎯 3标签多标签数据生成器")
    print("="*50)
    
    # 创建生成器
    generator = Tab3DataGenerator(
        support_data_dir=support_data_dir,
        output_dir=output_dir,
        overlap_ratio=overlap_ratio,
        samples_per_combination=samples_per_combination
    )
    
    # 生成所有数据
    start_time = time.time()
    generator.generate_all_data(num_threads=num_threads)
    end_time = time.time()
    
    print(f"\n🎉 数据生成完成！")
    print(f"⏱️  总耗时: {end_time - start_time:.2f}秒")

if __name__ == "__main__":
    main() 