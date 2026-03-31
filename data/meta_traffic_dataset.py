
import os
import random
import torch
import numpy as np
import pickle
import json
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset
import torch.distributed as dist
def is_main_process():
    """检查是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0

class QueryTrafficDataset(Dataset):
    """
    查询集数据加载器
    参考Few-shot Detection的listDataset设计
    """
    
    def __init__(self, 
                 json_index_path: str,
                 query_files_dir: str,
                 target_length: int = 30000,
                 activated_classes: List[int] = None):
        """
        Args:
            json_index_path: 查询集索引JSON文件路径
            query_files_dir: 查询集数据文件目录  
            target_length: 目标序列长度
            activated_classes: 激活的类别列表，默认0-59
        """
        self.json_index_path = json_index_path
        self.query_files_dir = query_files_dir
        self.target_length = target_length
        self.activated_classes = activated_classes if activated_classes else list(range(60))  # 0-59
        
        # 加载查询集索引
        self._load_query_index()
        if is_main_process():
            print(f"QueryTrafficDataset初始化完成:")
            print(f"  - 查询样本数量: {len(self.query_index)}")
            print(f"  - 激活类别数量: {len(self.activated_classes)}")
            print(f"  - 目标序列长度: {self.target_length}")
    
    def _load_query_index(self):
        """加载查询集索引"""
        if os.path.exists(self.json_index_path):
            with open(self.json_index_path, 'r') as f:
                query_file_names = json.load(f)
        else:
            # 获取目录下所有以 .pkl 结尾的文件名
            query_file_names = [f for f in os.listdir(self.query_files_dir) if f.endswith('.pkl')]
            # 建议排序，确保不同机器或不同运行次序下加载顺序一致
            query_file_names.sort()
        self.query_index = []
        for filename in query_file_names:
            # 解析文件名中的标签
            labels = self._parse_labels_from_filename(filename)
            
            if labels:  # 只保留有效标签的文件
                file_path = os.path.join(self.query_files_dir, filename)
                self.query_index.append({
                    'filename': filename,
                    'labels': labels,
                    'file_path': file_path
                })
                
        if is_main_process():
            print(f"有效查询样本数量: {len(self.query_index)}")
    
    def _parse_labels_from_filename(self, filename: str) -> List[int]:
        """
        从文件名解析标签
        文件名格式: "类别1_类别2_类别3_随机文件名.pkl"
        """
        basename = os.path.splitext(filename)[0]
        parts = basename.split('_')
        
        labels = []
        for part in parts:
            if 'novel' in part:
                continue
            else:
                try:
                    label = int(part)
                    if label in self.activated_classes:
                        labels.append(label)
                except ValueError:
                    # 非数字部分认为是随机文件名，停止解析
                    break
        
        return labels
    
    def _process_sequence(self, raw_data: List) -> torch.Tensor:
        """
        处理序列数据：截断或填充到目标长度
        """
        if len(raw_data) >= self.target_length:
            # 截断
            processed = raw_data[:self.target_length]
        else:
            # 填充0
            processed = raw_data + [0] * (self.target_length - len(raw_data))
        
        return torch.tensor(processed, dtype=torch.float32)
    
    def _labels_to_multihot(self, labels: List[int]) -> torch.Tensor:
        """
        将标签列表转换为多热编码
        """
        num_classes = len(self.activated_classes)
        multihot = torch.zeros(num_classes, dtype=torch.float32)
        
        for label in labels:
            if label in self.activated_classes:
                idx = self.activated_classes.index(label)
                multihot[idx] = 1.0
                
        return multihot
    
    def __len__(self) -> int:
        return len(self.query_index)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        获取查询样本
        
        Returns:
            query_data: (target_length,) 查询序列
            query_labels: (num_classes,) 多热编码标签
            metadata: 元数据字典
        """
        sample_info = self.query_index[idx]
        
        # 加载数据
        with open(sample_info['file_path'], 'rb') as f:
            sample_data = pickle.load(f)
        
        # 处理不同的数据格式
        if isinstance(sample_data, dict) :
            if 'data' in sample_data:
                raw_data = sample_data['data']
            else: 
                raw_data = sample_data['direction']
        elif isinstance(sample_data, (list, np.ndarray)):
            raw_data = sample_data
        else:
            # 其他格式直接使用
            raw_data = sample_data
        
        # 确保raw_data是列表格式
        if isinstance(raw_data, np.ndarray):
            raw_data = raw_data.tolist()
        elif not isinstance(raw_data, list):
            raw_data = [raw_data]
        
        # 处理序列
        query_data = self._process_sequence(raw_data)
        
        # 处理标签
        query_labels = self._labels_to_multihot(sample_info['labels']) ##Note 如果需要顺序信息，则不可使用Multihot
        
        # 元数据
        metadata = {
            'filename': sample_info['filename'],
            'original_labels': sample_info['labels'],
            'file_path': sample_info['file_path']
        }
        
        return query_data, query_labels, metadata


class SupportTrafficDataset(Dataset):
    """
    支持集数据加载器
    参考Few-shot Detection的MetaDataset设计
    为所有类别生成支持集，无需Episode采样
    支持两种模式：固定采样（用于few-shot调整）和随机采样（用于训练）
    """
    
    def __init__(self,
                 support_root_dir: str,
                 activated_classes: List[int] = None,
                 target_length: int = 30000,
                 shots_per_class: int = 1,
                 random_sampling: bool = False):
        """
        Args:
            support_root_dir: 支持集根目录
            activated_classes: 激活的类别列表，默认0-59
            target_length: 目标序列长度（修正为30000）
            shots_per_class: 每个类别的样本数
            random_sampling: 是否使用随机采样模式（True：每次随机选择，False：固定选择）
        """
        self.support_root_dir = support_root_dir
        self.activated_classes = activated_classes if activated_classes else list(range(60))  # 0-59
        self.target_length = target_length
        self.shots_per_class = shots_per_class
        self.random_sampling = random_sampling
        
        # 构建支持集索引
        self._build_support_index()
        
        if not self.random_sampling:
            # 固定采样模式：预生成所有类别的支持集
            self._prepare_all_support_data()
        else:
            if is_main_process():
            # 随机采样模式：仅记录文件索引，每次动态加载
                print(f"SupportTrafficDataset初始化完成 (随机采样模式):")
                print(f"  - 激活类别数量: {len(self.activated_classes)}")
                print(f"  - 每类样本数: {self.shots_per_class}")
                print(f"  - 目标序列长度: {self.target_length}")
                print(f"  - 随机采样: {self.random_sampling}")
    
    def _build_support_index(self):
        """构建支持集索引"""
        self.support_files_by_class = {}
        
        for class_id in self.activated_classes:
            class_dir = os.path.join(self.support_root_dir, str(class_id))
            if not os.path.exists(class_dir):
                print(f"警告: 类别{class_id}的目录不存在: {class_dir}")
                continue
            
            # 收集该类别的所有pkl文件
            class_files = [
                os.path.join(class_dir, f) 
                for f in os.listdir(class_dir) 
                if f.endswith('.pkl')
            ]
            
            if len(class_files) < self.shots_per_class:
                print(f"警告: 类别{class_id}样本不足，需要{self.shots_per_class}个，只有{len(class_files)}个")
            
            self.support_files_by_class[class_id] = class_files
            if is_main_process():
                print(f"类别{class_id}: 找到{len(class_files)}个支持样本")
    
    def _process_support_sequence(self, raw_data: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理支持集序列：补齐到目标长度并生成mask
        
        Returns:
            data: (target_length,) 补齐后的序列
            mask: (target_length,) 有效数据mask，1表示有效，0表示填充
        """
        original_length = len(raw_data)
        
        if original_length >= self.target_length:
            # 截断
            data = raw_data[:self.target_length]
            mask = torch.ones(self.target_length, dtype=torch.bool)
        else:
            # 填充0
            data = raw_data + [0] * (self.target_length - original_length)
            mask = torch.zeros(self.target_length, dtype=torch.bool)
            mask[:original_length] = True
        
        return torch.tensor(data, dtype=torch.float32), mask
    
    def _prepare_all_support_data(self):
        """预生成所有类别的支持集数据（固定采样模式）"""
        self.all_support_data = []
        self.all_support_masks = []
        self.class_order = []  # 记录类别顺序，确保索引对应
        
        for class_id in sorted(self.activated_classes):  # 排序确保一致性
            if class_id not in self.support_files_by_class:
                # 如果某个类别没有数据，创建零向量
                print(f"警告: 类别{class_id}没有支持样本")
                exit(0)
            
            class_files = self.support_files_by_class[class_id]
            
            for shot_idx in range(self.shots_per_class):
                # 固定选择文件（循环选择）
                if len(class_files) > 0:
                    file_idx = shot_idx % len(class_files)
                    file_path = class_files[file_idx]
                    
                    # 加载数据
                    data, mask = self._load_and_process_sample(file_path)
                else:
                    data = torch.zeros(self.target_length, dtype=torch.float32)
                    mask = torch.zeros(self.target_length, dtype=torch.bool)
                
                self.all_support_data.append(data)
                self.all_support_masks.append(mask)
                self.class_order.append(class_id)
        
        # 转换为tensor
        # shape: (num_classes * shots_per_class, target_length)
        self.support_data_tensor = torch.stack(self.all_support_data)
        self.support_masks_tensor = torch.stack(self.all_support_masks)
        if is_main_process():
            print(f"支持集数据准备完成 (固定采样模式):")
            print(f"  - 支持集形状: {self.support_data_tensor.shape}")
            print(f"  - 掩码形状: {self.support_masks_tensor.shape}")
            print(f"  - 类别顺序: {self.class_order}")
    
    def _load_and_process_sample(self, file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        加载并处理单个样本文件
        
        Args:
            file_path: 样本文件路径
            
        Returns:
            data: (target_length,) 处理后的序列
            mask: (target_length,) 有效数据mask
        """
        try:
            with open(file_path, 'rb') as f:
                sample_data = pickle.load(f)
            
            # 处理不同的数据格式
            if isinstance(sample_data, dict) :
                if 'data' in sample_data:
                    raw_data = sample_data['data']
                else: 
                    raw_data = sample_data['direction']
            elif isinstance(sample_data, (list, np.ndarray)):
                raw_data = sample_data
            else:
                # 其他格式直接使用
                raw_data = sample_data
            
            # 确保raw_data是列表格式
            if isinstance(raw_data, np.ndarray):
                raw_data = raw_data.tolist()
            elif not isinstance(raw_data, list):
                raw_data = [raw_data]
            
            # 处理序列和mask
            data, mask = self._process_support_sequence(raw_data)
            return data, mask
            
        except Exception as e:
            print(f"警告: 加载文件{file_path}失败: {e}，使用零向量")
            data = torch.zeros(self.target_length, dtype=torch.float32)
            mask = torch.zeros(self.target_length, dtype=torch.bool)
            return data, mask
    
    def _generate_random_support_batch(self) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        随机生成一个支持集batch（随机采样模式）
        
        Returns:
            support_data: (num_classes, shots_per_class, target_length)
            support_masks: (num_classes, shots_per_class, target_length)
            class_order: List[int] 类别顺序（保持0-59顺序）
        """
        import random
        
        batch_support_data = []
        batch_support_masks = []
        class_order = sorted(self.activated_classes)  # 保持顺序一致性
        
        for class_id in class_order:
            if class_id not in self.support_files_by_class:
                print(f"警告: 类别{class_id}没有支持样本")
                exit(0)
            
            class_files = self.support_files_by_class[class_id]
            
            for shot_idx in range(self.shots_per_class):
                if len(class_files) > 0:
                    # 随机选择文件
                    file_path = random.choice(class_files)
                    data, mask = self._load_and_process_sample(file_path)
                
                batch_support_data.append(data)
                batch_support_masks.append(mask)
        
        # 转换为tensor并重整形
        num_classes = len(class_order)
        support_data_tensor = torch.stack(batch_support_data)
        support_masks_tensor = torch.stack(batch_support_masks)
        
        # 重整形为 (num_classes, shots_per_class, target_length)
        support_data = support_data_tensor.view(
            num_classes, self.shots_per_class, self.target_length
        )
        support_masks = support_masks_tensor.view(
            num_classes, self.shots_per_class, self.target_length
        )
        
        return support_data, support_masks, class_order

    def get_all_support_data(self) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        获取所有类别的支持集数据
        
        Returns:
            support_data: (num_classes, shots_per_class, target_length)
            support_masks: (num_classes, shots_per_class, target_length)  
            class_order: List[int] 类别顺序
        """
        if self.random_sampling:
            # 随机采样模式：每次调用都生成新的随机样本
            return self._generate_random_support_batch()
        else:
            # 固定采样模式：返回预生成的数据
            num_classes = len(self.activated_classes)
            
            # 重整形为 (num_classes, shots_per_class, target_length)
            support_data = self.support_data_tensor.view(
                num_classes, self.shots_per_class, self.target_length
            )
            support_masks = self.support_masks_tensor.view(
                num_classes, self.shots_per_class, self.target_length
            )
            
            return support_data, support_masks, self.activated_classes
    
    def __len__(self) -> int:
        return len(self.all_support_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        获取单个支持样本（通常不直接使用，主要用get_all_support_data）
        """
        return (
            self.all_support_data[idx],
            self.all_support_masks[idx], 
            self.class_order[idx]
        )

