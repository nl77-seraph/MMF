"""
模拟数据生成器 - 用于生成WebsiteFingerprinting的Tor流量序列数据
支持多标签查询集和单标签支持集的生成
"""

import numpy as np
import torch
import random
from typing import Tuple, List, Dict, Optional

class TorTrafficGenerator:
    """Tor流量序列模拟数据生成器"""
    
    def __init__(self, 
                 query_length: int = 30000,
                 support_max_length: int = 10000,
                 num_classes: int = 3,
                 seed: int = 42):
        """
        初始化数据生成器
        
        Args:
            query_length: 查询集序列长度(多标签场景)
            support_max_length: 支持集序列最大长度(单标签场景)
            num_classes: 类别数(标签页数)
            seed: 随机种子
        """
        self.query_length = query_length
        self.support_max_length = support_max_length
        self.num_classes = num_classes
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 为每个类别定义特征模式(模拟不同网站的流量特征)
        self.class_patterns = self._generate_class_patterns()
        
    def _generate_class_patterns(self) -> Dict[int, Dict]:
        """为每个类别生成特征模式"""
        patterns = {}
        
        for class_id in range(self.num_classes):
            # 每个类别有不同的流量特征
            patterns[class_id] = {
                # 流量突发模式参数
                'burst_prob': 0.1 + 0.1 * class_id,  # 突发概率
                'burst_length': 50 + 20 * class_id,   # 突发长度
                'incoming_prob': 0.4 + 0.1 * class_id,  # 入向流量概率
                # 周期性模式参数
                'period': 500 + 100 * class_id,       # 周期长度
                'amplitude': 0.2 + 0.1 * class_id,    # 周期幅度
            }
            
        return patterns
    
    def _generate_single_class_sequence(self, 
                                      class_id: int, 
                                      length: int,
                                      strength: float = 1.0) -> np.ndarray:
        """
        生成单个类别的流量序列
        
        Args:
            class_id: 类别ID
            length: 序列长度
            strength: 类别特征强度(用于多标签混合)
            
        Returns:
            shape为(length,)的流量序列
        """
        pattern = self.class_patterns[class_id]
        sequence = np.zeros(length)
        
        # 基础随机流量
        base_traffic = np.random.choice([-1, 1], size=length, 
                                      p=[1-pattern['incoming_prob'], pattern['incoming_prob']])
        
        # 添加类别特定的突发模式
        burst_positions = np.random.random(length) < pattern['burst_prob']
        for i in range(length):
            if burst_positions[i]:
                burst_end = min(i + pattern['burst_length'], length)
                # 突发期间流量方向更集中
                burst_direction = 1 if random.random() > 0.5 else -1
                base_traffic[i:burst_end] = burst_direction
        
        # 添加周期性模式
        if pattern['period'] > 0:
            periodic_signal = pattern['amplitude'] * np.sin(2 * np.pi * np.arange(length) / pattern['period'])
            # 将周期信号转换为流量方向偏置
            periodic_bias = np.where(periodic_signal > 0, 1, -1)
            # 在部分位置应用周期偏置
            periodic_mask = np.random.random(length) < 0.3
            base_traffic[periodic_mask] = periodic_bias[periodic_mask]
        
        sequence = base_traffic * strength
        
        return sequence.astype(np.float32)
    
    def generate_query_sequence(self, class_weights: List[float]) -> np.ndarray:
        """
        生成多标签查询序列
        
        Args:
            class_weights: 每个类别的权重，长度为num_classes
            
        Returns:
            shape为(query_length,)的多标签混合序列
        """
        assert len(class_weights) == self.num_classes, "权重数量必须等于类别数"
        
        # 生成每个类别的序列
        class_sequences = []
        for class_id in range(self.num_classes):
            if class_weights[class_id] > 0:
                seq = self._generate_single_class_sequence(
                    class_id, self.query_length, strength=class_weights[class_id]
                )
                class_sequences.append(seq)
        
        # 混合多个类别的序列
        if len(class_sequences) == 0:
            # 如果所有权重都为0，生成纯随机序列
            return np.random.choice([-1, 1], size=self.query_length).astype(np.float32)
        elif len(class_sequences) == 1:
            return class_sequences[0]
        else:
            # 多个类别的加权混合
            mixed_sequence = np.zeros(self.query_length)
            for seq in class_sequences:
                mixed_sequence += seq
            
            # 将混合结果转换为[-1, 1]
            mixed_sequence = np.sign(mixed_sequence)
            # 处理零值(随机分配方向)
            zero_mask = mixed_sequence == 0
            mixed_sequence[zero_mask] = np.random.choice([-1, 1], size=np.sum(zero_mask))
            
            return mixed_sequence.astype(np.float32)
    
    def generate_support_sequence(self, class_id: int, 
                                length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成单标签支持序列
        
        Args:
            class_id: 类别ID
            length: 序列长度，如果为None则随机生成
            
        Returns:
            tuple: (padded_sequence, mask)
                - padded_sequence: shape为(query_length,)，padding到查询长度的序列
                - mask: shape为(query_length,)，标识有效数据位置的mask
        """
        if length is None:
            # 随机生成长度，确保至少有一些有效数据
            length = random.randint(1000, self.support_max_length)
        
        # 生成单类别序列
        sequence = self._generate_single_class_sequence(class_id, length)
        
        # Padding到查询长度
        padded_sequence = np.zeros(self.query_length, dtype=np.float32)
        mask = np.zeros(self.query_length, dtype=np.float32)
        
        # 将序列放在开头
        padded_sequence[:length] = sequence
        mask[:length] = 1.0  # 标记有效数据位置
        
        return padded_sequence, mask
    
    def generate_episode(self, 
                        batch_size: int = 8,
                        shots_per_class: int = 1) -> Dict[str, torch.Tensor]:
        """
        生成一个完整的episode，包含查询集和支持集
        
        Args:
            batch_size: 查询集batch大小
            shots_per_class: 每个类别的支持样本数
            
        Returns:
            dict包含:
                - query_data: (batch_size, query_length) 查询集数据
                - query_labels: (batch_size, num_classes) 查询集多标签
                - support_data: (num_classes, shots_per_class, query_length) 支持集数据
                - support_masks: (num_classes, shots_per_class, query_length) 支持集mask
        """
        # 生成查询集
        query_data = []
        query_labels = []
        
        for _ in range(batch_size):
            # 随机生成多标签权重
            weights = [random.uniform(0, 1) if random.random() > 0.3 else 0 
                      for _ in range(self.num_classes)]
            
            # 至少有一个类别权重大于0
            if sum(weights) == 0:
                weights[random.randint(0, self.num_classes-1)] = 1.0
            
            # 生成查询序列
            query_seq = self.generate_query_sequence(weights)
            query_data.append(query_seq)
            
            # 生成标签(二值化权重)
            labels = [1.0 if w > 0 else 0.0 for w in weights]
            query_labels.append(labels)
        
        # 生成支持集
        support_data = []
        support_masks = []
        
        for class_id in range(self.num_classes):
            class_support = []
            class_masks = []
            
            for _ in range(shots_per_class):
                support_seq, mask = self.generate_support_sequence(class_id)
                class_support.append(support_seq)
                class_masks.append(mask)
            
            support_data.append(class_support)
            support_masks.append(class_masks)
        
        # 转换为tensor (优化效率)
        return {
            'query_data': torch.tensor(np.array(query_data), dtype=torch.float32),
            'query_labels': torch.tensor(np.array(query_labels), dtype=torch.float32),
            'support_data': torch.tensor(np.array(support_data), dtype=torch.float32),
            'support_masks': torch.tensor(np.array(support_masks), dtype=torch.float32)
        }

def test_data_generator():
    """测试数据生成器"""
    print("测试Tor流量数据生成器...")
    
    # 创建生成器
    generator = TorTrafficGenerator(
        query_length=30000,
        support_max_length=10000,
        num_classes=3,
        seed=42
    )
    
    # 测试单个查询序列生成
    print("\n1. 测试单个查询序列生成:")
    weights = [0.5, 0.3, 0.0]  # 前两个类别混合
    query_seq = generator.generate_query_sequence(weights)
    print(f"查询序列形状: {query_seq.shape}")
    print(f"查询序列值范围: [{query_seq.min()}, {query_seq.max()}]")
    print(f"查询序列前10个值: {query_seq[:10]}")
    
    # 测试支持序列生成
    print("\n2. 测试支持序列生成:")
    support_seq, mask = generator.generate_support_sequence(class_id=0, length=5000)
    print(f"支持序列形状: {support_seq.shape}")
    print(f"支持mask形状: {mask.shape}")
    print(f"有效数据长度: {int(mask.sum())}")
    print(f"支持序列前10个值: {support_seq[:10]}")
    
    # 测试完整episode生成
    print("\n3. 测试完整episode生成:")
    episode = generator.generate_episode(batch_size=4, shots_per_class=2)
    
    for key, value in episode.items():
        print(f"{key}: {value.shape}")
    
    print(f"\n查询集标签示例:")
    print(episode['query_labels'][:2])
    
    print(f"\n支持集mask示例(第0类，第0个shot):")
    print(f"有效数据长度: {int(episode['support_masks'][0, 0].sum())}")
    
    print("\n✅ 数据生成器测试完成!")

if __name__ == "__main__":
    test_data_generator() 