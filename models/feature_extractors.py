"""
基于DF网络的特征提取器
参考Few-shot Detection项目设计，实现查询集和支持集的特征提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dynamic_conv1d import FeatureReweightingModule
from classification_head import MultiLabelClassificationHead

class DFBlock(nn.Module):
    """DF网络的单个Block"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 8, pool_size: int = 8, 
                 pool_stride: int = 4, dropout: float = 0.5,
                 activation: str = 'relu'):
        super(DFBlock, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1, 
                              padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv1d(in_channels=out_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size,
                              stride=1, 
                              padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        
        # 池化和dropout
        self.pool = nn.MaxPool1d(kernel_size=pool_size, 
                                stride=pool_stride, 
                                padding=pool_size // 2)
        self.dropout = nn.Dropout(p=dropout)
        
        # 激活函数
        if activation == 'elu':
            self.activation1 = nn.ELU(alpha=1.0)
            self.activation2 = nn.ELU(alpha=1.0)
        else:
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
    
    def forward(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        
        # 池化和dropout
        x = self.pool(x)
        x = self.dropout(x)
        
        return x

class DFFeatureExtractor(nn.Module):
    """基于DF网络的完整特征提取器"""
    
    def __init__(self, dropout: float = 0.5):
        super(DFFeatureExtractor, self).__init__()
        
        # DF网络参数(与原始DF网络保持一致)
        self.filter_nums = [32, 64, 128, 256]
        self.kernel_size = 8
        self.pool_sizes = [8, 8, 8, 8]
        self.pool_strides = [4, 4, 4, 4]
        
        # 构建4个Block
        self.block1 = DFBlock(1, self.filter_nums[0], 
                             self.kernel_size, self.pool_sizes[0], 
                             self.pool_strides[0], dropout, 'elu')
        
        self.block2 = DFBlock(self.filter_nums[0], self.filter_nums[1], 
                             self.kernel_size, self.pool_sizes[1], 
                             self.pool_strides[1], dropout, 'relu')
        
        self.block3 = DFBlock(self.filter_nums[1], self.filter_nums[2], 
                             self.kernel_size, self.pool_sizes[2], 
                             self.pool_strides[2], dropout, 'relu')
        
        self.block4 = DFBlock(self.filter_nums[2], self.filter_nums[3], 
                             self.kernel_size, self.pool_sizes[3], 
                             self.pool_strides[3], dropout, 'relu')
        
        self.blocks = [self.block1, self.block2, self.block3, self.block4]
    
    def forward(self, x, num_blocks: Optional[int] = None):
        """
        前向传播
        
        Args:
            x: 输入tensor, shape=(batch, length)
            num_blocks: 使用的block数量，如果为None则使用全部
            
        Returns:
            特征tensor
        """
        # 确保输入是3D: (batch, 1, length)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # 执行指定数量的blocks
        num_blocks = num_blocks if num_blocks is not None else len(self.blocks)
        
        for i in range(num_blocks):
            x = self.blocks[i](x)
        
        return x
    
    def forward_partial(self, x, num_blocks: int):
        """前向传播部分网络(用于支持集)"""
        return self.forward(x, num_blocks)
    
    def forward_full(self, x):
        """前向传播完整网络(用于查询集)"""
        x = self.forward(x, None)
        # 转置以匹配原始DF网络输出格式: (batch, length, channels)
        return x.transpose(1, 2)

class MetaLearnet(nn.Module):
    """
    元学习网络，用于生成动态权重。
    参考Few-shot Detection的learnet设计，并借鉴DFNet结构进行强化。
    该网络直接处理支持集样本（序列+mask），生成用于重加权的动态权重。
    """
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.5):
        """
        Args:
            in_channels (int): 输入通道数，根据设计应为2 (序列+mask)。
            out_channels (int): 输出通道数，应与查询集特征维度匹配。
            dropout (float): dropout比例。
        """
        super(MetaLearnet, self).__init__()
        
        # 借鉴DFNet的设计，构建一个强大的特征提取器
        self.filter_nums = [32, 64, 128, 256]
        self.kernel_size = 8
        self.pool_sizes = [8, 8, 8, 8]
        self.pool_strides = [4, 4, 4, 4]

        # Block 1
        self.block1 = DFBlock(in_channels, self.filter_nums[0], 
                             self.kernel_size, self.pool_sizes[0], 
                             self.pool_strides[0], dropout, 'elu')
        
        # Block 2
        self.block2 = DFBlock(self.filter_nums[0], self.filter_nums[1], 
                             self.kernel_size, self.pool_sizes[1], 
                             self.pool_strides[1], dropout, 'relu')
        
        # Block 3
        self.block3 = DFBlock(self.filter_nums[1], self.filter_nums[2], 
                             self.kernel_size, self.pool_sizes[2], 
                             self.pool_strides[2], dropout, 'relu')
        
        # Block 4
        self.block4 = DFBlock(self.filter_nums[2], self.filter_nums[3], 
                             self.kernel_size, self.pool_sizes[3], 
                             self.pool_strides[3], dropout, 'relu')
        
        # 全局平均池化，将时序维度压缩为1
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 最后的线性层，用于生成最终的权重
        self.weight_generator = nn.Linear(self.filter_nums[3], out_channels)

    def forward(self, x):
        """
        生成动态权重
        
        Args:
            x: 输入特征 (batch, 2, length)，其中通道0是数据，通道1是mask
            
        Returns:
            动态权重 (batch, out_channels)
        """
        # 依次通过4个DFBlock
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # 全局池化
        x = self.global_pool(x)  # (batch, channels, 1)
        
        # 展平
        x = x.squeeze(2)  # (batch, channels)
        
        # 生成最终权重
        dynamic_weights = self.weight_generator(x)  # (batch, out_channels)
        
        return dynamic_weights

class MultiMetaFingerNet(nn.Module):
    """
    多标签元指纹识别网络
    参考darknet_meta.py的设计思路
    """
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.5, 
                 support_blocks: int = 3, classification_method: str = 'binary',
                 unified_threshold: float = 0.4):
        super(MultiMetaFingerNet, self).__init__()
        
        self.num_classes = num_classes
        self.support_blocks = support_blocks
        self.classification_method = classification_method
        
        # 主特征提取网络(DF网络)
        self.feature_extractor = DFFeatureExtractor(dropout)
        
        # 计算中间特征维度
        # 经过3个block后的通道数
        self.support_feature_dim = self.feature_extractor.filter_nums[support_blocks - 1]  # 128
        # 完整网络输出的通道数
        self.query_feature_dim = self.feature_extractor.filter_nums[-1]  # 256
        
        # 元学习网络(learnet)
        # 输入通道为2（数据+mask），输出为查询特征维度
        self.meta_learnet = MetaLearnet(
            in_channels=2, 
            out_channels=self.query_feature_dim,
            dropout=dropout
        )
        
        # 特征重加权模块（1D动态卷积）
        self.feature_reweighting = FeatureReweightingModule(
            feature_dim=self.query_feature_dim,
            kernel_size=1  # 使用kernel_size=1进行点卷积
        )
        
        # 多标签分类头
        self.classification_head = MultiLabelClassificationHead(
            feature_dim=self.query_feature_dim,
            num_classes=num_classes,
            seq_len=119,  # 查询集特征的序列长度
            classification_method=classification_method,
            unified_threshold=unified_threshold
        )
        
        print(f"网络初始化完成:")
        print(f"  - 类别数: {num_classes}")
        print(f"  - 支持集使用前{support_blocks}个Block")
        print(f"  - 支持集特征维度: {self.support_feature_dim}")
        print(f"  - 查询集特征维度: {self.query_feature_dim}")
        print(f"  - 分类方法: {classification_method}")

    def query_forward(self, x):
        """
        查询集前向传播
        
        Args:
            x: 查询集数据 (batch, length)
            
        Returns:
            查询集特征 (batch, seq_len, query_feature_dim)
        """
        return self.feature_extractor.forward_full(x)
    
    def support_forward(self, x, mask=None):
        """
        支持集前向传播，生成动态权重
        
        Args:
            x: 支持集数据 (num_classes, shots, length)
            mask: 有效数据mask (num_classes, shots, length)
            
        Returns:
            动态权重 (num_classes, query_feature_dim)
        """
        num_classes, shots_per_class, length = x.shape
        
        # 重整形为 (num_classes*shots, length)
        x_reshaped = x.view(-1, length)
        
        # 如果没有提供mask，则创建一个全1的mask
        if mask is None:
            mask = torch.ones_like(x)
        mask_reshaped = mask.view(-1, length)
            
        # 将数据和mask在通道维度上拼接
        # unsqueeze(1) 增加一个通道维度
        support_input = torch.stack([x_reshaped, mask_reshaped], dim=1)
        # support_input shape: (num_classes*shots, 2, length)

        # 直接通过meta_learnet生成动态权重
        dynamic_weights = self.meta_learnet(support_input)
        # dynamic_weights: (num_classes*shots, query_feature_dim)
        
        # 重整形回 (num_classes, shots, query_feature_dim)
        dynamic_weights = dynamic_weights.view(num_classes, shots_per_class, -1)
        
        # 如果每个类别有多个shot，对权重进行平均
        if shots_per_class > 1:
            dynamic_weights = dynamic_weights.mean(dim=1)  # (num_classes, query_feature_dim)
        else:
            dynamic_weights = dynamic_weights.squeeze(1)  # (num_classes, query_feature_dim)
        
        return dynamic_weights
    
    def fusion_forward(self, query_features, dynamic_weights):
        """
        特征融合前向传播，使用1D动态卷积进行重加权
        
        Args:
            query_features: (batch, seq_len, feature_dim) 查询集特征
            dynamic_weights: (num_classes, feature_dim) 动态权重
            
        Returns:
            reweighted_features: (batch*num_classes, feature_dim, seq_len) 重加权特征
        """
        return self.feature_reweighting(query_features, dynamic_weights)
    
    def classification_forward(self, reweighted_features):
        """
        分类前向传播，使用TopM_MHSA注意力进行多标签分类
        
        Args:
            reweighted_features: (batch*num_classes, feature_dim, seq_len) 重加权特征
            
        Returns:
            logits: (batch, num_classes) 分类logits
            predictions: (batch, num_classes) 二值化预测
            probabilities: (batch, num_classes) 类别概率
        """
        logits = self.classification_head(reweighted_features)
        predictions, probabilities = self.classification_head.predict(reweighted_features)
        
        return logits, predictions, probabilities
    
    def forward(self, query_data, support_data, support_masks=None):
        """
        完整前向传播，包含特征融合和多标签分类
        
        Args:
            query_data: (batch, length)
            support_data: (num_classes, shots, length)
            support_masks: (num_classes, shots, length)
            
        Returns:
            dict包含查询特征、动态权重、融合特征和分类结果
        """
        # 查询集特征提取
        query_features = self.query_forward(query_data)
        
        # 支持集动态权重生成
        dynamic_weights = self.support_forward(support_data, support_masks)
        
        # 特征融合（1D动态卷积）
        reweighted_features = self.fusion_forward(query_features, dynamic_weights)
        
        # 多标签分类
        logits, predictions, probabilities = self.classification_forward(reweighted_features)
        
        return {
            'query_features': query_features,
            'dynamic_weights': dynamic_weights,
            'reweighted_features': reweighted_features,
            'logits': logits,
            'predictions': predictions,
            'probabilities': probabilities
        }

def test_feature_extractors():
    """测试特征提取器"""
    print("测试基于DF网络的特征提取器...")
    
    # 设置参数
    batch_size = 4
    num_classes = 3
    shots_per_class = 2
    query_length = 30000
    
    # 创建网络
    net = MultiMetaFingerNet(num_classes=num_classes, dropout=0.5, support_blocks=0)
    
    # 创建模拟数据
    query_data = torch.randn(batch_size, query_length)
    support_data = torch.randn(num_classes, shots_per_class, query_length)
    support_masks = torch.ones(num_classes, shots_per_class, query_length)
    
    # 模拟部分padding的情况
    for i in range(num_classes):
        for j in range(shots_per_class):
            # 随机设置有效长度
            valid_length = torch.randint(5000, 15000, (1,)).item()
            support_masks[i, j, valid_length:] = 0
    
    # 测试无mask的情况
    print("\n--- 测试无mask的场景 ---")
    with torch.no_grad():
        results_no_mask = net(query_data, support_data, None)
    print(f"  查询集特征 (无mask): {results_no_mask['query_features'].shape}")
    print(f"  动态权重 (无mask): {results_no_mask['dynamic_weights'].shape}")
    print(f"  分类logits (无mask): {results_no_mask['logits'].shape}")
    print(f"  预测结果 (无mask): {results_no_mask['predictions'].shape}")

    print(f"\n输入形状:")
    print(f"  查询集: {query_data.shape}")
    print(f"  支持集: {support_data.shape}")
    print(f"  支持集mask: {support_masks.shape}")
    
    # 前向传播
    with torch.no_grad():
        results = net(query_data, support_data, support_masks)
    
    print(f"\n输出形状:")
    print(f"  查询集特征: {results['query_features'].shape}")
    print(f"  动态权重: {results['dynamic_weights'].shape}")
    print(f"  重加权特征: {results['reweighted_features'].shape}")
    print(f"  分类logits: {results['logits'].shape}")
    print(f"  预测结果: {results['predictions'].shape}")
    print(f"  类别概率: {results['probabilities'].shape}")
    
    # 验证输出维度是否正确
    assert results['query_features'].shape == (batch_size, 119, net.query_feature_dim)
    assert results['dynamic_weights'].shape == (num_classes, net.query_feature_dim)
    assert results_no_mask['dynamic_weights'].shape == (num_classes, net.query_feature_dim)
    
    # 验证融合特征的维度
    expected_reweighted_shape = (batch_size * num_classes, net.query_feature_dim, 119)
    assert results['reweighted_features'].shape == expected_reweighted_shape, \
        f"重加权特征形状不匹配: {results['reweighted_features'].shape} != {expected_reweighted_shape}"
    
    # 验证分类结果的维度
    expected_classification_shape = (batch_size, num_classes)
    assert results['logits'].shape == expected_classification_shape, \
        f"分类logits形状不匹配: {results['logits'].shape} != {expected_classification_shape}"
    assert results['predictions'].shape == expected_classification_shape, \
        f"预测结果形状不匹配: {results['predictions'].shape} != {expected_classification_shape}"
    assert results['probabilities'].shape == expected_classification_shape, \
        f"类别概率形状不匹配: {results['probabilities'].shape} != {expected_classification_shape}"
    
    # 显示示例分类结果
    print(f"\n示例分类结果 (第一个样本):")
    print(f"  Logits: {results['logits'][0].cpu().numpy()}")
    print(f"  概率: {results['probabilities'][0].cpu().numpy()}")
    print(f"  预测: {results['predictions'][0].cpu().numpy()}")
    
    # 测试单独的模块
    print(f"\n测试单独模块:")
    
    # 测试查询集特征提取
    query_features = net.query_forward(query_data)
    print(f"  查询集特征 (单独): {query_features.shape}")
    
    # 测试支持集权重生成
    dynamic_weights = net.support_forward(support_data, support_masks)
    print(f"  动态权重 (单独): {dynamic_weights.shape}")
    
    # 测试特征融合
    reweighted_features = net.fusion_forward(query_features, dynamic_weights)
    print(f"  重加权特征 (单独): {reweighted_features.shape}")
    
    # 测试分类
    logits, predictions, probabilities = net.classification_forward(reweighted_features)
    print(f"  分类logits (单独): {logits.shape}")
    print(f"  预测结果 (单独): {predictions.shape}")
    print(f"  类别概率 (单独): {probabilities.shape}")
    
    print(f"\n✅ 特征提取器测试完成!")

if __name__ == "__main__":
    test_feature_extractors() 