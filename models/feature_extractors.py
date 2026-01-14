"""
增强的特征提取器 - 混合方案C
基于DF网络，增加SE注意力、Shot Attention等机制
目标: 将mAP从0.9+提升到0.95+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import sys
import os
from utils.misc import is_main_process
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dynamic_conv1d import FeatureReweightingModule
#os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch.distributed as dist


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (通道注意力)
    用于增强重要通道，抑制不重要通道
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, length)
        Returns:
            重加权后的特征
        """
        b, c, _ = x.size()
        # Squeeze: 全局平均池化
        y = self.squeeze(x).view(b, c)
        # Excitation: 学习通道权重
        y = self.excitation(y).view(b, c, 1)
        # Scale: 重加权
        return x * y.expand_as(x)


class ShotAttentionFusion(nn.Module):
    """
    Shot-level Attention融合模块
    对每个类别的多个shot进行加权融合，而非简单mean
    """
    
    def __init__(self, feature_dim: int):
        super(ShotAttentionFusion, self).__init__()
        # 注意力网络
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (num_classes, shots, feature_dim) 或 (num_classes, shots, channels, seq_len)
        Returns:
            融合后的特征: (num_classes, feature_dim) 或 (num_classes, channels, seq_len)
        """
        if len(x.shape) == 3:
            # Case 1: (num_classes, shots, feature_dim)
            attn_scores = self.attention(x)  # (num_classes, shots, 1)
            attn_weights = F.softmax(attn_scores, dim=1)  # 在shot维度softmax
            weighted_features = (x * attn_weights).sum(dim=1)  # (num_classes, feature_dim)
            return weighted_features
        elif len(x.shape) == 4:
            # Case 2: (num_classes, shots, channels, seq_len)
            num_classes, shots, channels, seq_len = x.shape
            # 先进行全局池化得到每个shot的特征向量
            pooled = x.mean(dim=-1)  # (num_classes, shots, channels)
            # 计算注意力权重
            attn_scores = self.attention(pooled)  # (num_classes, shots, 1)
            attn_weights = F.softmax(attn_scores, dim=1)  # (num_classes, shots, 1)
            # 扩展权重并加权融合
            attn_weights = attn_weights.unsqueeze(-1)  # (num_classes, shots, 1, 1)
            weighted_features = (x * attn_weights).sum(dim=1)  # (num_classes, channels, seq_len)
            return weighted_features
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")


class DFBlock(nn.Module):
    """DF网络的单个Block (与原版相同)"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 8, pool_size: int = 8, 
                 pool_stride: int = 4, dropout: float = 0.5,
                 activation: str = 'relu', use_se: bool = False):
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
        
        # 可选的SE Block
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels, reduction=16)
    
    def forward(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        
        # 可选的SE注意力
        if self.use_se:
            x = self.se(x)
        
        # 池化和dropout
        x = self.pool(x)
        x = self.dropout(x)
        
        return x


class DFFeatureExtractor(nn.Module):
    """基于DF网络的特征提取器 (可选增强版)"""
    
    def __init__(self, dropout: float = 0.5, use_se: bool = False):
        super(DFFeatureExtractor, self).__init__()
        
        # DF网络参数(与原始DF网络保持一致)
        self.filter_nums = [32, 64, 128, 256]
        self.kernel_size = 8
        self.pool_sizes = [8, 8, 8, 8]
        self.pool_strides = [4, 4, 4, 4]
        self.use_se = use_se
        
        # 构建4个Block
        self.block1 = DFBlock(1, self.filter_nums[0], 
                             self.kernel_size, self.pool_sizes[0], 
                             self.pool_strides[0], dropout, 'elu', use_se)
        
        self.block2 = DFBlock(self.filter_nums[0], self.filter_nums[1], 
                             self.kernel_size, self.pool_sizes[1], 
                             self.pool_strides[1], dropout, 'relu', use_se)
        
        self.block3 = DFBlock(self.filter_nums[1], self.filter_nums[2], 
                             self.kernel_size, self.pool_sizes[2], 
                             self.pool_strides[2], dropout, 'relu', use_se)
        
        self.block4 = DFBlock(self.filter_nums[2], self.filter_nums[3], 
                             self.kernel_size, self.pool_sizes[3], 
                             self.pool_strides[3], dropout, 'relu', use_se)
        
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


class EnhancedMetaLearnet(nn.Module):
    """
    增强的元学习网络 (混合方案C)
    
    改进点:
    1. 使用与DF相同的4层结构提取特征
    2. Shot-level Attention融合（替代简单mean）
    3. SE通道注意力机制
    4. 多层MLP权重生成器（替代单层Linear）
    """
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.5):
        super(EnhancedMetaLearnet, self).__init__()
        
        # 步骤1: 特征提取网络（与DF相同结构）
        self.filter_nums = [32, 64, 128, 256]
        self.kernel_size = 8
        self.pool_sizes = [8, 8, 8, 8]
        self.pool_strides = [4, 4, 4, 4]

        # 4个DFBlock
        self.block1 = DFBlock(in_channels, self.filter_nums[0], 
                             self.kernel_size, self.pool_sizes[0], 
                             self.pool_strides[0], dropout, 'elu', use_se=False)
        
        self.block2 = DFBlock(self.filter_nums[0], self.filter_nums[1], 
                             self.kernel_size, self.pool_sizes[1], 
                             self.pool_strides[1], dropout, 'relu', use_se=False)
        
        self.block3 = DFBlock(self.filter_nums[1], self.filter_nums[2], 
                             self.kernel_size, self.pool_sizes[2], 
                             self.pool_strides[2], dropout, 'relu', use_se=False)
        
        self.block4 = DFBlock(self.filter_nums[2], self.filter_nums[3], 
                             self.kernel_size, self.pool_sizes[3], 
                             self.pool_strides[3], dropout, 'relu', use_se=False)
        
        # 步骤2: Shot-level Attention融合
        self.shot_attention = ShotAttentionFusion(self.filter_nums[3])
        
        # 步骤3: SE通道注意力
        self.channel_attention = SEBlock(self.filter_nums[3], reduction=16)
        
        # 步骤4: 多层权重生成器
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.weight_generator = nn.Sequential(
            nn.Linear(self.filter_nums[3], self.filter_nums[3] * 2),
            nn.LayerNorm(self.filter_nums[3] * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.filter_nums[3] * 2, out_channels),
            nn.LayerNorm(out_channels)
        )
        
        # 残差连接（如果输入输出维度匹配）
        self.use_residual = (self.filter_nums[3] == out_channels)
        if not self.use_residual:
            self.residual_proj = nn.Linear(self.filter_nums[3], out_channels)
    
    def forward(self, x):
        """
        生成动态权重
        
        Args:
            x: 输入特征 (num_classes*shots, 2, length)
               或 (num_classes, shots, 2, length) 如果已经reshape
               其中通道0是数据，通道1是mask
            
        Returns:
            动态权重 (num_classes, out_channels)
        """
        # 处理输入维度
        if len(x.shape) == 4:
            # (num_classes, shots, 2, length)
            num_classes, shots_per_class, _, length = x.shape
            x_reshaped = x.view(num_classes * shots_per_class, 2, length)
        else:
            # (num_classes*shots, 2, length) - 需要从外部获知num_classes和shots
            # 这种情况下需要在调用时处理，这里暂时不支持
            raise ValueError("Input must be (num_classes, shots, 2, length)")
        
        # 步骤1: 特征提取（通过4个DFBlock）
        features = self.block1(x_reshaped)
        features = self.block2(features)
        features = self.block3(features)
        features = self.block4(features)
        # features: (num_classes*shots, 256, seq_len)
        
        # 步骤2: 应用通道注意力
        features = self.channel_attention(features)
        # features: (num_classes*shots, 256, seq_len)
        
        # 重整形回(num_classes, shots, 256, seq_len)
        _, channels, seq_len = features.shape
        features = features.view(num_classes, shots_per_class, channels, seq_len)
        
        # 步骤3: Shot-level Attention融合
        if shots_per_class > 1:
            features = self.shot_attention(features)  # (num_classes, 256, seq_len)
        else:
            features = features.squeeze(1)  # (num_classes, 256, seq_len)
        
        # 步骤4: 全局池化
        pooled_features = self.global_pool(features).squeeze(-1)  # (num_classes, 256)
        
        # 步骤5: 多层权重生成
        dynamic_weights = self.weight_generator(pooled_features)  # (num_classes, out_channels)
        
        # 步骤6: 残差连接（如果维度匹配）
        if self.use_residual:
            dynamic_weights = dynamic_weights + pooled_features
        elif hasattr(self, 'residual_proj'):
            dynamic_weights = dynamic_weights + self.residual_proj(pooled_features)
        
        return dynamic_weights


class EnhancedMultiMetaFingerNet(nn.Module):
    """
    增强的多标签元指纹识别网络 (混合方案C)
    
    改进:
    1. 使用EnhancedMetaLearnet
    2. 使用EnhancedClassificationHead (在classification_head_enhanced.py中)
    3. 保持1×1动态卷积不变
    """
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.5, 
                 support_blocks: int = 0, 
                 unified_threshold: float = 0.4, use_se_in_df: bool = False):
        super(EnhancedMultiMetaFingerNet, self).__init__()
        
        self.num_classes = num_classes
        self.support_blocks = support_blocks
        
        # 主特征提取网络(DF网络) - 可选添加SE
        self.feature_extractor = DFFeatureExtractor(dropout, use_se=use_se_in_df)
        
        # 计算中间特征维度
        self.support_feature_dim = self.feature_extractor.filter_nums[support_blocks - 1] if support_blocks > 0 else 128
        self.query_feature_dim = self.feature_extractor.filter_nums[-1]  # 256
        
        # 增强的元学习网络
        self.meta_learnet = EnhancedMetaLearnet(
            in_channels=2,  # 数据+mask
            out_channels=self.query_feature_dim,
            dropout=dropout
        )
        
        # 特征重加权模块（1D动态卷积）- 保持不变
        self.feature_reweighting = FeatureReweightingModule(
            feature_dim=self.query_feature_dim,
            kernel_size=1
        )
        
        # 增强的分类头
        from classification_head_enhanced import EnhancedClassificationHead
        self.classification_head = EnhancedClassificationHead(
            feature_dim=self.query_feature_dim,
            num_classes=num_classes,
            seq_len=80,
            num_topm_layers=3,  # 简化TopM，减少到2层
            num_cross_layers=3   # Cross-Class Attention层数
        )
        if is_main_process():
            print(f"增强网络初始化完成:")
            print(f"  - 类别数: {num_classes}")
            print(f"  - DF使用SE: {use_se_in_df}")
            print(f"  - 查询集特征维度: {self.query_feature_dim}")
            print(f"  - Meta学习网络: Enhanced (Shot Attention + SE + Deep MLP)")

    def query_forward(self, x):
        """查询集前向传播"""
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
        
        # 如果没有提供mask，则创建一个全1的mask
        if mask is None:
            mask = torch.ones_like(x)
        
        # 将数据和mask在通道维度上拼接
        # (num_classes, shots, length) + (num_classes, shots, length)
        # → (num_classes, shots, 2, length)
        support_input = torch.stack([x, mask], dim=2)
        
        # 通过增强的meta_learnet生成动态权重
        dynamic_weights = self.meta_learnet(support_input)
        # dynamic_weights: (num_classes, query_feature_dim)
        
        return dynamic_weights
    
    def fusion_forward(self, query_features, dynamic_weights):
        """特征融合前向传播"""
        return self.feature_reweighting(query_features, dynamic_weights)
    
    def classification_forward(self, reweighted_features):
        """分类前向传播"""
        logits = self.classification_head(reweighted_features)
        return logits
    
    def forward(self, query_data, support_data, support_masks=None):
        """
        完整前向传播
        
        Args:
            query_data: (batch, length)
            support_data: (num_classes, shots, length)
            support_masks: (num_classes, shots, length)
            
        Returns:
            dict包含查询特征、动态权重、融合特征和分类结果
        """
        # 查询集特征提取
        query_features = self.query_forward(query_data)
        
        # 支持集动态权重生成 (使用增强的MetaLearnet)
        dynamic_weights = self.support_forward(support_data, support_masks)
        
        # 特征融合（1D动态卷积）
        reweighted_features = self.fusion_forward(query_features, dynamic_weights)
        
        # 多标签分类
        logits = self.classification_forward(reweighted_features)
        
        return {
            'query_features': query_features,
            'dynamic_weights': dynamic_weights,
            'reweighted_features': reweighted_features,
            'logits': logits
        }


def test_enhanced_network():
    """测试增强网络"""
    print("="*60)
    print("测试增强的Multi-Meta-Finger网络")
    print("="*60)
    
    # 设置参数
    batch_size = 4
    num_classes = 3
    shots_per_class = 2
    query_length = 30000
    
    # 创建增强网络
    net = EnhancedMultiMetaFingerNet(
        num_classes=num_classes, 
        dropout=0.5, 
        support_blocks=0,
        use_se_in_df=False  # 先不用SE，测试基础版本
    )
    
    # 创建模拟数据
    query_data = torch.randn(batch_size, query_length)
    support_data = torch.randn(num_classes, shots_per_class, query_length)
    support_masks = torch.ones(num_classes, shots_per_class, query_length)
    
    print(f"\n输入形状:")
    print(f"  查询集: {query_data.shape}")
    print(f"  支持集: {support_data.shape}")
    print(f"  支持集mask: {support_masks.shape}")
    
    # 前向传播
    print(f"\n执行前向传播...")
    with torch.no_grad():
        results = net(query_data, support_data, support_masks)
    
    print(f"\n输出形状:")
    print(f"  查询集特征: {results['query_features'].shape}")
    print(f"  动态权重: {results['dynamic_weights'].shape}")
    print(f"  重加权特征: {results['reweighted_features'].shape}")
    print(f"  分类logits: {results['logits'].shape}")
    print(f"  预测结果: {results['predictions'].shape}")
    print(f"  类别概率: {results['probabilities'].shape}")
    
    # 验证输出维度
    assert results['query_features'].shape == (batch_size, 60, 256)
    assert results['dynamic_weights'].shape == (num_classes, 256)
    assert results['reweighted_features'].shape == (batch_size * num_classes, 256, 60)
    assert results['logits'].shape == (batch_size, num_classes)
    
    print(f"\n✅ 增强网络测试完成!")
    print(f"\n改进点:")
    print(f"  ✓ Shot Attention融合")
    print(f"  ✓ SE通道注意力")
    print(f"  ✓ 深层MLP权重生成")
    print(f"  ✓ 残差连接")


if __name__ == "__main__":
    test_enhanced_network()

