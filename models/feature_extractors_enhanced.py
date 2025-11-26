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
import random
from einops.layers.torch import Rearrange
import numpy as np
from utils.misc import is_main_process
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dynamic_conv1d import FeatureReweightingModule
#os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch.distributed as dist

class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(ConvBlock1d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,dilation=dilation, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,kernel_size=kernel_size,dilation=dilation, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        self.last_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.last_relu(out + res)


class LocalProfiling(nn.Module):
    """ Local Profiling module in ARES """
    def __init__(self):
        super(LocalProfiling, self).__init__()
        
        self.net = nn.Sequential(
            ConvBlock1d(in_channels=1, out_channels=32, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=32, out_channels=64, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=64, out_channels=128, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=128, out_channels=256, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class ARESBackbone(nn.Module):
    """
    集成 ARES 的分段处理流程：
    Roll -> Dividing -> LocalProfiling(CNN) -> Combination
    """
    def __init__(self, in_channels=1, num_segments=4, roll_max=2500):
        super(ARESBackbone, self).__init__()
        
        self.num_segments = num_segments
        self.roll_max = roll_max
        
        # 1. 分段: (Batch, Channel, Segments*P) -> (Batch*Segments, Channel, P)
        self.dividing = Rearrange('b c (n p) -> (b n) c p', n=num_segments)
        
        # 2. 局部特征提取 (CNN)
        self.profiling = LocalProfiling(in_channels=in_channels)
        
        # 3. 合并: (Batch*Segments, Channel, P_out) -> (Batch, Channel, Segments*P_out)
        self.combination = Rearrange('(b n) c p -> b c (n p)', n=num_segments)
        
        self.out_channels = self.profiling.out_channels # 256

    def forward(self, x):
        # 1. 随机平移增强 (仅训练阶段启用)
        # 这对 Multi-tab 场景非常重要，模拟截取位置的随机性
        if self.training and self.roll_max > 0:
            shift = np.random.randint(0, 1 + self.roll_max)
            x = torch.roll(x, shifts=shift, dims=-1)
        
        # 2. 分段
        # 注意：输入长度必须能被 num_segments 整除，否则 Rearrange 会报错
        # 通常 20000 / 4 = 5000，是可以整除的
        x = self.dividing(x)
        
        # 3. 提取特征
        x = self.profiling(x)
        
        # 4. 合并回序列
        x = self.combination(x)
        
        return x

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

class EnhancedMetaLearnet(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.5):
        super(EnhancedMetaLearnet, self).__init__()
        self.backbone = ARESBackbone(in_channels=in_channels, num_segments=4, roll_max=2500)

        self.shot_attention = ShotAttentionFusion(self.filter_nums[3])
        backbone_out_dim = self.backbone.out_channels
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # 权重生成器 MLP
        self.weight_generator = nn.Sequential(
            nn.Linear(backbone_out_dim, backbone_out_dim * 2),
            nn.LayerNorm(backbone_out_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(backbone_out_dim * 2, out_channels),
            nn.LayerNorm(out_channels)
        )
    
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
            num_classes, shots, _, length = x.shape
            x_reshaped = x.view(num_classes * shots, 2, length)
        else:
            # (num_classes*shots, 2, length) - 需要从外部获知num_classes和shots
            # 这种情况下需要在调用时处理，这里暂时不支持
            raise ValueError("Input must be (num_classes, shots, 2, length)")
        
        
        features = self.backbone(x_reshaped)
        pooled = self.global_pool(features).squeeze(-1)
        pooled = pooled.view(num_classes, shots, -1)
        pooled = pooled.mean(dim=1) # (Classes, 256)
        weights = self.weight_generator(pooled)
        return weights


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
        self.feature_extractor = ARESBackbone(in_channels=1, num_segments=4, roll_max=2500)
        
        self.query_feature_dim = self.feature_extractor.out_channels # 256
        
        # 2. Meta Learner (使用 ARES 结构)
        # Support 包含 Mask，in_channels=2
        self.meta_learnet = EnhancedMetaLearnet(
            in_channels=2,
            out_channels=self.query_feature_dim,
            dropout=dropout
        )
        
        # 3. 特征重加权 (保持不变)
        self.feature_reweighting = FeatureReweightingModule(
            feature_dim=self.query_feature_dim,
            kernel_size=1
        )
        
        # 4. 分类头 (保持不变)
        # 这里的 seq_len 需要大致匹配 ARES 的输出长度
        # ARES 下采样率 256, 输入 20000 -> 输出 78左右
        # 你的 classification_head_enhanced 默认 seq_len 适配即可
        from classification_head_enhanced import EnhancedClassificationHead
        self.classification_head = EnhancedClassificationHead(
            feature_dim=self.query_feature_dim,
            num_classes=num_classes,
            seq_len=80, # ARES output len approx 78 for 20k input
            num_topm_layers=3,
            num_cross_layers=1
        )

    def query_forward(self, x):
        """查询集前向传播"""
        return self.feature_extractor(x)
    
    def support_forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones_like(x)
        support_input = torch.stack([x, mask], dim=2)
        return self.meta_learnet(support_input)
    
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


