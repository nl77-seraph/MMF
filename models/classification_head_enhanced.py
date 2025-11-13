"""
增强的多标签分类头 - 混合方案C
核心改进: Cross-Class Attention建模类间关系
保持: 独立二分类思想
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_, DropPath


class CrossClassAttention(nn.Module):
    """
    跨类别注意力机制
    让每个类别能够"看到"其他类别的信息，建模multi-tab场景下的类别共现模式
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(CrossClassAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Multi-Head Self-Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # 输入格式: (batch, seq, feature)
        )
        
        # Layer Normalization
        self.norm = nn.LayerNorm(feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_classes, feature_dim) - 每个类别的特征
        
        Returns:
            增强后的特征: (batch, num_classes, feature_dim)
        """
        # Self-Attention: 每个类别attend to所有其他类别
        attn_out, attn_weights = self.multihead_attn(x, x, x)
        
        # 残差连接 + LayerNorm
        x = self.norm(x + self.dropout(attn_out))
        
        return x


class SimplifiedTopMAttention(nn.Module):
    """
    简化的TopM注意力机制
    保留TopM的核心思想但减少层数以降低复杂度
    """
    
    def __init__(self, dim: int, num_heads: int, dropout: float, top_m: int):
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.top_m = top_m

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout),
        )
        self.proj_drop = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.zeros(B, self.num_heads, N, N, device=q.device, requires_grad=False)
        index = torch.topk(attn, k=self.top_m, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(x)
        return x


class EnhancedClassificationHead(nn.Module):
    """
    增强的多标签分类头 (混合方案C)
    
    改进:
    1. 简化TopM MHSA (减少层数)
    2. 添加Cross-Class Attention (核心创新)
    3. 增强的MLP分类器
    4. 保持独立二分类结构
    """
    
    def __init__(self, feature_dim: int = 256, num_classes: int = 3, seq_len: int = 119, 
                 classification_method: str = 'binary', unified_threshold: float = 0.4,
                 num_topm_layers: int = 2, num_cross_layers: int = 2):
        super(EnhancedClassificationHead, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.classification_method = classification_method
        self.unified_threshold = unified_threshold
        self.num_topm_layers = num_topm_layers
        self.num_cross_layers = num_cross_layers
        
        # TopM配置 (简化版，减少层数)
        embed_dim = feature_dim  # 256
        num_heads = 8
        dropout = 0.1
        top_m = min(20, seq_len)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        
        # 步骤1: 简化的TopM MHSA (仅1-2层)
        self.topm_layers = nn.ModuleList([
            SimplifiedTopMAttention(embed_dim, num_heads, dropout, top_m)
            for _ in range(num_topm_layers)
        ])
        self.topm_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_topm_layers)
        ])
        
        # 步骤2: Cross-Class Attention (核心改进)
        self.cross_class_layers = nn.ModuleList([
            CrossClassAttention(embed_dim, num_heads, dropout)
            for _ in range(num_cross_layers)
        ])
        
        # 步骤3: 增强的分类器 (每个类别独立)
        if classification_method == 'binary':
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.LayerNorm(embed_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, embed_dim // 4),
                nn.LayerNorm(embed_dim // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 4, 1),  # 二分类
            )
        else:
            raise ValueError(f"Unknown classification_method: {classification_method}")
        
        print(f"增强分类头初始化:")
        print(f"  - 特征维度: {feature_dim}")
        print(f"  - 类别数: {num_classes}")
        print(f"  - TopM层数: {num_topm_layers} (简化)")
        print(f"  - Cross-Class层数: {num_cross_layers} (核心改进)")
        print(f"  - 分类方法: {classification_method}")

    def forward(self, reweighted_features):
        """
        Args:
            reweighted_features: (batch*num_classes, feature_dim, seq_len)
        
        Returns:
            logits: (batch, num_classes)
        """
        if self.classification_method == 'binary':
            return self.forward_binary(reweighted_features)
        else:
            raise ValueError(f"Unknown classification_method: {self.classification_method}")
    
    def forward_binary(self, reweighted_features):
        """
        增强的二分类方法
        """
        batch_times_classes, feature_dim, seq_len = reweighted_features.shape
        batch_size = batch_times_classes // self.num_classes
        
        # 步骤1: Reshape
        # (batch*num_classes, feature_dim, seq_len) → (batch, num_classes, feature_dim, seq_len)
        features = reweighted_features.view(batch_size, self.num_classes, feature_dim, seq_len)
        
        # 步骤2: 对每个类别单独应用TopM Attention
        class_features_list = []
        
        for class_idx in range(self.num_classes):
            # 提取当前类别的特征: (batch, feature_dim, seq_len)
            class_features = features[:, class_idx, :, :]
            
            # 转置: (batch, seq_len, feature_dim)
            class_features = class_features.transpose(1, 2)
            
            # 添加位置编码
            class_features = class_features + self.pos_embed
            
            # 应用简化的TopM MHSA (仅1-2层)
            for i in range(self.num_topm_layers):
                class_features = class_features + self.topm_layers[i](class_features)
                class_features = self.topm_norms[i](class_features)
            
            # 全局平均池化: (batch, seq_len, feature_dim) → (batch, feature_dim)
            pooled_features = class_features.mean(dim=1)
            class_features_list.append(pooled_features)
        
        # 拼接所有类别特征: (batch, num_classes, feature_dim)
        all_class_features = torch.stack(class_features_list, dim=1)
        
        # 步骤3: Cross-Class Attention (建模类间关系)
        for cross_layer in self.cross_class_layers:
            all_class_features = cross_layer(all_class_features)
        # all_class_features: (batch, num_classes, feature_dim)
        
        # 步骤4: 独立二分类
        class_logits = []
        for class_idx in range(self.num_classes):
            class_feature = all_class_features[:, class_idx, :]  # (batch, feature_dim)
            class_logit = self.classifier(class_feature)  # (batch, 1)
            class_logits.append(class_logit)
        
        # 拼接: (batch, num_classes)
        logits = torch.cat(class_logits, dim=1)
        
        return logits
    
    def predict(self, reweighted_features, threshold=None):
        """
        预测函数
        
        Args:
            reweighted_features: 重加权特征
            threshold: 二分类阈值
            
        Returns:
            predictions: (batch, num_classes) 二值化预测
            probabilities: (batch, num_classes) 概率
        """
        if threshold is None:
            threshold = 0.5
        
        logits = self.forward(reweighted_features)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > threshold).float()
        
        return predictions, probabilities


def test_enhanced_classification_head():
    """测试增强分类头"""
    print("="*60)
    print("测试增强的分类头")
    print("="*60)
    
    # 设置参数
    batch_size = 4
    num_classes = 3
    feature_dim = 256
    seq_len = 119
    
    # 创建模拟数据 (重加权后的特征)
    reweighted_features = torch.randn(batch_size * num_classes, feature_dim, seq_len)
    
    print(f"\n输入形状:")
    print(f"  重加权特征: {reweighted_features.shape}")
    
    # 创建增强分类头
    head = EnhancedClassificationHead(
        feature_dim=feature_dim,
        num_classes=num_classes,
        seq_len=seq_len,
        classification_method='binary',
        num_topm_layers=2,  # 简化版，仅2层
        num_cross_layers=2   # Cross-Class层数
    )
    
    # 前向传播
    print(f"\n执行前向传播...")
    with torch.no_grad():
        logits = head(reweighted_features)
        predictions, probabilities = head.predict(reweighted_features)
    
    print(f"\n输出形状:")
    print(f"  Logits: {logits.shape}")
    print(f"  Predictions: {predictions.shape}")
    print(f"  Probabilities: {probabilities.shape}")
    
    # 验证
    assert logits.shape == (batch_size, num_classes)
    assert predictions.shape == (batch_size, num_classes)
    assert probabilities.shape == (batch_size, num_classes)
    
    print(f"\n示例输出 (第一个样本):")
    print(f"  Logits: {logits[0].numpy()}")
    print(f"  Probabilities: {probabilities[0].numpy()}")
    print(f"  Predictions: {predictions[0].numpy()}")
    
    print(f"\n✅ 增强分类头测试完成!")
    print(f"\n改进点:")
    print(f"  ✓ 简化TopM MHSA (减少到{head.num_topm_layers}层)")
    print(f"  ✓ Cross-Class Attention ({head.num_cross_layers}层)")
    print(f"  ✓ 增强MLP分类器 (3层)")
    print(f"  ✓ 保持独立二分类结构")


if __name__ == "__main__":
    test_enhanced_classification_head()


