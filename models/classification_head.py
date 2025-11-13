"""
多标签分类头实现
基于TopM_MHSA注意力机制，将重加权特征转换为多标签分类结果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath, Mlp


class TopMAttention(nn.Module):
    """
    TopM注意力机制，只关注最重要的M个位置
    参考ARES项目的实现
    """
    
    def __init__(self, dim, num_heads, dropout, top_m):
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


class MHSA_Block(nn.Module):
    """
    多头自注意力块
    """
    
    def __init__(self, embed_dim, nhead, dim_feedforward, dropout, top_m):
        super().__init__()
        drop_path_rate = 0.1
        self.attn = TopMAttention(embed_dim, nhead, dropout, top_m)
        self.drop_path = DropPath(drop_path_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=dim_feedforward, 
                      act_layer=nn.GELU, drop=0.1)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TopM_MHSA(nn.Module):
    """
    多层TopM注意力机制
    """
    
    def __init__(self, embed_dim, num_heads, num_mhsa_layers, dim_feedforward, dropout, top_m):
        super().__init__()
        self.nets = nn.ModuleList([
            MHSA_Block(embed_dim, num_heads, dim_feedforward, dropout, top_m) 
            for _ in range(num_mhsa_layers)
        ])

    def forward(self, x, pos_embed):
        output = x + pos_embed
        for layer in self.nets:
            output = layer(output)
        return output


class MultiLabelClassificationHead(nn.Module):
    """
    多标签分类头（支持两种分类方法）
    方法1：binary - 基于TopM_MHSA注意力机制，对每个类别进行独立的二分类
    """
    
    def __init__(self, feature_dim=256, num_classes=3, seq_len=119, 
                 classification_method='binary', unified_threshold=0.4):
        super(MultiLabelClassificationHead, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.classification_method = classification_method
        self.unified_threshold = unified_threshold
        
        # TopM_MHSA配置（参考ARES）
        embed_dim = feature_dim  # 256
        num_heads = 8
        dim_feedforward = feature_dim * 4  # 1024
        num_mhsa_layers = 4
        dropout = 0.1
        top_m = min(20, seq_len)  # 最多关注20个位置，但不超过序列长度
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        # 共享的TopM_MHSA模块
        self.attention = TopM_MHSA(
            embed_dim=embed_dim,
            num_heads=num_heads, 
            num_mhsa_layers=num_mhsa_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            top_m=top_m
        )
        
        if classification_method == 'binary':
            # 方法1：二分类头（每个类别独立）
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1),  # 二分类：有/无
            )
        else:
            raise ValueError(f"Unknown classification_method: {classification_method}")
        
        # 初始化位置编码
        trunc_normal_(self.pos_embed, std=0.02)
        
        print(f"多标签分类头初始化:")
        print(f"  - 特征维度: {feature_dim}")
        print(f"  - 类别数: {num_classes}")
        print(f"  - 序列长度: {seq_len}")
        print(f"  - TopM参数: {top_m}")
        print(f"  - 分类方法: {classification_method}")

    def forward(self, reweighted_features):
        """
        Args:
            reweighted_features: (batch*num_classes, feature_dim, seq_len) 
                                重加权特征
            
        Returns:
            logits: (batch, num_classes) 每个类别的分类logits
        """
        if self.classification_method == 'binary':
            return self.forward_binary(reweighted_features)
        else:
            raise ValueError(f"Unknown classification_method: {self.classification_method}")
    
    def forward_binary(self, reweighted_features):
        """
        方法1：二分类方法（原有方法）
        每个类别独立进行二分类判断
        """
        batch_times_classes, feature_dim, seq_len = reweighted_features.shape
        batch_size = batch_times_classes // self.num_classes
        
        # 重整形：(batch*num_classes, feature_dim, seq_len) -> (batch, num_classes, feature_dim, seq_len)
        features = reweighted_features.view(batch_size, self.num_classes, feature_dim, seq_len)
        
        # 收集每个类别的分类结果
        class_logits = []
        
        for class_idx in range(self.num_classes):
            # 提取当前类别的特征：(batch, feature_dim, seq_len)
            class_features = features[:, class_idx, :, :]
            
            # 转置为注意力机制需要的格式：(batch, seq_len, feature_dim)
            class_features = class_features.transpose(1, 2)
            
            # 应用TopM_MHSA注意力
            attended_features = self.attention(class_features, self.pos_embed)
            # attended_features: (batch, seq_len, feature_dim)
            
            # 全局平均池化：对序列维度取平均
            pooled_features = attended_features.mean(dim=1)  # (batch, feature_dim)
            
            # 二分类
            class_logit = self.classifier(pooled_features)  # (batch, 1)
            class_logits.append(class_logit)
        
        # 拼接所有类别的结果：(batch, num_classes)
        logits = torch.cat(class_logits, dim=1)
        
        return logits
    

    
    def predict(self, reweighted_features, threshold=None):
        """
        预测函数，返回二值化的多标签结果（支持两种方法）
        
        Args:
            reweighted_features: 重加权特征
            threshold: 二分类阈值，如果为None则使用默认阈值
            
        Returns:
            predictions: (batch, num_classes) 二值化预测结果
            probabilities: (batch, num_classes) 每个类别的概率
        """
        # 设置默认阈值
        if threshold is None:
            if self.classification_method == 'binary':
                threshold = 0.5
        
        logits = self.forward(reweighted_features)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > threshold).float()
        
        return predictions, probabilities

