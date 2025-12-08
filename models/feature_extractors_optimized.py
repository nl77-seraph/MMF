"""
优化版特征提取器 - Base Training优化
主要优化:
1. 简化MetaLearnet - 参考Fewshot_Detection的轻量设计
2. 并行化Classification Head
3. 移除Cross-Class Attention (可选开启)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dynamic_conv1d import FeatureReweightingModule

import torch.distributed as dist

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


class Dividing(nn.Module):
    """将长序列切分为多个段"""
    def __init__(self, num_segments):
        super(Dividing, self).__init__()
        self.num_segments = num_segments

    def forward(self, x):
        B, C, L = x.shape
        p = L // self.num_segments
        x = x.view(B, C, self.num_segments, p)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B * self.num_segments, C, p)
        return x


class Combination(nn.Module):
    """将多个段合并回长序列"""
    def __init__(self, num_segments):
        super(Combination, self).__init__()
        self.num_segments = num_segments

    def forward(self, x):
        Bn, C, p = x.shape
        B = Bn // self.num_segments
        x = x.view(B, self.num_segments, C, p)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, C, self.num_segments * p)
        return x


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(ConvBlock1d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding="same"),
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
    """Local Profiling module - 查询集特征提取"""
    def __init__(self, in_channels):
        super(LocalProfiling, self).__init__()
        self.net = nn.Sequential(
            ConvBlock1d(in_channels, 32, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(32, 64, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(64, 128, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(128, 256, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
        )
        self.out_channels = 256

    def forward(self, x):
        return self.net(x)


class ARESBackbone(nn.Module):
    """查询集Backbone: 分段 -> CNN -> 合并"""
    def __init__(self, in_channels=1, num_segments=4):
        super(ARESBackbone, self).__init__()
        self.num_segments = num_segments
        self.dividing = Dividing(num_segments)
        self.profiling = LocalProfiling(in_channels)
        self.combination = Combination(num_segments)
        self.out_channels = self.profiling.out_channels

    def forward(self, x):
        x = self.dividing(x)
        x = self.profiling(x)
        x = self.combination(x)
        return x


# ============= 简化版 MetaLearnet =============
class LightweightMetaLearnet(nn.Module):
    """
    轻量级MetaLearnet - 参考Fewshot_Detection的设计
    
    关键简化:
    1. 不使用分段处理（ARES的Dividing/Combination）
    2. 使用更少的卷积层
    3. 直接GlobalMaxPool提取全局特征
    
    这样设计的好处:
    1. 计算量大幅减少
    2. 更容易适配novel classes (few-shot阶段)
    3. 保持与Fewshot_Detection一致的设计思路
    """
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.3):
        super(LightweightMetaLearnet, self).__init__()
        
        self.out_channels = out_channels
        
        # 简化的CNN backbone - 参考reweighting_net.cfg的设计
        # 输入: (batch, 2, length) 其中 channel 0=data, channel 1=mask
        self.backbone = nn.Sequential(
            # Block 1: 32 channels
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            # Block 2: 64 channels  
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            # Block 3: 128 channels
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            # Block 4: 256 channels
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            # Block 5: 256 channels
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
        )
        
        # GlobalMaxPool - 与Fewshot_Detection一致
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # 简化的权重生成器
        self.weight_generator = nn.Sequential(
            nn.Linear(256, out_channels),
            nn.LayerNorm(out_channels)
        )
    
    def forward(self, x):
        """
        Args:
            x: (num_classes, shots, 2, length)
        Returns:
            动态权重: (num_classes, out_channels)
        """
        num_classes, shots, channels, length = x.shape
        
        # Reshape: (num_classes * shots, 2, length)
        x = x.view(num_classes * shots, channels, length)
        
        # CNN特征提取
        features = self.backbone(x)  # (num_classes * shots, 256, L')
        
        # GlobalMaxPool
        pooled = self.global_pool(features).squeeze(-1)  # (num_classes * shots, 256)
        
        # Reshape并取mean融合shots
        pooled = pooled.view(num_classes, shots, -1)  # (num_classes, shots, 256)
        pooled = pooled.mean(dim=1)  # (num_classes, 256)
        
        # 生成动态权重
        weights = self.weight_generator(pooled)  # (num_classes, out_channels)
        
        return weights


# ============= 优化版 Classification Head =============
class OptimizedClassificationHead(nn.Module):
    """
    优化的分类头
    
    主要优化:
    1. TopM Attention 并行处理所有类别（不再串行循环）
    2. 可选关闭 Cross-Class Attention
    3. 简化的分类器
    """
    
    def __init__(self, feature_dim: int = 256, num_classes: int = 60, seq_len: int = 72,
                 num_topm_layers: int = 2, use_cross_attention: bool = False,
                 num_cross_layers: int = 1, dropout: float = 0.1,
                 use_cosine_classifier: bool = False):
        super(OptimizedClassificationHead, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.num_topm_layers = num_topm_layers
        self.use_cross_attention = use_cross_attention
        self.use_cosine_classifier = use_cosine_classifier
        num_heads = 8
        top_m = min(20, seq_len)
        
        # 位置编码 - 所有类别共享
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, feature_dim) * 0.02)
        
        # TopM Attention层
        self.topm_layers = nn.ModuleList([
            OptimizedTopMAttention(feature_dim, num_heads, dropout, top_m)
            for _ in range(num_topm_layers)
        ])
        self.topm_norms = nn.ModuleList([
            nn.LayerNorm(feature_dim) for _ in range(num_topm_layers)
        ])
        
        # Cross-Class Attention (可选)
        if use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=feature_dim, num_heads=num_heads,
                dropout=dropout, batch_first=True
            )
            self.cross_norm = nn.LayerNorm(feature_dim)
        
        # 简化的分类器
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1),
        )
        if use_cosine_classifier:
            # 每个类一个 prototype + 一个全局 scale
            self.class_prototypes = nn.Parameter(
                torch.randn(num_classes, feature_dim)
            )
            self.scale = nn.Parameter(torch.tensor(10.0))
        
        if is_main_process():
            print(f"OptimizedClassificationHead:")
            print(f"  - TopM层数: {num_topm_layers}")
            print(f"  - Cross-Attention: {'开启' if use_cross_attention else '关闭'}")
            print(f"  - 序列长度: {seq_len}")
            print(f"  - 使用余弦相似度分类器: {'开启' if use_cosine_classifier else '关闭'}")
    
    def forward(self, reweighted_features):
        """
        并行处理所有类别
        
        Args:
            reweighted_features: (batch * num_classes, feature_dim, seq_len)
        Returns:
            logits: (batch, num_classes)
        """
        batch_times_classes, feature_dim, seq_len = reweighted_features.shape
        batch_size = batch_times_classes // self.num_classes
        
        # 转置: (batch * num_classes, seq_len, feature_dim)
        x = reweighted_features.transpose(1, 2)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # TopM Attention - 所有类别并行处理
        for i in range(self.num_topm_layers):
            x = x + self.topm_layers[i](x)
            x = self.topm_norms[i](x)
        
        # 全局平均池化: (batch * num_classes, seq_len, feature_dim) → (batch * num_classes, feature_dim)
        x = x.mean(dim=1)
        x = x.view(batch_size, self.num_classes, feature_dim)
        # 可选的Cross-Class Attention
        if self.use_cross_attention:
            # Reshape: (batch, num_classes, feature_dim)
            x = x.view(batch_size, self.num_classes, feature_dim)
            attn_out, _ = self.cross_attn(x, x, x)
            x = self.cross_norm(x + attn_out)
            x = x.view(batch_size * self.num_classes, feature_dim)
        
        if self.use_cosine_classifier:
            # 余弦相似度 per class
            # x: (B, C, F)   prototypes: (C, F)
            x_norm = F.normalize(x, p=2, dim=-1)       # (B, C, F)
            w_norm = F.normalize(self.class_prototypes, p=2, dim=-1)  # (C, F)

            # 内积：对最后一维求和
            logits = (x_norm * w_norm.unsqueeze(0)).sum(dim=-1)       # (B, C)
            logits = self.scale * logits
        else:
            # 原来的共享 MLP 路径
            x_flat = x.view(batch_size, self.num_classes, feature_dim)        # (B*C, F)
            logits = self.classifier(x_flat)                          # (B*C, 1)
            logits = logits.view(batch_size, self.num_classes)
        
        return logits


class OptimizedTopMAttention(nn.Module):
    """优化的TopM Attention"""
    
    def __init__(self, dim: int, num_heads: int, dropout: float, top_m: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.top_m = top_m
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # TopM masking
        if self.top_m < N:
            topk_indices = torch.topk(attn, k=self.top_m, dim=-1)[1]
            mask = torch.zeros_like(attn)
            mask.scatter_(-1, topk_indices, 1.0)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


# ============= 优化版主网络 =============
class OptimizedMultiMetaFingerNet(nn.Module):
    """
    优化版多标签元指纹识别网络
    
    优化点:
    1. LightweightMetaLearnet - 简化支持集处理
    2. OptimizedClassificationHead - 并行化分类头
    3. 可配置的Cross-Attention开关
    """
    
    def __init__(self, num_classes: int = 60, dropout: float = 0.15,
                 use_cross_attention: bool = False,
                 num_topm_layers: int = 2,
                 meta_learnet_type: str = 'lightweight',
                 use_cosine_classifier: bool = False):  # 'lightweight' or 'full'
        super(OptimizedMultiMetaFingerNet, self).__init__()
        
        self.num_classes = num_classes
        
        # 1. 查询集特征提取 - 保持ARES Backbone
        self.feature_extractor = ARESBackbone(in_channels=1, num_segments=4)
        self.query_feature_dim = self.feature_extractor.out_channels  # 256
        
        # 2. MetaLearnet - 使用轻量版
        if meta_learnet_type == 'lightweight':
            self.meta_learnet = LightweightMetaLearnet(
                in_channels=2,  # data + mask
                out_channels=self.query_feature_dim,
                dropout=dropout
            )
        else:
            # 保留完整版选项
            from feature_extractors_enhanced import EnhancedMetaLearnet
            self.meta_learnet = EnhancedMetaLearnet(
                in_channels=2,
                out_channels=self.query_feature_dim,
                dropout=dropout
            )
        
        # 3. 特征重加权 - 1D动态卷积
        self.feature_reweighting = FeatureReweightingModule(
            feature_dim=self.query_feature_dim,
            kernel_size=1
        )
        
        # 4. 优化的分类头
        self.classification_head = OptimizedClassificationHead(
            feature_dim=self.query_feature_dim,
            num_classes=num_classes,
            seq_len=72,
            num_topm_layers=num_topm_layers,
            use_cross_attention=use_cross_attention,
            dropout=dropout,
            use_cosine_classifier=use_cosine_classifier
        )
        
        if is_main_process():
            self._print_model_info()
    
    def _print_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n🚀 OptimizedMultiMetaFingerNet:")
        print(f"  - 总参数: {total_params:,}")
        print(f"  - 可训练: {trainable_params:,}")
    
    def forward(self, query_data, support_data, support_masks=None):
        """
        Args:
            query_data: (batch, length) 或 (batch, 1, length)
            support_data: (num_classes, shots, length)
            support_masks: (num_classes, shots, length)
        Returns:
            dict with logits and intermediate features
        """
        # 查询集特征提取
        if len(query_data.shape) == 2:
            query_data = query_data.unsqueeze(1)
        query_features = self.feature_extractor(query_data)  # (batch, 256, seq_len)
        query_features = query_features.transpose(1, 2)  # (batch, seq_len, 256)
        
        # 支持集动态权重生成
        if support_masks is None:
            support_masks = torch.ones_like(support_data)
        support_input = torch.stack([support_data, support_masks], dim=2)  # (num_classes, shots, 2, length)
        dynamic_weights = self.meta_learnet(support_input)  # (num_classes, 256)
        
        # 特征融合（1D动态卷积）
        reweighted_features = self.feature_reweighting(query_features, dynamic_weights)
        
        # 多标签分类
        logits = self.classification_head(reweighted_features)
        
        return {
            'logits': logits,
            'query_features': query_features,
            'dynamic_weights': dynamic_weights,
            'reweighted_features': reweighted_features
        }


# ============= 测试代码 =============
if __name__ == '__main__':
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 创建模型
    model = OptimizedMultiMetaFingerNet(
        num_classes=60,
        dropout=0.15,
        use_cross_attention=False,
        num_topm_layers=2,
        meta_learnet_type='lightweight'
    ).to(device)
    
    # 模拟输入
    batch_size = 32
    num_classes = 60
    shots = 20
    query_length = 20000
    support_length = 10000
    
    query_data = torch.randn(batch_size, query_length).to(device)
    support_data = torch.randn(num_classes, shots, support_length).to(device)
    support_masks = torch.ones_like(support_data)
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(query_data, support_data, support_masks)
    
    # 测试速度
    torch.cuda.synchronize()
    start = time.time()
    n_iters = 10
    for _ in range(n_iters):
        with torch.no_grad():
            output = model(query_data, support_data, support_masks)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"\n⏱️ 推理速度测试:")
    print(f"  - {n_iters} 次迭代耗时: {elapsed:.3f}s")
    print(f"  - 平均每次: {elapsed/n_iters*1000:.1f}ms")
    print(f"  - Output shape: {output['logits'].shape}")



