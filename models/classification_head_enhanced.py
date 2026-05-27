
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_, DropPath
from utils.misc import is_main_process

class CrossClassAttention(nn.Module):

    
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(CrossClassAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Multi-Head Self-Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Input format: (batch, seq, feature).
        )
        
        # Layer Normalization
        self.norm = nn.LayerNorm(feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_classes, feature_dim) - features for each class.
        
        Returns:
            Enhanced features: (batch, num_classes, feature_dim)
        """
        # Self-attention: each class attends to all other classes.
        attn_out, attn_weights = self.multihead_attn(x, x, x)
        
        # Residual connection + LayerNorm.
        x = self.norm(x + self.dropout(attn_out))
        
        return x


class SimplifiedTopMAttention(nn.Module):
    """
    Simplified TopM attention.
    Keeps the core TopM idea while reducing the number of layers to lower complexity.
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

    
    def __init__(self, feature_dim: int = 256, num_classes: int = 3, seq_len: int = 119, 
                 num_topm_layers: int = 2, num_cross_layers: int = 2):
        super(EnhancedClassificationHead, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.num_topm_layers = num_topm_layers
        self.num_cross_layers = num_cross_layers
        
        # TopM configuration with fewer layers.
        embed_dim = feature_dim  # 256
        num_heads = 8
        dropout = 0.1
        top_m = min(20, seq_len)
        
        # Positional encoding.
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        
        # Step 1: simplified TopM MHSA with only 1-2 layers.
        self.topm_layers = nn.ModuleList([
            SimplifiedTopMAttention(embed_dim, num_heads, dropout, top_m)
            for _ in range(num_topm_layers)
        ])
        self.topm_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_topm_layers)
        ])
        
        self.cross_class_layers = nn.ModuleList([
            CrossClassAttention(embed_dim, num_heads, dropout)
            for _ in range(num_cross_layers)
        ])
        
        # Step 3: enhanced classifier, independent for each class.
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 1),  # Binary classification.
        )

        if is_main_process():
            print(f"Enhanced classification head initialized:")
            print(f"  - Feature dimension: {feature_dim}")
            print(f"  - Number of classes: {num_classes}")
            print(f"  - TopM layers: {num_topm_layers} (simplified)")
            print(f"  - Cross-class layers: {num_cross_layers} (core improvement)")

    def forward(self, reweighted_features):
        """
        Args:
            reweighted_features: (batch*num_classes, feature_dim, seq_len)
        
        Returns:
            logits: (batch, num_classes)
        """

        return self.forward_binary(reweighted_features)

    
    def forward_binary(self, reweighted_features):
        """
        Enhanced binary classification method.
        """
        batch_times_classes, feature_dim, seq_len = reweighted_features.shape
        batch_size = batch_times_classes // self.num_classes
        
        # Step 1: reshape.
        # (batch*num_classes, feature_dim, seq_len) → (batch, num_classes, feature_dim, seq_len)
        features = reweighted_features.view(batch_size, self.num_classes, feature_dim, seq_len)
        
        # Step 2: apply TopM attention separately for each class.
        class_features_list = []
        
        for class_idx in range(self.num_classes):
            # Extract features for the current class: (batch, feature_dim, seq_len).
            class_features = features[:, class_idx, :, :]
            
            # Transpose to (batch, seq_len, feature_dim).
            class_features = class_features.transpose(1, 2)
            
            # Add positional encoding.
            class_features = class_features + self.pos_embed
            
            # Apply simplified TopM MHSA.
            for i in range(self.num_topm_layers):
                class_features = class_features + self.topm_layers[i](class_features)
                class_features = self.topm_norms[i](class_features)
            
            # Global average pooling: (batch, seq_len, feature_dim) -> (batch, feature_dim).
            pooled_features = class_features.mean(dim=1)
            class_features_list.append(pooled_features)
        
        # Concatenate features for all classes: (batch, num_classes, feature_dim).
        all_class_features = torch.stack(class_features_list, dim=1)
        
        for cross_layer in self.cross_class_layers:
            all_class_features = cross_layer(all_class_features)
        # all_class_features: (batch, num_classes, feature_dim)
        bs, nc, fd = all_class_features.shape
        class_features_flat = all_class_features.view(bs * nc, fd)
        logits_flat = self.classifier(class_features_flat)  # (batch*num_classes, 1)
        logits = logits_flat.view(batch_size, nc)
        
        return logits
    
