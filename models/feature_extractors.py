"""
Enhanced feature extractor for hybrid scheme C.
Based on the DF network, with SE attention, shot attention, and related mechanisms.
Goal: improve mAP from 0.9+ to 0.95+.
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
    Squeeze-and-Excitation block for channel attention.
    Enhances important channels and suppresses less useful channels.
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
            Reweighted features.
        """
        b, c, _ = x.size()
        # Squeeze: global average pooling.
        y = self.squeeze(x).view(b, c)
        # Excitation: learn channel weights.
        y = self.excitation(y).view(b, c, 1)
        # Scale: reweight.
        return x * y.expand_as(x)


class ShotAttentionFusion(nn.Module):
    """
    Shot-level attention fusion module.
    Uses weighted fusion over multiple shots for each class instead of a simple mean.
    """
    
    def __init__(self, feature_dim: int):
        super(ShotAttentionFusion, self).__init__()
        # Attention network.
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (num_classes, shots, feature_dim) or (num_classes, shots, channels, seq_len)
        Returns:
            Fused features: (num_classes, feature_dim) or (num_classes, channels, seq_len)
        """
        if len(x.shape) == 3:
            # Case 1: (num_classes, shots, feature_dim)
            attn_scores = self.attention(x)  # (num_classes, shots, 1)
            attn_weights = F.softmax(attn_scores, dim=1)  # Softmax over the shot dimension.
            weighted_features = (x * attn_weights).sum(dim=1)  # (num_classes, feature_dim)
            return weighted_features
        elif len(x.shape) == 4:
            # Case 2: (num_classes, shots, channels, seq_len)
            num_classes, shots, channels, seq_len = x.shape
            # Apply global pooling first to get a feature vector for each shot.
            pooled = x.mean(dim=-1)  # (num_classes, shots, channels)
            # Compute attention weights.
            attn_scores = self.attention(pooled)  # (num_classes, shots, 1)
            attn_weights = F.softmax(attn_scores, dim=1)  # (num_classes, shots, 1)
            # Expand weights and apply weighted fusion.
            attn_weights = attn_weights.unsqueeze(-1)  # (num_classes, shots, 1, 1)
            weighted_features = (x * attn_weights).sum(dim=1)  # (num_classes, channels, seq_len)
            return weighted_features
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")


class DFBlock(nn.Module):
    """Single DF network block, identical to the original version."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 8, pool_size: int = 8, 
                 pool_stride: int = 4, dropout: float = 0.5,
                 activation: str = 'relu', use_se: bool = False):
        super(DFBlock, self).__init__()
        
        # First convolution layer.
        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1, 
                              padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        
        # Second convolution layer.
        self.conv2 = nn.Conv1d(in_channels=out_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size,
                              stride=1, 
                              padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        
        # Pooling and dropout.
        self.pool = nn.MaxPool1d(kernel_size=pool_size, 
                                stride=pool_stride, 
                                padding=pool_size // 2)
        self.dropout = nn.Dropout(p=dropout)
        
        # Activation functions.
        if activation == 'elu':
            self.activation1 = nn.ELU(alpha=1.0)
            self.activation2 = nn.ELU(alpha=1.0)
        else:
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        
        # Optional SE block.
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels, reduction=16)
    
    def forward(self, x):
        # First convolution block.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        
        # Second convolution block.
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        
        # Optional SE attention.
        if self.use_se:
            x = self.se(x)
        
        # Pooling and dropout.
        x = self.pool(x)
        x = self.dropout(x)
        
        return x


class DFFeatureExtractor(nn.Module):
    """Feature extractor based on the DF network, with optional enhancements."""
    
    def __init__(self, dropout: float = 0.5, use_se: bool = False):
        super(DFFeatureExtractor, self).__init__()
        
        # DF network parameters, kept consistent with the original DF network.
        self.filter_nums = [32, 64, 128, 256]
        self.kernel_size = 8
        self.pool_sizes = [8, 8, 8, 8]
        self.pool_strides = [4, 4, 4, 4]
        self.use_se = use_se
        
        # Build four blocks.
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
        Forward pass.
        
        Args:
            x: Input tensor, shape=(batch, length).
            num_blocks: Number of blocks to use. If None, all blocks are used.
            
        Returns:
            Feature tensor.
        """
        # Ensure the input is 3D: (batch, 1, length).
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Run the specified number of blocks.
        num_blocks = num_blocks if num_blocks is not None else len(self.blocks)
        
        for i in range(num_blocks):
            x = self.blocks[i](x)
        
        return x
    
    def forward_partial(self, x, num_blocks: int):
        """Forward through part of the network for the support set."""
        return self.forward(x, num_blocks)
    
    def forward_full(self, x):
        """Forward through the full network for the query set."""
        x = self.forward(x, None)
        # Transpose to match the original DF network output format: (batch, length, channels).
        return x.transpose(1, 2)


class EnhancedMetaLearnet(nn.Module):
    """
    Enhanced meta-learning network for hybrid scheme C.
    
    Improvements:
    1. Use the same four-layer structure as DF for feature extraction.
    2. Use shot-level attention fusion instead of a simple mean.
    3. Use SE channel attention.
    4. Use a multi-layer MLP weight generator instead of a single linear layer.
    """
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.5):
        super(EnhancedMetaLearnet, self).__init__()
        
        # Step 1: feature extraction network with the same structure as DF.
        self.filter_nums = [32, 64, 128, 256]
        self.kernel_size = 8
        self.pool_sizes = [8, 8, 8, 8]
        self.pool_strides = [4, 4, 4, 4]

        # Four DF blocks.
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
        
        # Step 2: shot-level attention fusion.
        self.shot_attention = ShotAttentionFusion(self.filter_nums[3])
        
        # Step 3: SE channel attention.
        self.channel_attention = SEBlock(self.filter_nums[3], reduction=16)
        
        # Step 4: multi-layer weight generator.
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.weight_generator = nn.Sequential(
            nn.Linear(self.filter_nums[3], self.filter_nums[3] * 2),
            nn.LayerNorm(self.filter_nums[3] * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.filter_nums[3] * 2, out_channels),
            nn.LayerNorm(out_channels)
        )
        
        # Residual connection if input and output dimensions match.
        self.use_residual = (self.filter_nums[3] == out_channels)
        if not self.use_residual:
            self.residual_proj = nn.Linear(self.filter_nums[3], out_channels)
    
    def forward(self, x):
        """
        Generate dynamic weights.
        
        Args:
            x: Input features (num_classes*shots, 2, length),
               or (num_classes, shots, 2, length) if already reshaped.
               Channel 0 is data and channel 1 is the mask.
            
        Returns:
            Dynamic weights (num_classes, out_channels).
        """
        # Handle input dimensions.
        if len(x.shape) == 4:
            # (num_classes, shots, 2, length)
            num_classes, shots_per_class, _, length = x.shape
            x_reshaped = x.view(num_classes * shots_per_class, 2, length)
        else:
            # (num_classes*shots, 2, length) needs num_classes and shots from outside.
            # This case must be handled at the call site and is not supported here yet.
            raise ValueError("Input must be (num_classes, shots, 2, length)")
        
        # Step 1: feature extraction through four DF blocks.
        features = self.block1(x_reshaped)
        features = self.block2(features)
        features = self.block3(features)
        features = self.block4(features)
        # features: (num_classes*shots, 256, seq_len)
        
        # Step 2: apply channel attention.
        features = self.channel_attention(features)
        # features: (num_classes*shots, 256, seq_len)
        
        # Reshape back to (num_classes, shots, 256, seq_len).
        _, channels, seq_len = features.shape
        features = features.view(num_classes, shots_per_class, channels, seq_len)
        
        # Step 3: shot-level attention fusion.
        if shots_per_class > 1:
            features = self.shot_attention(features)  # (num_classes, 256, seq_len)
        else:
            features = features.squeeze(1)  # (num_classes, 256, seq_len)
        
        # Step 4: global pooling.
        pooled_features = self.global_pool(features).squeeze(-1)  # (num_classes, 256)
        
        # Step 5: multi-layer weight generation.
        dynamic_weights = self.weight_generator(pooled_features)  # (num_classes, out_channels)
        
        # Step 6: residual connection if dimensions match.
        if self.use_residual:
            dynamic_weights = dynamic_weights + pooled_features
        elif hasattr(self, 'residual_proj'):
            dynamic_weights = dynamic_weights + self.residual_proj(pooled_features)
        
        return dynamic_weights


class EnhancedMultiMetaFingerNet(nn.Module):
    """
    Enhanced multi-label meta fingerprinting network for hybrid scheme C.
    
    Improvements:
    1. Use EnhancedMetaLearnet.
    2. Use EnhancedClassificationHead from classification_head_enhanced.py.
    3. Keep the 1x1 dynamic convolution unchanged.
    """
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.5, 
                 support_blocks: int = 0, 
                 unified_threshold: float = 0.4, use_se_in_df: bool = False):
        super(EnhancedMultiMetaFingerNet, self).__init__()
        
        self.num_classes = num_classes
        self.support_blocks = support_blocks
        
        # Main feature extraction network (DF network), optionally with SE.
        self.feature_extractor = DFFeatureExtractor(dropout, use_se=use_se_in_df)
        
        # Compute intermediate feature dimensions.
        self.support_feature_dim = self.feature_extractor.filter_nums[support_blocks - 1] if support_blocks > 0 else 128
        self.query_feature_dim = self.feature_extractor.filter_nums[-1]  # 256
        
        # Enhanced meta-learning network.
        self.meta_learnet = EnhancedMetaLearnet(
            in_channels=2,  # Data + mask.
            out_channels=self.query_feature_dim,
            dropout=dropout
        )
        
        # Feature reweighting module (1D dynamic convolution), unchanged.
        self.feature_reweighting = FeatureReweightingModule(
            feature_dim=self.query_feature_dim,
            kernel_size=1
        )
        
        # Enhanced classification head.
        from classification_head_enhanced import EnhancedClassificationHead
        self.classification_head = EnhancedClassificationHead(
            feature_dim=self.query_feature_dim,
            num_classes=num_classes,
            seq_len=80,
            num_topm_layers=2,  # Simplified TopM, reduced to two layers.
            num_cross_layers=2   # Number of cross-class attention layers.
        )
        if is_main_process():
            print(f"Enhanced network initialization complete:")
            print(f"  - Number of classes: {num_classes}")
            print(f"  - DF uses SE: {use_se_in_df}")
            print(f"  - Query feature dimension: {self.query_feature_dim}")
            print(f"  - Meta-learning network: Enhanced (Shot Attention + SE + Deep MLP)")

    def query_forward(self, x):
        """Query set forward pass."""
        return self.feature_extractor.forward_full(x)
    
    def support_forward(self, x, mask=None):
        """
        Support set forward pass that generates dynamic weights.
        
        Args:
            x: Support set data (num_classes, shots, length).
            mask: Valid data mask (num_classes, shots, length).
            
        Returns:
            Dynamic weights (num_classes, query_feature_dim).
        """
        num_classes, shots_per_class, length = x.shape
        
        # Create an all-one mask if no mask is provided.
        if mask is None:
            mask = torch.ones_like(x)
        
        # Stack data and mask on the channel dimension.
        # (num_classes, shots, length) + (num_classes, shots, length)
        # → (num_classes, shots, 2, length)
        support_input = torch.stack([x, mask], dim=2)
        
        # Generate dynamic weights through the enhanced meta_learnet.
        dynamic_weights = self.meta_learnet(support_input)
        # dynamic_weights: (num_classes, query_feature_dim)
        
        return dynamic_weights
    
    def fusion_forward(self, query_features, dynamic_weights):
        """Feature fusion forward pass."""
        return self.feature_reweighting(query_features, dynamic_weights)
    
    def classification_forward(self, reweighted_features):
        """Classification forward pass."""
        logits = self.classification_head(reweighted_features)
        return logits
    
    def forward(self, query_data, support_data, support_masks=None):
        """
        Full forward pass.
        
        Args:
            query_data: (batch, length)
            support_data: (num_classes, shots, length)
            support_masks: (num_classes, shots, length)
            
        Returns:
            Dictionary containing query features, dynamic weights, fused features, and classification results.
        """
        # Extract query set features.
        query_features = self.query_forward(query_data)
        
        # Generate support set dynamic weights with EnhancedMetaLearnet.
        dynamic_weights = self.support_forward(support_data, support_masks)
        
        # Feature fusion with 1D dynamic convolution.
        reweighted_features = self.fusion_forward(query_features, dynamic_weights)
        
        # Multi-label classification.
        logits = self.classification_forward(reweighted_features)
        
        return {
            'query_features': query_features,
            'dynamic_weights': dynamic_weights,
            'reweighted_features': reweighted_features,
            'logits': logits
        }

