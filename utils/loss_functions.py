"""
损失函数模块
专门处理多标签分类中的类别不均衡问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    加权二元交叉熵损失
    专门处理正负样本不均衡问题
    """
    
    def __init__(self, pos_weight=None, reduction='mean'):
        """
        Args:
            pos_weight: 正样本权重，shape=(num_classes,)
            reduction: 'mean', 'sum', 'none'
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, logits, targets):
        """
        Args:
            logits: 模型输出logits, shape=(batch, num_classes)
            targets: 真实标签, shape=(batch, num_classes)
        """
        loss = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None, reduction='mean'):
        """
        Args:
            alpha: 平衡因子
            gamma: 聚焦参数，gamma越大越关注困难样本
            pos_weight: 正样本权重
            reduction: 'mean', 'sum', 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, logits, targets):
        """
        Args:
            logits: 模型输出logits, shape=(batch, num_classes)
            targets: 真实标签, shape=(batch, num_classes)
        """
        # 计算基础BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, 
            pos_weight=self.pos_weight,
            reduction='none'
        )
        
        # 计算概率
        probs = torch.sigmoid(logits)
        
        # 计算p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # 计算alpha_t
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        else:
            focal_loss = (1 - p_t) ** self.gamma * bce_loss
        
        # 应用reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失
    有助于提高模型泛化性能
    """
    
    def __init__(self, smoothing=0.1, pos_weight=None):
        """
        Args:
            smoothing: 平滑参数，0表示不平滑
            pos_weight: 正样本权重
        """
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        
    def forward(self, logits, targets):
        """
        Args:
            logits: 模型输出logits, shape=(batch, num_classes)
            targets: 真实标签, shape=(batch, num_classes)
        """
        if self.smoothing > 0:
            # 标签平滑
            smooth_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        else:
            smooth_targets = targets
            
        loss = F.binary_cross_entropy_with_logits(
            logits, smooth_targets,
            pos_weight=self.pos_weight
        )
        return loss


class BalancedBCELoss(nn.Module):
    """
    动态平衡的BCE损失
    根据batch中的正负样本比例动态调整权重
    """
    
    def __init__(self, beta=0.9999, reduction='mean'):
        """
        Args:
            beta: 平衡参数
            reduction: 'mean', 'sum', 'none'
        """
        super(BalancedBCELoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        
    def forward(self, logits, targets):
        """
        Args:
            logits: 模型输出logits, shape=(batch, num_classes)
            targets: 真实标签, shape=(batch, num_classes)
        """
        # 计算每个类别的有效样本数
        pos_count = targets.sum(dim=0)  # (num_classes,)
        neg_count = (1 - targets).sum(dim=0)  # (num_classes,)
        
        # 计算动态权重
        pos_weight = (1 - self.beta) / (1 - self.beta ** pos_count)
        neg_weight = (1 - self.beta) / (1 - self.beta ** neg_count)
        
        # 避免除零
        pos_weight = torch.where(pos_count > 0, pos_weight, torch.ones_like(pos_weight))
        neg_weight = torch.where(neg_count > 0, neg_weight, torch.ones_like(neg_weight))
        
        # 计算加权BCE
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 应用权重
        weighted_loss = targets * pos_weight * bce_loss + (1 - targets) * neg_weight * bce_loss
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


def get_loss_function(loss_type, **kwargs):
    """
    损失函数工厂函数
    
    Args:
        loss_type: 损失函数类型
        **kwargs: 损失函数参数
    """
    if loss_type == 'weighted_bce':
        return WeightedBCELoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'label_smooth':
        return LabelSmoothingLoss(**kwargs)
    elif loss_type == 'balanced_bce':
        return BalancedBCELoss(**kwargs)
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def test_loss_functions():
    """测试损失函数"""
    print("测试损失函数...")
    
    # 模拟数据
    batch_size = 8
    num_classes = 60
    
    # 模拟严重不均衡的数据（大部分为0）
    logits = torch.randn(batch_size, num_classes)
    targets = torch.zeros(batch_size, num_classes)
    
    # 每个样本只有2-3个正标签
    for i in range(batch_size):
        pos_indices = torch.randperm(num_classes)[:3]
        targets[i, pos_indices] = 1.0
    
    print(f"数据形状: logits={logits.shape}, targets={targets.shape}")
    print(f"正样本比例: {targets.mean():.4f}")
    
    # 计算正样本权重
    pos_ratio = targets.sum() / (targets.numel() - targets.sum())
    pos_weight = torch.full((num_classes,), 1.0 / pos_ratio)
    
    # 测试不同损失函数
    losses = {
        'BCE': nn.BCEWithLogitsLoss(),
        'Weighted BCE': WeightedBCELoss(pos_weight=pos_weight),
        'Focal': FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight),
        'Balanced BCE': BalancedBCELoss(),
        'Label Smooth': LabelSmoothingLoss(smoothing=0.1, pos_weight=pos_weight)
    }
    
    print(f"\n损失函数对比:")
    for name, criterion in losses.items():
        loss = criterion(logits, targets)
        print(f"  {name}: {loss.item():.4f}")


if __name__ == '__main__':
    test_loss_functions() 