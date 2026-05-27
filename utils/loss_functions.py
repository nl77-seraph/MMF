"""
Loss function module.
Handles class imbalance in multi-label classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    Weighted binary cross-entropy loss.
    Handles positive/negative sample imbalance.
    """
    
    def __init__(self, pos_weight=None, reduction='mean'):
        """
        Args:
            pos_weight: Positive sample weights, shape=(num_classes,)
            reduction: 'mean', 'sum', 'none'
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, logits, targets):
        """
        Args:
            logits: Model output logits, shape=(batch, num_classes)
            targets: Ground-truth labels, shape=(batch, num_classes)
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
            alpha: Balance factor.
            gamma: Focusing parameter; larger values focus more on hard samples.
            pos_weight: Positive sample weights.
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
            logits: Model output logits, shape=(batch, num_classes)
            targets: Ground-truth labels, shape=(batch, num_classes)
        """
        # Compute base BCE loss.
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, 
            pos_weight=self.pos_weight,
            reduction='none'
        )
        
        # Compute probabilities.
        probs = torch.sigmoid(logits)
        
        # Compute p_t.
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute alpha_t.
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        else:
            focal_loss = (1 - p_t) ** self.gamma * bce_loss
        
        # Apply reduction.
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss.
    Helps improve model generalization.
    """
    
    def __init__(self, smoothing=0.1, pos_weight=None):
        """
        Args:
            smoothing: Smoothing parameter; 0 means no smoothing.
            pos_weight: Positive sample weights.
        """
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        
    def forward(self, logits, targets):
        """
        Args:
            logits: Model output logits, shape=(batch, num_classes)
            targets: Ground-truth labels, shape=(batch, num_classes)
        """
        if self.smoothing > 0:
            # Label smoothing.
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
    Dynamically balanced BCE loss.
    Adjusts weights based on positive/negative ratios in the batch.
    """
    
    def __init__(self, beta=0.9999, reduction='mean'):
        """
        Args:
            beta: Balance parameter.
            reduction: 'mean', 'sum', 'none'
        """
        super(BalancedBCELoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        
    def forward(self, logits, targets):
        """
        Args:
            logits: Model output logits, shape=(batch, num_classes)
            targets: Ground-truth labels, shape=(batch, num_classes)
        """
        # Compute the effective sample count for each class.
        pos_count = targets.sum(dim=0)  # (num_classes,)
        neg_count = (1 - targets).sum(dim=0)  # (num_classes,)
        
        # Compute dynamic weights.
        pos_weight = (1 - self.beta) / (1 - self.beta ** pos_count)
        neg_weight = (1 - self.beta) / (1 - self.beta ** neg_count)
        
        # Avoid division by zero.
        pos_weight = torch.where(pos_count > 0, pos_weight, torch.ones_like(pos_weight))
        neg_weight = torch.where(neg_count > 0, neg_weight, torch.ones_like(neg_weight))
        
        # Compute weighted BCE.
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Apply weights.
        weighted_loss = targets * pos_weight * bce_loss + (1 - targets) * neg_weight * bce_loss
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label:
      - Use (1-p)^gamma_pos * log(p) for positive samples.
      - Use p^gamma_neg * log(1-p) for negative samples.
      - Add clipping to the negative term to reduce the dominance of very easy negatives.
    Expects logits before sigmoid.
    """
    def __init__(self, gamma_pos=0.0, gamma_neg=3.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gp = gamma_pos
        self.gn = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        # logits, targets: (B, C); targets are in {0, 1}.
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid                 # p
        xs_neg = 1.0 - x_sigmoid           # 1-p

        # Negative-term clipping reduces the weight of easy negative samples.
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Focal modulation.
        loss_pos = targets * ((1.0 - xs_pos) ** self.gp) * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * (xs_pos ** self.gn)   * torch.log(xs_neg.clamp(min=self.eps))

        loss = -(loss_pos + loss_neg)
        return loss.mean()


def get_loss_function(loss_type, **kwargs):
    """
    Loss function factory.
    
    Args:
        loss_type: Loss function type.
        **kwargs: Loss function parameters.
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
    """Test loss functions."""
    print("Testing loss functions...")
    
    # Simulated data.
    batch_size = 8
    num_classes = 60
    
    # Simulate severely imbalanced data where most labels are 0.
    logits = torch.randn(batch_size, num_classes)
    targets = torch.zeros(batch_size, num_classes)
    
    # Each sample has only 2-3 positive labels.
    for i in range(batch_size):
        pos_indices = torch.randperm(num_classes)[:3]
        targets[i, pos_indices] = 1.0
    
    print(f"Data shape: logits={logits.shape}, targets={targets.shape}")
    print(f"Positive sample ratio: {targets.mean():.4f}")
    
    # Compute positive sample weights.
    pos_ratio = targets.sum() / (targets.numel() - targets.sum())
    pos_weight = torch.full((num_classes,), 1.0 / pos_ratio)
    
    # Test different loss functions.
    losses = {
        'BCE': nn.BCEWithLogitsLoss(),
        'Weighted BCE': WeightedBCELoss(pos_weight=pos_weight),
        'Focal': FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight),
        'Balanced BCE': BalancedBCELoss(),
        'Label Smooth': LabelSmoothingLoss(smoothing=0.1, pos_weight=pos_weight)
    }
    
    print(f"\nLoss function comparison:")
    for name, criterion in losses.items():
        loss = criterion(logits, targets)
        print(f"  {name}: {loss.item():.4f}")


if __name__ == '__main__':
    test_loss_functions() 