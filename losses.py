"""
src/losses.py
=============
Loss functions used in ablation C.

  - WeightedCrossEntropyLoss: standard cross-entropy with inverse-frequency class weights
  - FocalLoss: Lin et al. (2017) — down-weights easy examples to focus on hard ones

Both are motivated by Prince Ch. 5 (Loss Functions) and the class imbalance
present in ISIC 2019 (NV is 12,875 samples; VASC is 253 samples).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """
    Standard cross-entropy loss with per-class weights.
    
    Class weights are typically set to inverse class frequency (computed in dataset.py).
    This directly addresses the imbalance problem at the loss level (Ch. 5).
    
    L(y, p) = -sum_c w_c * y_c * log(p_c)
    """
    def __init__(self, weight=None, reduction="mean"):
        super().__init__()
        self.weight    = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        weight = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(logits, targets, weight=weight, reduction=self.reduction)


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017): RetinaNet paper.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Key idea (Ch. 5 perspective): standard cross-entropy treats all samples equally.
    For imbalanced datasets, the many easy examples (e.g., NV) dominate the gradient.
    Focal loss multiplies CE by (1-p_t)^gamma, shrinking the contribution of
    well-classified easy examples and letting the model focus on hard/rare cases.
    
    gamma=0 → standard cross-entropy.
    gamma=2 → typical setting (used in ablation C).
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        """
        Args:
            gamma: focusing parameter. Higher = more focus on hard examples.
            alpha: class weights tensor (optional). If None, uniform weighting.
            reduction: 'mean' or 'sum'.
        """
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        # Standard CE loss (unreduced, per-sample)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # p_t: predicted probability for the correct class
        probs = F.softmax(logits, dim=1)
        p_t   = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma
        focal_loss   = focal_weight * ce_loss

        # Apply alpha (class weights) if provided
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_loss(cfg, class_weights=None):
    """
    Factory function: builds the loss function from config.
    
    Args:
        cfg: parsed YAML config dict
        class_weights: tensor of per-class weights (from dataset.py)
    
    Returns:
        nn.Module
    """
    loss_cfg = cfg["loss"]
    use_weights = loss_cfg.get("use_class_weights", True)
    weights = class_weights if use_weights else None

    if loss_cfg["type"] == "cross_entropy":
        return WeightedCrossEntropyLoss(weight=weights)

    elif loss_cfg["type"] == "focal":
        gamma = loss_cfg.get("focal_gamma", 2.0)
        alpha = loss_cfg.get("focal_alpha", None)
        if alpha is not None:
            alpha = torch.tensor(alpha, dtype=torch.float32)
        elif use_weights and weights is not None:
            alpha = weights  # use class weights as alpha
        return FocalLoss(gamma=gamma, alpha=alpha)

    else:
        raise ValueError(f"Unknown loss type: {loss_cfg['type']}. Use 'cross_entropy' or 'focal'.")


if __name__ == "__main__":
    # Smoke test
    logits  = torch.randn(8, 8)
    targets = torch.randint(0, 8, (8,))
    weights = torch.ones(8)

    ce  = WeightedCrossEntropyLoss(weight=weights)
    fl  = FocalLoss(gamma=2.0)
    print(f"CE loss:    {ce(logits, targets):.4f}")
    print(f"Focal loss: {fl(logits, targets):.4f}")
