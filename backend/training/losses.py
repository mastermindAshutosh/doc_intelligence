import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """
    Focal loss with per-class alpha weighting.

    Fix vs original: alpha is moved to the same device as targets before
    indexing. The original `self.alpha[targets]` fails silently or raises
    a device mismatch error when training on CUDA, because register_buffer
    keeps alpha on CPU unless explicitly moved.

    Alpha should be computed via dataset.compute_class_weights() to handle
    class imbalance — rare classes (DEED, VALUATION) get higher weight.
    """

    def __init__(self, alpha: Tensor, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.register_buffer("alpha", alpha)   # (n_classes,)
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        # Move alpha to same device as targets — fixes the silent CPU/CUDA mismatch
        alpha = self.alpha.to(targets.device)

        if self.label_smoothing > 0:
            n_classes  = logits.shape[-1]
            smooth_val = self.label_smoothing / n_classes
            # Soft targets: (1 - smoothing) on true class, smoothing/C elsewhere
            with torch.no_grad():
                soft_targets = torch.full_like(logits, smooth_val)
                soft_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + smooth_val)
            log_probs = F.log_softmax(logits, dim=-1)
            ce = -(soft_targets * log_probs).sum(dim=-1)   # (B,)
        else:
            ce = F.cross_entropy(logits, targets, reduction="none")

        pt      = torch.exp(-ce)
        alpha_t = alpha[targets]                           # (B,) — now device-safe
        loss    = alpha_t * ((1 - pt) ** self.gamma) * ce
        return loss.mean()
