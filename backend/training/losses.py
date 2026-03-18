import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class FocalLoss(nn.Module):
    def __init__(self, alpha: Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)   # (n_classes,)
        self.gamma = gamma

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        ce      = F.cross_entropy(logits, targets, reduction="none")
        pt      = torch.exp(-ce)
        alpha_t = self.alpha[targets]
        return (alpha_t * ((1 - pt) ** self.gamma) * ce).mean()
