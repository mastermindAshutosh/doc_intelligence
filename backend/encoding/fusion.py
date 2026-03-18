import torch
import torch.nn as nn
from torch import Tensor

class FeatureFuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_norm   = nn.LayerNorm(384)
        self.layout_norm = nn.LayerNorm(256)
        self.meta_norm   = nn.LayerNorm(12)
        self.projection  = nn.Sequential(
            nn.Linear(652, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256)
        )

    def forward(self, text_emb: Tensor, layout_emb: Tensor, meta_feat: Tensor) -> Tensor:
        fused = torch.cat([
            self.text_norm(text_emb),    # (B, 384)
            self.layout_norm(layout_emb), # (B, 256)
            self.meta_norm(meta_feat),    # (B, 12)
        ], dim=-1)                        # (B, 652)
        return self.projection(fused)     # (B, 256)
