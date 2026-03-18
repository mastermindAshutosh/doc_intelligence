import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from backend.config import settings

EXIT_LAYERS  = [3, 6, 12]
EXIT_WEIGHTS = [0.3, 0.3, 0.4]   # loss contribution per exit
EXIT_THRESH  = [0.95, 0.92, 0.88] # confidence threshold per exit (production)

class MultiExitClassifier(nn.Module):
    def __init__(self, input_dim=256, hidden=256, n_classes=5):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
            for _ in range(12)
        ])
        
        self.input_proj = nn.Linear(input_dim, hidden)
        self.exit_heads = nn.ModuleList([
            nn.Linear(hidden, n_classes) for _ in EXIT_LAYERS
        ])

    def forward_inference(self, x: Tensor) -> tuple[Tensor, int]:
        h = self.input_proj(x)
        exit_idx = 0
        
        for layer_i, layer in enumerate(self.encoder_layers):
            h = layer(h)
            
            if (layer_i + 1) in EXIT_LAYERS:
                logits = self.exit_heads[exit_idx](h)
                conf   = F.softmax(logits, dim=-1).max(dim=-1).values
                
                # Check early exit threshold across whole batch
                if (conf >= EXIT_THRESH[exit_idx]).all():
                    return logits, layer_i + 1   # early exit
                
                exit_idx += 1
                
        return logits, 12

    def training_loss(self, x: Tensor, targets: Tensor, focal_loss_fn) -> Tensor:
        h = self.input_proj(x)
        total_loss = 0.0
        exit_idx = 0
        
        for layer_i, layer in enumerate(self.encoder_layers):
            h = layer(h)
            
            if (layer_i + 1) in EXIT_LAYERS:
                logits = self.exit_heads[exit_idx](h)
                total_loss += EXIT_WEIGHTS[exit_idx] * focal_loss_fn(logits, targets)
                exit_idx += 1
                
        return total_loss
