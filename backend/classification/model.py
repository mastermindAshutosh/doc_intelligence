import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from backend.config import settings

EXIT_LAYERS  = [3, 6, 12]
EXIT_WEIGHTS = [0.3, 0.3, 0.4]    # loss contribution per exit
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
        self._n_classes = n_classes

    def forward_inference(self, x: Tensor) -> tuple[Tensor, int]:
        """
        Per-item early exit. Each item in the batch exits as soon as its
        confidence crosses the threshold for that exit layer — it doesn't
        wait for the rest of the batch.

        Previously: .all() forced every item to run to the deepest layer
        if even one item was uncertain. This fix lets fast docs exit at
        layer 3 while hard docs continue to layer 12.

        Returns:
            final_logits: (B, n_classes)
            last_exit: the deepest layer any item reached
        """
        h = self.input_proj(x)
        B = h.shape[0]
        device = x.device

        # Accumulate results per item
        final_logits = torch.zeros(B, self._n_classes, device=device)
        done_mask = torch.zeros(B, dtype=torch.bool, device=device)
        last_exit = EXIT_LAYERS[-1]   # track deepest layer reached
        exit_idx = 0
        logits = None                  # will be set at first exit head

        for layer_i, layer in enumerate(self.encoder_layers):
            # Skip already-exited items to save compute
            active = ~done_mask
            if not active.any():
                break
            h[active] = layer(h[active])

            if (layer_i + 1) in EXIT_LAYERS:
                logits = self.exit_heads[exit_idx](h)           # (B, C)
                conf = F.softmax(logits, dim=-1).max(dim=-1).values  # (B,)

                # Items that are active AND confident enough exit here
                newly_done = active & (conf >= EXIT_THRESH[exit_idx])
                final_logits[newly_done] = logits[newly_done].detach()
                done_mask |= newly_done

                if newly_done.any():
                    last_exit = layer_i + 1   # at least one item exited here

                exit_idx += 1

        # Any items that never met a threshold get the final layer's logits
        remaining = ~done_mask
        if remaining.any() and logits is not None:
            final_logits[remaining] = logits[remaining]

        return final_logits, last_exit

    def training_loss(self, x: Tensor, targets: Tensor, focal_loss_fn) -> Tensor:
        """
        Training uses all exits for every item (no masking).
        Each exit head contributes proportionally via EXIT_WEIGHTS.
        """
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
