import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv, global_add_pool
from torch_geometric.utils import softmax

class DocumentGraphEncoder(nn.Module):
    def __init__(self, in_dim=399, hidden=256, out_dim=256, heads=8, layers=3):
        super().__init__()
        self.convs = nn.ModuleList([
            GATv2Conv(in_dim, hidden//heads, heads=heads, edge_dim=4),
            GATv2Conv(hidden, hidden//heads, heads=heads, edge_dim=4),
            GATv2Conv(hidden, out_dim, heads=1, edge_dim=4),
        ])
        self.pool_gate      = nn.Linear(out_dim, 1)
        self.pool_transform = nn.Linear(out_dim, out_dim)
        self.layer_norm     = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_attr, batch=None) -> Tensor:
        if batch is None: batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index, edge_attr))
            x = F.dropout(x, p=0.1, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr)
        x = self.layer_norm(x)
        gate   = self.pool_gate(x)
        values = self.pool_transform(x)
        attn = softmax(gate, batch)
        return global_add_pool(attn * values, batch)  # (B, 256)
