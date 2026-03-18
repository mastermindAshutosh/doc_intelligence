import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from backend.ingestion.models import IngestedDoc
from backend.config import settings

def _chunk(tokens: list[int], size: int = 512, stride: int = 64) -> list[list[int]]:
    if not tokens: return []
    chunks = []
    i = 0
    while i < len(tokens):
        chunks.append(tokens[i:i + size])
        i += size - stride
        if i >= len(tokens): break
    return chunks

class TextEncoder(nn.Module):
    def __init__(self, model_name: str | None = None):
        super().__init__()
        name = model_name or settings.encoder_name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model     = AutoModel.from_pretrained(name)
        self.model.eval()
        self.attn_pool = nn.Linear(settings.text_embed_dim, 1)

    def encode(self, doc: IngestedDoc, strategy: str = "first_last") -> Tensor:
        full_text = " ".join(p.text for p in doc.pages)
        tokens    = self.tokenizer.encode(full_text, add_special_tokens=False)
        if strategy == "first_last":
            return self._first_last_encode(tokens)
        elif strategy == "hierarchical":
            return self._hierarchical_encode(tokens)
        raise ValueError(f"Unknown strategy: {strategy}")

    def _first_last_encode(self, tokens: list[int]) -> Tensor:
        head = tokens[:256]
        tail = tokens[-256:] if len(tokens) > 256 else []
        sep  = [self.tokenizer.sep_token_id] if tail else []
        ids  = torch.tensor([[self.tokenizer.cls_token_id] + head + sep + tail])
        if ids.shape[1] > 512:
            ids = ids[:, :512]
        with torch.no_grad():
            out = self.model(ids)
        return out.last_hidden_state[:, 0, :]  

    def _encode_chunk(self, chunk_tokens: list[int]) -> Tensor:
        ids = torch.tensor([[self.tokenizer.cls_token_id] + chunk_tokens])
        if ids.shape[1] > 512:
            ids = ids[:, :512]
        with torch.no_grad():
            out = self.model(ids)
        return out.last_hidden_state[:, 0, :]

    def _hierarchical_encode(self, tokens: list[int]) -> Tensor:
        if not tokens:
            return torch.zeros(1, settings.text_embed_dim)
        chunks = _chunk(tokens, size=511, stride=64)
        embs   = [self._encode_chunk(c) for c in chunks]
        stacked = torch.cat(embs, dim=0)                   
        attn   = torch.softmax(self.attn_pool(stacked), dim=0)  
        return (attn * stacked).sum(dim=0, keepdim=True)         
