import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset
from scipy.stats import chisquare


# ---------------------------------------------------------------------------
# Document dataclass
# ---------------------------------------------------------------------------

@dataclass
class Document:
    doc_hash:  str
    matter_id: str
    label_id:  int
    path:      Path          # path to raw PDF/DOCX on disk
    label_str: str = ""      # human-readable label e.g. "TAX"


# ---------------------------------------------------------------------------
# Embedding dataset — wraps pre-computed embeddings for fast DataLoader use
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """
    Stores pre-computed fused embeddings (B, 256) and their labels.

    Pre-computing avoids re-running the full encoder stack on every epoch,
    which is the dominant training cost. Think of it like pre-sorting
    recycling into labelled bins before the sorting machine runs —
    you pay the cost once and iterate cheaply.

    Usage:
        dataset = EmbeddingDataset.from_documents(docs, service)
        loader  = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(self, embeddings: Tensor, labels: Tensor):
        assert embeddings.shape[0] == labels.shape[0], \
            f"Embedding/label count mismatch: {embeddings.shape[0]} vs {labels.shape[0]}"
        self.embeddings = embeddings   # (N, 256)
        self.labels     = labels       # (N,) int64

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.embeddings[idx], self.labels[idx]

    @classmethod
    def from_documents(
        cls,
        docs: list[Document],
        service,                 # ClassificationService — provides encoders
        device: str = "cpu",
    ) -> "EmbeddingDataset":
        """
        Encode all documents once and store embeddings in memory.
        Skips documents that fail to encode (logs a warning).
        """
        embeddings, labels = [], []

        for doc in docs:
            try:
                raw_bytes = doc.path.read_bytes()
                ingested  = service.ingester.ingest(raw_bytes)
                graph     = service.graph_builder.build(ingested)

                layout_e = torch.zeros(1, 256)
                text_e   = torch.zeros(1, 384)

                if hasattr(graph, "x"):
                    layout_e = service.graph_encoder(
                        graph.x, graph.edge_index, graph.edge_attr
                    )
                if hasattr(service.text_encoder, "encode"):
                    text_e = service.text_encoder.encode(
                        ingested, strategy="first_last"
                    )

                meta_f   = service._build_meta_features(ingested)
                embedding = service.fuser(text_e, layout_e, meta_f)  # (1, 256)

                embeddings.append(embedding.squeeze(0).detach().cpu())
                labels.append(doc.label_id)

            except Exception as e:
                print(f"[WARN] Skipping {doc.path.name} ({doc.doc_hash[:8]}): {e}")

        return cls(
            embeddings=torch.stack(embeddings),
            labels=torch.tensor(labels, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Split builder
# ---------------------------------------------------------------------------

def _group_by(items: list, key) -> defaultdict:
    d: defaultdict = defaultdict(list)
    for item in items:
        d[key(item)].append(item)
    return d


def _flatten(list_of_lists: list[list]) -> list:
    return [item for sublist in list_of_lists for item in sublist]


def _label_counts(docs: list[Document], n_classes: int) -> list[int]:
    counts = defaultdict(int)
    for d in docs:
        counts[d.label_id] += 1
    return [counts.get(i, 0) for i in range(n_classes)]


def _assert_chi_squared_split(
    train: list[Document],
    val:   list[Document],
    n_classes: int,
    p_threshold: float = 0.05,
) -> None:
    """
    Runs an actual chi-square test to verify val class distribution
    matches train. Raises if the split is badly skewed.

    The original stub built the count arrays but never called chisquare().
    A skewed split (e.g. all DEED docs in val) would pass silently and
    produce misleading validation metrics.
    """
    c_train = _label_counts(train, n_classes)
    c_val   = _label_counts(val,   n_classes)

    # Scale val counts to same total as train for fair comparison
    total_train = sum(c_train)
    total_val   = sum(c_val)

    if total_val == 0:
        raise ValueError("Validation split is empty.")

    scale = total_train / total_val
    expected = [c * scale for c in c_val]

    # Avoid zero expected counts (chisquare undefined)
    safe_obs = [o for o, e in zip(c_train, expected) if e > 0]
    safe_exp = [e for e in expected if e > 0]

    if len(safe_obs) < 2:
        return   # too few classes to test meaningfully

    _, p_value = chisquare(safe_obs, f_exp=safe_exp)

    if p_value < p_threshold:
        raise ValueError(
            f"Train/val class distributions differ significantly "
            f"(chi-square p={p_value:.4f} < {p_threshold}). "
            f"Train counts: {c_train}, Val counts: {c_val}. "
            f"Try a different random seed or collect more data."
        )


def build_splits(
    docs:        list[Document],
    n_classes:   int,
    seed:        int   = 42,
    ratios:      tuple = (0.70, 0.15, 0.15),
    p_threshold: float = 0.05,
) -> tuple[list[Document], list[Document], list[Document]]:
    """
    Splits documents into train/val/test by matter_id to prevent leakage —
    documents from the same legal matter never span splits.

    Validates that train and val have statistically similar class distributions
    via chi-square test. Raises ValueError on a badly skewed split.
    """
    by_matter = _group_by(docs, key=lambda d: d.matter_id)
    matters   = list(by_matter.keys())

    rng = random.Random(seed)
    rng.shuffle(matters)

    n = len(matters)
    t = int(n * ratios[0])
    v = t + int(n * ratios[1])

    train = _flatten(by_matter[m] for m in matters[:t])
    val   = _flatten(by_matter[m] for m in matters[t:v])
    test  = _flatten(by_matter[m] for m in matters[v:])

    _assert_chi_squared_split(train, val, n_classes, p_threshold)

    print(
        f"Split: {len(train)} train / {len(val)} val / {len(test)} test "
        f"across {n} matters"
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Class weight computation (for FocalLoss alpha)
# ---------------------------------------------------------------------------

def compute_class_weights(docs: list[Document], n_classes: int) -> Tensor:
    """
    Inverse-frequency class weights for focal loss alpha.

    Rare classes (DEED, VALUATION) get higher weight so the model doesn't
    just learn to predict the majority class (AGREEMENT). Like a recycling
    sorter that's been deliberately slowed down on cardboard so it pays
    more attention to the rarer materials it keeps missing.
    """
    counts = _label_counts(docs, n_classes)
    total  = sum(counts)
    weights = [total / (n_classes * max(c, 1)) for c in counts]
    w = torch.tensor(weights, dtype=torch.float32)
    return w / w.sum() * n_classes   # normalise so weights sum to n_classes
