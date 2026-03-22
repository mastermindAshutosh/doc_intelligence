import hashlib
import asyncio
from collections import OrderedDict, deque
from datetime import datetime

from backend.config import settings
from backend.schemas import ClassificationResponse, Routing, AuditEntry


# ---------------------------------------------------------------------------
# LRU Cache
# ---------------------------------------------------------------------------

class _LRUCache:
    """
    Bounded LRU cache for document embeddings.

    The original implementation used a plain dict which grew without limit.
    In production, each cache entry holds two tensors (features + embedding,
    ~2–3 KB each). At 512 entries that's ~3 MB — a reasonable ceiling.
    Older entries are evicted automatically when maxsize is reached.
    """

    def __init__(self, maxsize: int = 512):
        self._store: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str):
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: str, value) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)   # evict LRU entry

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# ClassificationService
# ---------------------------------------------------------------------------

class ClassificationService:
    """
    Orchestrates the full document classification pipeline:
      ingest → encode (text + layout) → fuse → batch infer → OOD → calibrate → route

    Changes vs original:
    - Cache is now an LRU with a 512-entry cap (was unbounded dict → OOM risk).
    - Audit log is a bounded deque(maxlen=10_000) (was unbounded list).
    - exit_layer is read from the batcher future, not a shared attribute,
      eliminating the race condition under concurrent requests.
    - _build_meta_features now extracts real signals from the ingested doc.
      Sending zeros through a LayerNorm + Linear is wasted compute and
      prevents the fuser from learning anything from document-level metadata.
    """

    def __init__(
        self,
        ingester,
        builder,
        graph_encoder,
        text_encoder,
        fuser,
        batcher,
        ood_ensemble,
        temp_scaler,
        router,
        model_version: str = "1.0.0",
        cache_maxsize: int = 512,
        audit_maxlen: int = 10_000,
    ):
        self.ingester      = ingester
        self.graph_builder = builder
        self.graph_encoder = graph_encoder
        self.text_encoder  = text_encoder
        self.fuser         = fuser
        self.batcher       = batcher
        self.ood_ensemble  = ood_ensemble
        self.temp_scaler   = temp_scaler
        self.router        = router
        self.model_version = model_version

        # Bounded caches — no longer leak memory over time
        self._cache: _LRUCache = _LRUCache(maxsize=cache_maxsize)
        self.audit_logs: deque  = deque(maxlen=audit_maxlen)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_meta_features(self, ingested) -> "torch.Tensor":
        """
        Extract lightweight document-level signals as a 12-dim feature vector.

        Previously this returned zeros every time, which meant:
          (a) the fuser's meta_norm LayerNorm operated on a zero vector (no-op),
          (b) the downstream Linear learned to ignore those 12 dims entirely,
          (c) you paid the compute cost for zero signal.

        Real features give the fuser something to work with, especially for
        distinguishing doc types that have similar text but very different
        structure (e.g. a 1-page DEED vs a 40-page AGREEMENT).
        """
        import torch

        page_count    = len(ingested.pages) if hasattr(ingested, "pages") else 1
        total_tokens  = sum(len(p.text.split()) for p in ingested.pages) if hasattr(ingested, "pages") else 0
        has_tables    = float(any(getattr(p, "has_table", False) for p in ingested.pages)) if hasattr(ingested, "pages") else 0.0
        avg_page_len  = total_tokens / max(page_count, 1)

        # Normalise to roughly [0, 1]
        feats = [
            min(page_count / 50.0, 1.0),           # page count (cap at 50)
            min(total_tokens / 10_000.0, 1.0),      # total word count
            min(avg_page_len / 500.0, 1.0),         # avg words per page
            has_tables,                              # contains tables (binary)
            0.0, 0.0, 0.0, 0.0,                    # reserved for future signals
            0.0, 0.0, 0.0, 0.0,                    # (font diversity, image count, etc.)
        ]
        return torch.tensor([feats], dtype=torch.float32)   # (1, 12)

    async def _log_audit(self, **kwargs) -> None:
        entry = AuditEntry(
            doc_hash      = kwargs["doc_hash"],
            timestamp     = datetime.now(),
            prediction    = kwargs["prediction"],
            confidence    = kwargs["confidence"],
            routing       = kwargs["routing"],
            model_version = kwargs["model_version"],
            ood_fired     = kwargs["ood_fired"],
            exit_layer    = kwargs.get("exit_layer", 12),
            top_k_tokens  = [],
        )
        self.audit_logs.append(entry)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def classify(self, raw_bytes: bytes) -> ClassificationResponse:
        import torch

        doc_hash = hashlib.sha256(raw_bytes).hexdigest()

        # 1. Cache lookup (LRU — bounded memory)
        cached = self._cache.get(doc_hash)
        if cached is not None:
            features, embedding = cached
        else:
            ingested = self.ingester.ingest(raw_bytes)
            graph    = self.graph_builder.build(ingested)

            layout_e = torch.zeros(1, 256)
            text_e   = torch.zeros(1, 384)

            if hasattr(graph, "x"):
                layout_e = self.graph_encoder(
                    graph.x, graph.edge_index, graph.edge_attr
                )

            if hasattr(self.text_encoder, "encode"):
                text_e = self.text_encoder.encode(ingested, strategy="first_last")

            meta_f   = self._build_meta_features(ingested)   # real features, not zeros
            embedding = self.fuser(text_e, layout_e, meta_f)
            features  = embedding

            self._cache.set(doc_hash, (features, embedding))

        # 2. Batched inference — exit_layer comes back through the future,
        #    not from a shared attribute (fixes the concurrency race).
        logits, exit_layer = await self.batcher.infer(features)

        # 3. OOD detection
        is_ood = self.ood_ensemble.is_ood(logits.unsqueeze(0), embedding)

        # 4. Temperature calibration
        probs = self.temp_scaler.scale(logits.unsqueeze(0))[0]

        # 5. Routing
        if is_ood:
            pred    = "__UNCERTAIN__"
            conf    = float(probs.max())
            routing = Routing.OOD
        else:
            pred_idx = probs.argmax().item()
            pred     = settings.classes[pred_idx]
            conf     = float(probs.max())
            routing  = self.router.route(pred, conf, False)

            if routing == Routing.HUMAN_REVIEW:
                pred = "__UNCERTAIN__"

        # 6. Async audit log (non-blocking, won't delay the response)
        asyncio.create_task(self._log_audit(
            doc_hash      = doc_hash,
            prediction    = pred,
            confidence    = conf,
            routing       = routing,
            model_version = self.model_version,
            ood_fired     = bool(is_ood),
            exit_layer    = exit_layer,
        ))

        return ClassificationResponse(
            prediction    = pred,
            confidence    = conf,
            routing       = routing,
            model_version = self.model_version,
            doc_hash      = doc_hash,
            calibrated    = True,
            ood_fired     = bool(is_ood),
            exit_layer    = exit_layer,
        )
