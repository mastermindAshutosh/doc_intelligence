import hashlib
import asyncio
from datetime import datetime

from backend.config import settings
from backend.schemas import ClassificationResponse, Routing, AuditEntry


class ClassificationService:
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
        model_version="1.0.0"
    ):
        self.ingester = ingester
        self.graph_builder = builder
        self.graph_encoder = graph_encoder
        self.text_encoder = text_encoder
        self.fuser = fuser
        self.batcher = batcher
        self.ood_ensemble = ood_ensemble
        self.temp_scaler = temp_scaler
        self.router = router
        self.model_version = model_version

        self.cache = {}
        self.audit_logs = []

    def _build_meta_features(self, ingested):
        import torch
        return torch.zeros(1, 12)

    async def _get_cached(self, key):
        return self.cache.get(key)

    async def _set_cache(self, key, value):
        self.cache[key] = value

    async def _log_audit(self, **kwargs):
        entry = AuditEntry(
            doc_hash=kwargs["doc_hash"],
            timestamp=datetime.now(),
            prediction=kwargs["prediction"],
            confidence=kwargs["confidence"],
            routing=kwargs["routing"],
            model_version=kwargs["model_version"],
            ood_fired=kwargs["ood_fired"],
            exit_layer=kwargs.get("exit_layer", 12),
            top_k_tokens=[]
        )
        self.audit_logs.append(entry)

    async def classify(self, raw_bytes: bytes) -> ClassificationResponse:
        import torch

        doc_hash = hashlib.sha256(raw_bytes).hexdigest()

        # 1. Cache
        cached = await self._get_cached(doc_hash)
        if cached:
            features, embedding = cached
        else:
            ingested = self.ingester.ingest(raw_bytes)
            graph = self.graph_builder.build(ingested)

            layout_e = torch.zeros(1, 256)
            text_e = torch.zeros(1, 384)

            if hasattr(graph, "x"):
                layout_e = self.graph_encoder(
                    graph.x, graph.edge_index, graph.edge_attr
                )

            if hasattr(self.text_encoder, "encode"):
                text_e = self.text_encoder.encode(
                    ingested, strategy="first_last"
                )

            meta_f = self._build_meta_features(ingested)

            embedding = self.fuser(text_e, layout_e, meta_f)
            features = embedding

            await self._set_cache(doc_hash, (features, embedding))

        # 2. Model inference (batched)
        logits = await self.batcher.infer(features)

        # 3. OOD
        is_ood = self.ood_ensemble.is_ood(
            logits.unsqueeze(0), embedding
        )

        # 4. Calibration
        probs = self.temp_scaler.scale(logits.unsqueeze(0))[0]

        # 5. Routing
        if is_ood:
            pred = "__UNCERTAIN__"
            conf = float(probs.max())
            routing = Routing.OOD
        else:
            pred_idx = probs.argmax().item()
            pred = settings.classes[pred_idx]
            conf = float(probs.max())
            routing = self.router.route(pred, conf, False)

            if routing == Routing.HUMAN_REVIEW:
                pred = "__UNCERTAIN__"

        # 6. Audit (non-blocking)
        asyncio.create_task(self._log_audit(
            doc_hash=doc_hash,
            prediction=pred,
            confidence=conf,
            routing=routing,
            model_version=self.model_version,
            ood_fired=bool(is_ood),
            exit_layer=self.batcher.last_exit_layer,
        ))

        return ClassificationResponse(
            prediction=pred,
            confidence=conf,
            routing=routing,
            model_version=self.model_version,
            doc_hash=doc_hash,
            calibrated=True,
            ood_fired=bool(is_ood),
            exit_layer=self.batcher.last_exit_layer
        )
