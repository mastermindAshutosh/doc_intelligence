import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.schemas import ClassificationResponse, MonitoringSnapshot, AuditEntry

# Import all modules
from backend.ingestion.extractor import DocumentIngester # Fixed name
from backend.layout.graph_builder import DocumentGraphBuilder
from backend.layout.graph_encoder import DocumentGraphEncoder
from backend.encoding.text_encoder import TextEncoder
from backend.encoding.fusion import FeatureFuser
from backend.ood.energy import EnergyOODScorer
from backend.ood.mahalanobis import MahalanobisOODScorer
from backend.ood.ensemble import OODEnsemble
from backend.classification.model import MultiExitClassifier
from backend.classification.calibration import TemperatureScaler
from backend.classification.router import ConfidenceRouter
from backend.serving.batcher import DynamicBatcher
from backend.serving.service import ClassificationService
from backend.monitoring.drift import ConfidenceDriftDetector
from backend.monitoring.metrics import RollingECEMetric

# 1. Instantiate Core Components
extractor = DocumentIngester()
builder   = DocumentGraphBuilder()
l_encoder = DocumentGraphEncoder()
t_encoder = TextEncoder()
fuser     = FeatureFuser()

# OOD
energy = EnergyOODScorer()
mahal  = MahalanobisOODScorer()
ood_ens = OODEnsemble(energy, mahal)

# Classifier / Calibration
classifier = MultiExitClassifier()
scaler     = TemperatureScaler()
router     = ConfidenceRouter()

# Serving
batcher = DynamicBatcher(classifier)
service = ClassificationService(
    ingester=extractor,
    builder=builder,
    graph_encoder=l_encoder,
    text_encoder=t_encoder,
    fuser=fuser,
    batcher=batcher,
    ood_ensemble=ood_ens,
    temp_scaler=scaler,
    router=router
)

from backend.monitoring.metrics import MetricsStore
metrics_store = MetricsStore()
# drift_detector = ConfidenceDriftDetector()
# ece_metric     = RollingECEMetric()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start dynamic batcher in background
    task = asyncio.create_task(batcher.run())
    yield
    # Cleanup background worker
    task.cancel()

app = FastAPI(
    title="Document Intelligence API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {"status": "ok", "model_version": service.model_version}

@app.post("/classify", response_model=ClassificationResponse)
async def classify_doc(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        # Allow reading raw for initial test benchmarks if needed
        pass
        
    try:
        raw_bytes = await file.read()
        res = await service.classify(raw_bytes)
        
        # Update rolling monitor metrics as non-blocking event update 
        # (Usually in audit listeners, but we put it straight here in app context)
        # Assuming we eventually get actual back-truth labels to compare for rolling ECE.
        # But for inference, we can track rolling drift instead.
        # Update metrics pipeline
        import numpy as np
        metrics_store.update(
            confidence=res.confidence,
            prediction=res.prediction,
            routing=res.routing.value if hasattr(res.routing, 'value') else str(res.routing),
            probs=np.array([[res.confidence]]), # Placeholder
            actual_label=0 # Placeholder
        )
            
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring", response_model=MonitoringSnapshot)
def get_monitoring():
    return metrics_store.get_snapshot()
