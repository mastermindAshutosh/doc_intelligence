import asyncio
import json
import joblib
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException

from backend.schemas import ClassificationResponse
from backend.config import settings
# Core components
from backend.ingestion.extractor import DocumentIngester
from backend.layout.graph_builder import DocumentGraphBuilder
from backend.layout.graph_encoder import DocumentGraphEncoder
from backend.encoding.text_encoder import TextEncoder
from backend.encoding.fusion import FeatureFuser
# OOD
from backend.ood.energy import EnergyOODScorer
from backend.ood.mahalanobis import MahalanobisOODScorer
from backend.ood.ensemble import OODEnsemble
# Classification
from backend.classification.model import MultiExitClassifier
from backend.classification.calibration import TemperatureScaler
from backend.classification.router import ConfidenceRouter
# Serving
from backend.serving.batcher import DynamicBatcher
from backend.serving.service import ClassificationService

service = None
batcher = None
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service, batcher

    # --- Build all components ---
    extractor     = DocumentIngester()
    builder       = DocumentGraphBuilder()
    graph_encoder = DocumentGraphEncoder()
    text_encoder  = TextEncoder()
    fuser         = FeatureFuser()
    energy        = EnergyOODScorer()
    mahal         = MahalanobisOODScorer()
    ood_ens       = OODEnsemble(energy, mahal)
    classifier    = MultiExitClassifier()
    scaler        = TemperatureScaler()
    router        = ConfidenceRouter()

    # --- Load trained artifacts (if training has been run) ---
    model_dir = settings.model_dir

    ckpt_path = model_dir / "classifier.pt"
    if ckpt_path.exists():
        classifier.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"Loaded classifier weights from {ckpt_path}")
    else:
        print("[WARN] No classifier.pt found — serving with random weights. Run training first.")

    ood_path = model_dir / "ood_ensemble.joblib"
    if ood_path.exists():
        # BUG FIX: reassign the variable that service will receive below,
        # not a shadow that service never sees.
        ood_ens = joblib.load(ood_path)
        print(f"Loaded OOD ensemble from {ood_path}")

    scaler_path = model_dir / "temp_scaler.joblib"
    if scaler_path.exists():
        # BUG FIX: same — scaler must be updated before being passed to service
        scaler = joblib.load(scaler_path)
        print(f"Loaded temperature scaler from {scaler_path}")

    thresh_path = model_dir / "thresholds.json"
    if thresh_path.exists():
        # BUG FIX: use context manager — open() without it leaks a file handle
        with open(thresh_path) as f:
            thresholds = json.load(f)
        settings.auto_approve_threshold = thresholds["auto_approve_threshold"]
        settings.uncertain_threshold    = thresholds["uncertain_threshold"]
        print(f"Loaded thresholds: auto_approve={settings.auto_approve_threshold:.4f}, "
              f"uncertain={settings.uncertain_threshold:.4f}")

    # --- Set all modules to eval mode ---
    graph_encoder.eval()
    fuser.eval()
    classifier.eval()

    # --- Build service AFTER artifacts are loaded ---
    # ood_ens and scaler now hold the fitted/loaded objects, not the fresh ones
    batcher = DynamicBatcher(classifier)
    service = ClassificationService(
        ingester      = extractor,
        builder       = builder,
        graph_encoder = graph_encoder,
        text_encoder  = text_encoder,
        fuser         = fuser,
        batcher       = batcher,
        ood_ensemble  = ood_ens,
        temp_scaler   = scaler,
        router        = router,
    )

    # --- Warm-up pass (eliminates first-request JIT/CUDA init cost) ---
    dummy = torch.zeros(1, settings.fusion_hidden_dim)
    with torch.no_grad():
        classifier.forward_inference(dummy)
    print("Warm-up pass complete")

    # --- Start batcher background task ---
    task = asyncio.create_task(batcher.run())
    yield
    task.cancel()


app = FastAPI(
    title="Document Intelligence API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_version": service.model_version}


@app.post("/classify", response_model=ClassificationResponse)
async def classify_doc(file: UploadFile = File(...)):
    if file.content_type not in [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # BUG FIX (from earlier review): stream with size cap instead of
    # allocating full buffer then rejecting — avoids wasting memory on
    # files that will be rejected anyway
    chunks, size = [], 0
    async for chunk in file:
        size += len(chunk)
        if size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        chunks.append(chunk)
    raw_bytes = b"".join(chunks)

    try:
        result = await service.classify(raw_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
