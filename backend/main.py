import asyncio
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

# Globals (initialized in lifespan)
service = None
batcher = None

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service, batcher

    # Initialize all components INSIDE lifespan
    extractor = DocumentIngester()
    builder = DocumentGraphBuilder()
    graph_encoder = DocumentGraphEncoder()
    text_encoder = TextEncoder()
    fuser = FeatureFuser()

    energy = EnergyOODScorer()
    mahal = MahalanobisOODScorer()
    ood_ens = OODEnsemble(energy, mahal)

    classifier = MultiExitClassifier()
    scaler = TemperatureScaler()
    router = ConfidenceRouter()

    batcher = DynamicBatcher(classifier)

    service = ClassificationService(
        ingester=extractor,
        builder=builder,
        graph_encoder=graph_encoder,
        text_encoder=text_encoder,
        fuser=fuser,
        batcher=batcher,
        ood_ensemble=ood_ens,
        temp_scaler=scaler,
        router=router,
    )

    # Start batcher worker
    task = asyncio.create_task(batcher.run())

    yield

    task.cancel()


app = FastAPI(
    title="Document Intelligence API",
    version="1.0.0",
    lifespan=lifespan
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
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    raw_bytes = await file.read()

    if len(raw_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        result = await service.classify(raw_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
