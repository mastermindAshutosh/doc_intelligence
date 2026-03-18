from pydantic import BaseModel
from enum import Enum
from datetime import datetime

# ── Routing ──────────────────────────────────────────────────────────
class Routing(str, Enum):
    DIRECT       = "direct"        # high confidence, auto-approved
    ASYNC_CONFIRM = "async_confirm" # medium confidence, heavy model confirms
    HUMAN_REVIEW  = "human_review"  # low confidence, human labels
    OOD           = "ood"           # out-of-distribution, reject

# ── Classification Response ──────────────────────────────────────────
class ClassificationResponse(BaseModel):
    prediction:    str            # class name or __UNCERTAIN__
    confidence:    float          # calibrated probability [0,1]
    routing:       Routing
    model_version: str
    doc_hash:      str
    calibrated:    bool = True
    ood_fired:     bool = False
    exit_layer:    int | None = None

# ── Audit Log Entry ──────────────────────────────────────────────────
class AuditEntry(BaseModel):
    doc_hash:      str
    timestamp:     datetime
    prediction:    str
    confidence:    float
    routing:       Routing
    model_version: str
    ood_fired:     bool
    exit_layer:    int | None
    top_k_tokens:  list[str]      # top 10 evidence tokens

# ── Monitoring Metrics ───────────────────────────────────────────────
class MonitoringSnapshot(BaseModel):
    timestamp:          datetime
    uncertain_rate_24h: float
    confidence_dist:    dict[str, list[float]]  # class → recent scores
    ece_rolling_7d:     float
    drift_flags:        dict[str, bool]         # class → drifted
    override_rate_7d:   float | None
    ocr_quality_p10:    float

# ── Batch Request ────────────────────────────────────────────────────
class BatchClassificationRequest(BaseModel):
    doc_hashes: list[str]   # references to already-uploaded docs

class BatchClassificationResponse(BaseModel):
    results:     list[ClassificationResponse]
    failed:      list[str]   # doc_hashes that errored
    duration_ms: float
