from pydantic_settings import BaseSettings
from pathlib import Path
import os

# --- Load .env file into os.environ to populate HF_TOKEN etc. before any transformer imports ---
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

class Settings(BaseSettings):
    # --- Input ---
    supported_formats: list[str] = ["pdf", "docx", "png", "tiff", "jpeg"]
    max_pages: int = 200
    max_file_mb: int = 50
    text_quality_threshold: float = 0.35   # below -> OCR triggered

    # --- Classes ---
    classes: list[str] = ["TAX", "AGREEMENT", "VALUATION", "HR", "DEED", "__UNCERTAIN__"]
    n_classes: int = 5                      # trained classes (excl. __UNCERTAIN__)

    # --- SLOs ---
    sync_latency_p99_ms: int = 800
    async_latency_p99_s: int = 30

    # --- Calibration gates (deployment blockers) ---
    max_ece_macro: float = 0.02
    max_ece_per_class: float = 0.04
    min_macro_f1: float = 0.92

    # --- OOD ---
    ood_fpr_target: float = 0.05            # 5% false OOD alarm rate
    ood_temperature: float = 1.0

    # --- Serving ---
    max_batch_size: int = 32
    batch_flush_ms: int = 20

    # --- Confidence routing thresholds (per class) ---
    # These are defaults; calibrate() overwrites with fitted values
    auto_approve_threshold: float = 0.92
    uncertain_threshold: float = 0.65

    # --- Training ---
    focal_loss_gamma: float = 2.0           # tune on val set
    label_smoothing: float = 0.1
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_embed_dim: int = 384
    layout_embed_dim: int = 256
    meta_dim: int = 12
    fusion_hidden_dim: int = 256

    # --- Paths ---
    model_dir: Path = Path("models/")
    cache_dir: Path = Path(".cache/")

    model_config = {"env_prefix": "DOCINT_"}

settings = Settings()
