"""
train.py — end-to-end training entrypoint for doc_intelligence.

Runs all four phases in sequence:
  1. Classifier training   (MultiExitClassifier + FocalLoss)
  2. OOD fitting           (EnergyOODScorer + MahalanobisOODScorer)
  3. Temperature scaling   (TemperatureScaler calibration)
  4. Threshold tuning      (auto_approve + uncertain thresholds)

Usage:
    python train.py --data_dir ./data/labeled --epochs 40 --device cuda

Data directory layout expected:
    data/labeled/
        TAX/         ← one folder per class, named to match settings.classes
            doc1.pdf
            doc2.docx
        AGREEMENT/
            ...
        VALUATION/
        HR/
        DEED/
        manifest.csv  ← optional: doc_hash, matter_id, label columns
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from backend.config import settings
from backend.classification.model import MultiExitClassifier
from backend.classification.calibration import TemperatureScaler
from backend.classification.router import ConfidenceRouter
from backend.ood.energy import EnergyOODScorer
from backend.ood.mahalanobis import MahalanobisOODScorer
from backend.ood.ensemble import OODEnsemble
from backend.ingestion.extractor import DocumentIngester
from backend.layout.graph_builder import DocumentGraphBuilder
from backend.layout.graph_encoder import DocumentGraphEncoder
from backend.encoding.text_encoder import TextEncoder
from backend.encoding.fusion import FeatureFuser
from backend.serving.service import ClassificationService

from backend.training.dataset import Document, EmbeddingDataset, build_splits, compute_class_weights
from backend.training.losses import FocalLoss
from backend.training.trainer import DocumentTrainer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train doc_intelligence classifier")
    p.add_argument("--data_dir",   type=Path, required=True,  help="Root of labeled data folder")
    p.add_argument("--epochs",     type=int,  default=40)
    p.add_argument("--batch_size", type=int,  default=32)
    p.add_argument("--lr",         type=float,default=3e-4)
    p.add_argument("--device",     type=str,  default="cpu")
    p.add_argument("--seed",       type=int,  default=42)
    p.add_argument("--wandb",      action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--freeze_encoder_epochs", type=int, default=5,
                   help="Epochs to train with text encoder frozen before fine-tuning")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_documents(data_dir: Path) -> list[Document]:
    """
    Scans data_dir for PDF/DOCX files organised in per-class subdirectories.
    Reads matter_id from manifest.csv if present, otherwise uses filename stem.

    manifest.csv columns (optional): doc_hash, matter_id, label
    """
    label_map = {cls: i for i, cls in enumerate(settings.classes) if cls != "__UNCERTAIN__"}

    # Try manifest first
    manifest_path = data_dir / "manifest.csv"
    manifest: dict[str, str] = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                manifest[row["doc_hash"]] = row.get("matter_id", row["doc_hash"])

    docs: list[Document] = []
    for label_str, label_id in label_map.items():
        class_dir = data_dir / label_str
        if not class_dir.exists():
            print(f"[WARN] No folder found for class '{label_str}' at {class_dir}")
            continue

        for path in class_dir.glob("**/*"):
            if path.suffix.lower() not in {".pdf", ".docx"}:
                continue
            raw = path.read_bytes()
            doc_hash  = hashlib.sha256(raw).hexdigest()
            matter_id = manifest.get(doc_hash, path.stem)  # fallback to filename

            docs.append(Document(
                doc_hash  = doc_hash,
                matter_id = matter_id,
                label_id  = label_id,
                label_str = label_str,
                path      = path,
            ))

    print(f"Loaded {len(docs)} documents across {len(label_map)} classes")
    for label_str, label_id in label_map.items():
        count = sum(1 for d in docs if d.label_id == label_id)
        print(f"  {label_str}: {count}")
    return docs


# ---------------------------------------------------------------------------
# Phase 2 — OOD fitting
# ---------------------------------------------------------------------------

def fit_ood(
    ood_ensemble: OODEnsemble,
    train_dataset: EmbeddingDataset,
    model: MultiExitClassifier,
    device: str,
) -> None:
    """
    Collect training embeddings and fit OOD detector statistics.
    Both EnergyOODScorer and MahalanobisOODScorer need in-distribution
    embeddings to establish their baseline — without this they have no
    reference point and will either flag everything or nothing as OOD.
    """
    print("\n--- Phase 2: OOD fitting ---")
    model.eval()
    all_embeddings = []
    all_labels     = []

    with torch.no_grad():
        for emb, label in DataLoader(train_dataset, batch_size=64):
            all_embeddings.append(emb.to(device))
            all_labels.append(label.to(device))

    embeddings = torch.cat(all_embeddings)
    labels     = torch.cat(all_labels)

    if hasattr(ood_ensemble.energy_scorer, "fit"):
        ood_ensemble.energy_scorer.fit(embeddings, labels)
        print("  EnergyOODScorer fitted")

    if hasattr(ood_ensemble.mahal_scorer, "fit"):
        ood_ensemble.mahal_scorer.fit(embeddings, labels)
        print("  MahalanobisOODScorer fitted")


# ---------------------------------------------------------------------------
# Phase 3 — Temperature calibration
# ---------------------------------------------------------------------------

def calibrate_temperature(
    temp_scaler: TemperatureScaler,
    model: MultiExitClassifier,
    val_dataset: EmbeddingDataset,
    device: str,
) -> None:
    """
    Fit temperature T on the val set so that confidence scores are
    actually calibrated (i.e. 90% confidence → 90% accuracy).
    Without this, the model is almost always overconfident, which means
    HUMAN_REVIEW routing only triggers for borderline-impossible cases
    rather than genuinely uncertain ones.
    """
    print("\n--- Phase 3: Temperature calibration ---")
    model.eval()

    if hasattr(temp_scaler, "fit"):
        all_logits, all_labels = [], []
        with torch.no_grad():
            for emb, label in DataLoader(val_dataset, batch_size=64):
                logits, _ = model.forward_inference(emb.to(device))
                all_logits.append(logits.cpu())
                all_labels.append(label)

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        temp_scaler.fit(logits, labels)
        print(f"  Temperature fitted: T={getattr(temp_scaler, 'temperature', 'N/A')}")
    else:
        print("  [WARN] TemperatureScaler has no fit() method — using default T=1.0")


# ---------------------------------------------------------------------------
# Phase 4 — Threshold tuning
# ---------------------------------------------------------------------------

def tune_thresholds(
    model: MultiExitClassifier,
    temp_scaler: TemperatureScaler,
    val_dataset: EmbeddingDataset,
    device: str,
) -> dict[str, float]:
    """
    Sweep auto_approve and uncertain thresholds on the val set to find
    values that hit the target human-review rate while minimising errors
    sent to DIRECT routing.

    Returns the best thresholds found. The original config values (0.92, 0.65)
    are arbitrary defaults — fitting on val data almost always improves routing.
    """
    print("\n--- Phase 4: Threshold tuning ---")
    model.eval()

    all_confs, all_correct = [], []
    with torch.no_grad():
        for emb, label in DataLoader(val_dataset, batch_size=64):
            logits, _ = model.forward_inference(emb.to(device))
            probs = temp_scaler.scale(logits)
            conf, pred = probs.max(dim=-1)
            correct = (pred == label.to(device))
            all_confs.extend(conf.cpu().tolist())
            all_correct.extend(correct.cpu().tolist())

    # Find auto_approve threshold: highest confidence where accuracy >= 0.99
    thresholds = sorted(set(all_confs), reverse=True)
    best_auto = settings.auto_approve_threshold
    for t in thresholds:
        mask = [c >= t for c in all_confs]
        acc_at_t = sum(c for c, m in zip(all_correct, mask) if m) / max(sum(mask), 1)
        if acc_at_t >= 0.99:
            best_auto = t
            break

    # Uncertain threshold: lowest confidence where accuracy still >= 0.80
    best_uncertain = settings.uncertain_threshold
    for t in reversed(thresholds):
        mask = [c >= t for c in all_confs]
        acc_at_t = sum(c for c, m in zip(all_correct, mask) if m) / max(sum(mask), 1)
        if acc_at_t >= 0.80:
            best_uncertain = t
            break

    print(f"  auto_approve_threshold: {best_auto:.4f}  (was {settings.auto_approve_threshold})")
    print(f"  uncertain_threshold:    {best_uncertain:.4f}  (was {settings.uncertain_threshold})")
    return {"auto_approve_threshold": best_auto, "uncertain_threshold": best_uncertain}


# ---------------------------------------------------------------------------
# Save all artifacts
# ---------------------------------------------------------------------------

def save_artifacts(
    model: MultiExitClassifier,
    ood_ensemble: OODEnsemble,
    temp_scaler: TemperatureScaler,
    thresholds: dict,
    metrics: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    torch.save(model.state_dict(), out_dir / "classifier.pt")
    joblib.dump(ood_ensemble,      out_dir / "ood_ensemble.joblib")
    joblib.dump(temp_scaler,       out_dir / "temp_scaler.joblib")

    with open(out_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    with open(out_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll artifacts saved to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # --- Build serving components (encoders) ---
    print("Initialising encoders...")
    ingester      = DocumentIngester()
    builder       = DocumentGraphBuilder()
    graph_encoder = DocumentGraphEncoder().eval()
    text_encoder  = TextEncoder()
    text_encoder.model.eval()
    fuser         = FeatureFuser().eval()

    # Minimal service shell for encoding — batcher/OOD/router not needed here
    from types import SimpleNamespace
    service = SimpleNamespace(
        ingester      = ingester,
        graph_builder = builder,
        graph_encoder = graph_encoder,
        text_encoder  = text_encoder,
        fuser         = fuser,
        _build_meta_features = ClassificationService._build_meta_features.__func__,
    )
    # Bind method properly
    import types
    service._build_meta_features = types.MethodType(
        ClassificationService._build_meta_features, service
    )

    # --- Load + split documents ---
    docs = load_documents(args.data_dir)
    if len(docs) < 50:
        raise ValueError(f"Only {len(docs)} documents found. Need at least 50 to train meaningfully.")

    train_docs, val_docs, test_docs = build_splits(
        docs, n_classes=settings.n_classes, seed=args.seed
    )

    # --- Pre-compute embeddings (pay encoding cost once) ---
    print("\nPre-computing embeddings (this takes a while)...")
    train_dataset = EmbeddingDataset.from_documents(train_docs, service, device=args.device)
    val_dataset   = EmbeddingDataset.from_documents(val_docs,   service, device=args.device)
    test_dataset  = EmbeddingDataset.from_documents(test_docs,  service, device=args.device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    # --- Build model + loss + optimizer ---
    model = MultiExitClassifier(
        input_dim=settings.fusion_hidden_dim,
        hidden=settings.fusion_hidden_dim,
        n_classes=settings.n_classes,
    ).to(args.device)

    class_weights = compute_class_weights(train_docs, settings.n_classes)
    criterion     = FocalLoss(
        alpha           = class_weights,
        gamma           = settings.focal_loss_gamma,
        label_smoothing = settings.label_smoothing,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # --- Phase 1: Classifier training ---
    print("\n--- Phase 1: Classifier training ---")

    # Freeze text encoder for first N epochs, then unfreeze for fine-tuning.
    # Training a randomly-initialised classifier on top of frozen pretrained
    # features first lets the exit heads stabilise before gradients flow
    # back through the encoder. Same reason you let concrete set before
    # adding the next floor.
    if args.freeze_encoder_epochs > 0:
        for param in text_encoder.model.parameters():
            param.requires_grad = False
        print(f"Text encoder frozen for first {args.freeze_encoder_epochs} epochs")

    trainer = DocumentTrainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        criterion    = criterion,
        optimizer    = optimizer,
        n_classes    = settings.n_classes,
        device       = args.device,
        checkpoint_dir = settings.model_dir,
        patience     = 7,
        use_wandb    = args.wandb,
        n_epochs     = args.epochs,
    )

    best_metrics = trainer.fit()

    # --- Phases 2–4 ---
    energy = EnergyOODScorer()
    mahal  = MahalanobisOODScorer()
    ood_ensemble = OODEnsemble(energy, mahal)
    temp_scaler  = TemperatureScaler()

    fit_ood(ood_ensemble, train_dataset, model, args.device)
    calibrate_temperature(temp_scaler, model, val_dataset, args.device)
    thresholds = tune_thresholds(model, temp_scaler, val_dataset, args.device)

    # --- Save everything ---
    save_artifacts(
        model        = model,
        ood_ensemble = ood_ensemble,
        temp_scaler  = temp_scaler,
        thresholds   = thresholds,
        metrics      = best_metrics,
        out_dir      = settings.model_dir,
    )

    print("\nRun complete. To serve the trained model, restart the FastAPI app.")
    print("The lifespan() function will load artifacts from settings.model_dir automatically.")


if __name__ == "__main__":
    main()
