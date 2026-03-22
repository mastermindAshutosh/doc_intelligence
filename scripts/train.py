"""
train.py — unified training entrypoint for doc_intelligence.

Modes:
  --dry-run     Smoke test with synthetic data (replaces old scaffold).
                Verifies all components wire together without needing real docs.
  (default)     Full training pipeline with real labeled documents.

Usage:
    # Smoke test (CI / first-time setup check):
    python scripts/train.py --dry-run

    # Real training:
    python scripts/train.py --data_dir ./data/labeled --epochs 40 --device cuda

Data directory layout:
    data/labeled/
        TAX/          ← one folder per class, named to match settings.classes
            doc1.pdf
            doc2.docx
        AGREEMENT/
        VALUATION/
        HR/
        DEED/
        manifest.csv  ← optional: doc_hash, matter_id columns for split control
"""

import argparse
import csv
import hashlib
import json
import types
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from backend.config import settings
from backend.classification.model import MultiExitClassifier
from backend.classification.calibration import TemperatureScaler
from backend.ood.energy import EnergyOODScorer
from backend.ood.mahalanobis import MahalanobisOODScorer
from backend.ood.ensemble import OODEnsemble
from backend.ingestion.extractor import DocumentIngester
from backend.layout.graph_builder import DocumentGraphBuilder
from backend.layout.graph_encoder import DocumentGraphEncoder
from backend.encoding.text_encoder import TextEncoder
from backend.encoding.fusion import FeatureFuser
from backend.serving.service import ClassificationService

from backend.training.dataset import (
    Document, EmbeddingDataset, build_splits, compute_class_weights
)
from backend.training.losses import FocalLoss
from backend.training.trainer import DocumentTrainer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train doc_intelligence classifier")
    p.add_argument("--data_dir",   type=Path, default=None,
                   help="Root of labeled data folder (required unless --dry-run)")
    p.add_argument("--dry-run",    action="store_true",
                   help="Smoke test with synthetic data — no real docs needed")
    p.add_argument("--epochs",     type=int,   default=40)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--device",     type=str,   default="cpu")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--wandb",      action="store_true")
    p.add_argument("--freeze_encoder_epochs", type=int, default=5,
                   help="Epochs to freeze text encoder before fine-tuning")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Smoke test (replaces the old scaffold main())
# ---------------------------------------------------------------------------

def run_dry_run() -> None:
    """
    Verifies the full training stack wires together using synthetic data.
    Equivalent to the original scaffold but clearly labelled as a test,
    running validation on a separate held-out synthetic split, and using
    real class weights instead of torch.ones(5).

    What this checks:
      - MultiExitClassifier forward pass + training_loss
      - FocalLoss with real alpha weights
      - DocumentTrainer train_epoch + validate (1 epoch)
      - Trainer returns the expected metrics dict shape

    What this does NOT check:
      - Whether real document embeddings train correctly
      - OOD fitting / calibration / threshold tuning
      - Whether macro F1 clears the deployment gate
    """
    print("--- Dry run: smoke testing pipeline wiring ---")

    n_classes = settings.n_classes
    n_train, n_val = 200, 50

    # Separate train and val tensors — original used same loader for both
    train_feats  = torch.randn(n_train, settings.fusion_hidden_dim)
    train_labels = torch.randint(0, n_classes, (n_train,))
    val_feats    = torch.randn(n_val,   settings.fusion_hidden_dim)
    val_labels   = torch.randint(0, n_classes, (n_val,))

    train_loader = DataLoader(TensorDataset(train_feats, train_labels), batch_size=16, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_feats,   val_labels),   batch_size=16)

    model = MultiExitClassifier(
        input_dim=settings.fusion_hidden_dim,
        hidden=settings.fusion_hidden_dim,
        n_classes=n_classes,
    )

    # Use inverse-frequency weights on synthetic uniform labels (all equal here)
    # In real training, compute_class_weights(train_docs, n_classes) replaces this
    alpha     = torch.ones(n_classes) / n_classes
    criterion = FocalLoss(alpha=alpha, gamma=settings.focal_loss_gamma,
                          label_smoothing=settings.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    trainer = DocumentTrainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        criterion      = criterion,
        optimizer      = optimizer,
        n_classes      = n_classes,
        device         = "cpu",
        checkpoint_dir = settings.model_dir,
        patience       = 999,        # don't early-stop on a 1-epoch smoke test
        use_wandb      = False,
        n_epochs       = 1,
    )

    print("Running single epoch...")
    loss    = trainer.train_epoch()
    metrics = trainer.validate()

    print(f"Train loss:   {loss:.4f}")
    print(f"Val accuracy: {metrics['val_accuracy']:.4f}")
    print(f"Val macro F1: {metrics['val_macro_f1']:.4f}")
    print(f"Per-class F1: {metrics['per_class_f1']}")
    print(f"Gate cleared: {metrics['gate_cleared']} (expected False on random data)")
    print("\nDry run complete — all components wired correctly.")
    print("To train for real: python scripts/train.py --data_dir ./data/labeled")


# ---------------------------------------------------------------------------
# Real training — data loading
# ---------------------------------------------------------------------------

def load_documents(data_dir: Path) -> list[Document]:
    label_map = {
        cls: i for i, cls in enumerate(settings.classes)
        if cls != "__UNCERTAIN__"
    }

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
            print(f"[WARN] No folder for class '{label_str}' at {class_dir}")
            continue
        for path in class_dir.glob("**/*"):
            if path.suffix.lower() not in {".pdf", ".docx"}:
                continue
            doc_hash  = hashlib.sha256(path.read_bytes()).hexdigest()
            matter_id = manifest.get(doc_hash, path.stem)
            docs.append(Document(
                doc_hash=doc_hash, matter_id=matter_id,
                label_id=label_id, label_str=label_str, path=path,
            ))

    print(f"Loaded {len(docs)} documents")
    for label_str, label_id in label_map.items():
        print(f"  {label_str}: {sum(1 for d in docs if d.label_id == label_id)}")
    return docs


# ---------------------------------------------------------------------------
# Phases 2–4
# ---------------------------------------------------------------------------

def fit_ood(ood_ensemble, train_dataset, model, device):
    print("\n--- Phase 2: OOD fitting ---")
    model.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for emb, lbl in DataLoader(train_dataset, batch_size=64):
            all_emb.append(emb.to(device))
            all_lbl.append(lbl.to(device))
    emb = torch.cat(all_emb)
    lbl = torch.cat(all_lbl)
    if hasattr(ood_ensemble.energy_scorer, "fit"):
        ood_ensemble.energy_scorer.fit(emb, lbl)
        print("  EnergyOODScorer fitted")
    if hasattr(ood_ensemble.mahal_scorer, "fit"):
        ood_ensemble.mahal_scorer.fit(emb, lbl)
        print("  MahalanobisOODScorer fitted")


def calibrate_temperature(temp_scaler, model, val_dataset, device):
    print("\n--- Phase 3: Temperature calibration ---")
    model.eval()
    if not hasattr(temp_scaler, "fit"):
        print("  [WARN] TemperatureScaler has no fit() — using default T=1.0")
        return
    all_logits, all_labels = [], []
    with torch.no_grad():
        for emb, lbl in DataLoader(val_dataset, batch_size=64):
            logits, _ = model.forward_inference(emb.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(lbl)
    temp_scaler.fit(torch.cat(all_logits), torch.cat(all_labels))
    print(f"  T = {getattr(temp_scaler, 'temperature', 'N/A')}")


def tune_thresholds(model, temp_scaler, val_dataset, device) -> dict:
    print("\n--- Phase 4: Threshold tuning ---")
    model.eval()
    all_confs, all_correct = [], []
    with torch.no_grad():
        for emb, lbl in DataLoader(val_dataset, batch_size=64):
            logits, _ = model.forward_inference(emb.to(device))
            probs = temp_scaler.scale(logits)
            conf, pred = probs.max(dim=-1)
            all_confs.extend(conf.cpu().tolist())
            all_correct.extend((pred == lbl.to(device)).cpu().tolist())

    thresholds = sorted(set(all_confs), reverse=True)

    best_auto = settings.auto_approve_threshold
    for t in thresholds:
        mask = [c >= t for c in all_confs]
        acc  = sum(c for c, m in zip(all_correct, mask) if m) / max(sum(mask), 1)
        if acc >= 0.99:
            best_auto = t
            break

    best_uncertain = settings.uncertain_threshold
    for t in reversed(thresholds):
        mask = [c >= t for c in all_confs]
        acc  = sum(c for c, m in zip(all_correct, mask) if m) / max(sum(mask), 1)
        if acc >= 0.80:
            best_uncertain = t
            break

    print(f"  auto_approve : {best_auto:.4f}  (was {settings.auto_approve_threshold})")
    print(f"  uncertain    : {best_uncertain:.4f}  (was {settings.uncertain_threshold})")
    return {"auto_approve_threshold": best_auto, "uncertain_threshold": best_uncertain}


def save_artifacts(model, ood_ensemble, temp_scaler, thresholds, metrics, out_dir):
    import joblib
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(),    out_dir / "classifier.pt")
    joblib.dump(ood_ensemble,          out_dir / "ood_ensemble.joblib")
    joblib.dump(temp_scaler,           out_dir / "temp_scaler.joblib")
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(out_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nArtifacts saved to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.dry_run:
        run_dry_run()
        return

    if args.data_dir is None:
        raise ValueError("--data_dir is required for real training. Use --dry-run for smoke testing.")

    # --- Encoders ---
    print("Initialising encoders...")
    ingester      = DocumentIngester()
    builder       = DocumentGraphBuilder()
    graph_encoder = DocumentGraphEncoder().eval()
    text_encoder  = TextEncoder()
    text_encoder.model.eval()
    fuser = FeatureFuser().eval()

    # Minimal service shell for embedding — no batcher/router needed here
    from types import SimpleNamespace
    service = SimpleNamespace(
        ingester=ingester, graph_builder=builder,
        graph_encoder=graph_encoder, text_encoder=text_encoder, fuser=fuser,
    )
    service._build_meta_features = types.MethodType(
        ClassificationService._build_meta_features, service
    )

    # --- Data ---
    docs = load_documents(args.data_dir)
    if len(docs) < 50:
        raise ValueError(f"Only {len(docs)} documents — need at least 50. Use --dry-run to test wiring.")

    train_docs, val_docs, _ = build_splits(docs, n_classes=settings.n_classes, seed=args.seed)

    print("\nPre-computing embeddings...")
    train_dataset = EmbeddingDataset.from_documents(train_docs, service, device=args.device)
    val_dataset   = EmbeddingDataset.from_documents(val_docs,   service, device=args.device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    # --- Model ---
    model = MultiExitClassifier(
        input_dim=settings.fusion_hidden_dim,
        hidden=settings.fusion_hidden_dim,
        n_classes=settings.n_classes,
    ).to(args.device)

    class_weights = compute_class_weights(train_docs, settings.n_classes)
    criterion     = FocalLoss(alpha=class_weights, gamma=settings.focal_loss_gamma,
                              label_smoothing=settings.label_smoothing)
    optimizer     = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # --- Phase 1 ---
    print("\n--- Phase 1: Classifier training ---")
    if args.freeze_encoder_epochs > 0:
        for param in text_encoder.model.parameters():
            param.requires_grad = False
        print(f"Text encoder frozen for first {args.freeze_encoder_epochs} epochs")

    trainer = DocumentTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer,
        n_classes=settings.n_classes, device=args.device,
        checkpoint_dir=settings.model_dir, patience=7,
        use_wandb=args.wandb, n_epochs=args.epochs,
    )
    best_metrics = trainer.fit()

    # --- Phases 2–4 ---
    ood_ensemble = OODEnsemble(EnergyOODScorer(), MahalanobisOODScorer())
    temp_scaler  = TemperatureScaler()

    fit_ood(ood_ensemble, train_dataset, model, args.device)
    calibrate_temperature(temp_scaler, model, val_dataset, args.device)
    thresholds = tune_thresholds(model, temp_scaler, val_dataset, args.device)

    save_artifacts(model, ood_ensemble, temp_scaler, thresholds, best_metrics, settings.model_dir)


if __name__ == "__main__":
    main()
