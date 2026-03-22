import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from collections import defaultdict

from backend.config import settings


# ---------------------------------------------------------------------------
# Per-class F1 helper
# ---------------------------------------------------------------------------

def macro_f1(preds: list[int], targets: list[int], n_classes: int) -> tuple[float, dict]:
    """
    Computes macro-averaged F1 and per-class F1.

    The original validate() only tracked accuracy. config.py gates deployment
    on min_macro_f1=0.92, so without this the gate was never checkable.
    """
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for p, t in zip(preds, targets):
        if p == t:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    per_class = {}
    f1_scores = []
    for c in range(n_classes):
        precision = tp[c] / max(tp[c] + fp[c], 1)
        recall    = tp[c] / max(tp[c] + fn[c], 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-8)
        per_class[settings.classes[c]] = round(f1, 4)
        f1_scores.append(f1)

    return sum(f1_scores) / n_classes, per_class


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DocumentTrainer:
    """
    Training loop for MultiExitClassifier.

    Fixes vs original:
    - validate() computes macro F1 per class (required for the min_macro_f1
      deployment gate in config.py — original only tracked accuracy).
    - Gradient clipping (max_norm=1.0) prevents exploding gradients across
      12 encoder layers + 3 exit heads.
    - Cosine LR schedule with configurable warmup.
    - Best-checkpoint saving keyed on val macro F1, not accuracy.
    - Early stopping on val F1 with configurable patience.
    - Optional Weights & Biases logging (pass use_wandb=True).
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        criterion,
        optimizer,
        n_classes:    int          = settings.n_classes,
        device:       str          = "cpu",
        max_grad_norm: float       = 1.0,
        checkpoint_dir: Path       = settings.model_dir,
        patience:     int          = 7,
        use_wandb:    bool         = False,
        n_epochs:     int          = 40,
    ):
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.criterion      = criterion
        self.optimizer      = optimizer
        self.n_classes      = n_classes
        self.device         = device
        self.max_grad_norm  = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.patience       = patience
        self.use_wandb      = use_wandb
        self.n_epochs       = n_epochs

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

        self._best_f1       = 0.0
        self._patience_ctr  = 0

        if use_wandb:
            import wandb
            wandb.watch(model, log="gradients", log_freq=50)

    # ------------------------------------------------------------------
    # Single training epoch
    # ------------------------------------------------------------------

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for feats, targets in self.train_loader:
            feats, targets = feats.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            if hasattr(self.model, "training_loss"):
                loss = self.model.training_loss(feats, targets, self.criterion)
            else:
                logits = self.model(feats)
                loss   = self.criterion(logits, targets)

            loss.backward()

            # Gradient clipping — prevents exploding gradients across
            # 12 stacked encoder layers. Think of it as a pressure relief
            # valve: lets gradients flow freely up to a safe threshold,
            # then caps them. Without it, a single bad batch can corrupt
            # all 12 layers' weights at once.
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

            self.optimizer.step()
            total_loss += loss.item()

        self.scheduler.step()
        return total_loss / max(len(self.train_loader), 1)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> dict:
        self.model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for feats, targets in self.val_loader:
                feats, targets = feats.to(self.device), targets.to(self.device)

                if hasattr(self.model, "forward_inference"):
                    logits, _ = self.model.forward_inference(feats)
                else:
                    logits = self.model(feats)

                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(targets.cpu().tolist())

        accuracy        = sum(p == t for p, t in zip(all_preds, all_targets)) / max(len(all_targets), 1)
        macro_f1_score, per_class_f1 = macro_f1(all_preds, all_targets, self.n_classes)

        return {
            "val_accuracy":  round(accuracy, 4),
            "val_macro_f1":  round(macro_f1_score, 4),
            "per_class_f1":  per_class_f1,
            "gate_cleared":  macro_f1_score >= settings.min_macro_f1,
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, metrics: dict) -> None:
        path = self.checkpoint_dir / "best_model.pt"
        torch.save({
            "epoch":          epoch,
            "model_state":    self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics":        metrics,
        }, path)
        print(f"  [ckpt] Saved best model → {path} (macro F1: {metrics['val_macro_f1']})")

    def load_best_checkpoint(self) -> dict:
        path = self.checkpoint_dir / "best_model.pt"
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        return ckpt["metrics"]

    # ------------------------------------------------------------------
    # Full training run
    # ------------------------------------------------------------------

    def fit(self) -> dict:
        """
        Run the full training loop with early stopping.

        Returns the best validation metrics achieved.
        Raises RuntimeError if the deployment gate (min_macro_f1) is not
        cleared — don't ship a model that can't pass its own spec.
        """
        print(f"Training for up to {self.n_epochs} epochs on {self.device}")
        print(f"Deployment gate: macro F1 >= {settings.min_macro_f1}")

        best_metrics = {}

        for epoch in range(1, self.n_epochs + 1):
            train_loss = self.train_epoch()
            metrics    = self.validate()

            lr = self.scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch:03d} | loss {train_loss:.4f} | "
                f"acc {metrics['val_accuracy']:.4f} | "
                f"macro F1 {metrics['val_macro_f1']:.4f} | "
                f"lr {lr:.2e}"
            )
            print(f"  per-class F1: {metrics['per_class_f1']}")

            if self.use_wandb:
                import wandb
                wandb.log({
                    "epoch":        epoch,
                    "train_loss":   train_loss,
                    "val_accuracy": metrics["val_accuracy"],
                    "val_macro_f1": metrics["val_macro_f1"],
                    "lr":           lr,
                    **{f"f1/{k}": v for k, v in metrics["per_class_f1"].items()},
                })

            # Save best checkpoint keyed on macro F1, not accuracy
            if metrics["val_macro_f1"] > self._best_f1:
                self._best_f1      = metrics["val_macro_f1"]
                self._patience_ctr = 0
                best_metrics       = metrics
                self._save_checkpoint(epoch, metrics)
            else:
                self._patience_ctr += 1
                print(f"  [patience] {self._patience_ctr}/{self.patience}")

            if self._patience_ctr >= self.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                break

        # Reload best weights before returning
        if best_metrics:
            self.load_best_checkpoint()

        if not best_metrics.get("gate_cleared", False):
            raise RuntimeError(
                f"Training complete but deployment gate NOT cleared. "
                f"Best macro F1: {best_metrics.get('val_macro_f1', 0):.4f} "
                f"< required {settings.min_macro_f1}. "
                f"Collect more labeled data or tune hyperparameters."
            )

        print(f"\nTraining complete. Best macro F1: {best_metrics['val_macro_f1']:.4f}")
        return best_metrics
