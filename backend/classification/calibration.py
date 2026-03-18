import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from scipy.optimize import minimize
from scipy.special import logsumexp
from backend.config import settings

class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Minimize NLL on validation set via L-BFGS."""
        def nll(T):
            s = logits / T[0]
            log_sm = s - logsumexp(s, axis=1, keepdims=True)
            loss = -log_sm[np.arange(len(labels)), labels].mean()
            return loss

        result = minimize(
            nll, 
            x0=[1.5], 
            method="L-BFGS-B", 
            bounds=[(0.05, 20.0)]
        )
        self.temperature = float(result.x[0])
        return self.temperature

    def scale(self, logits: Tensor) -> Tensor:
        return F.softmax(logits / self.temperature, dim=-1)

    def ece(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
        """Expected Calibration Error. Target: < 0.02."""
        confs   = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == labels).astype(float)
        bins    = np.linspace(0, 1, n_bins + 1)
        ece     = 0.0
        
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confs >= lo) & (confs < hi)
            if mask.sum() == 0: continue
            
            ece += (mask.sum() / len(labels)) * abs(confs[mask].mean() - correct[mask].mean())
            
        return ece

    def ece_per_class(self, probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """Must be called after fit(). Returns {class_name: ece}."""
        res = {}
        for i, cls in enumerate(settings.classes[:settings.n_classes]):
            mask = labels == i
            if mask.sum() > 0:
                # Need to convert int correctly for labels per mask
                lbl_masked = (labels[mask] == i).astype(int)
                res[cls] = self.ece(probs[mask], lbl_masked)
        return res
