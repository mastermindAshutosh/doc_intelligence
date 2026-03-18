import torch
from torch import Tensor
from sklearn.metrics import roc_auc_score

class EnergyOODScorer:
    def __init__(self, temperature: float = 1.0):
        self.T = temperature
        self.threshold = None

    def score(self, logits: Tensor) -> Tensor:
        """Lower = more in-distribution (Energy Score > 0)."""
        shifted = logits - logits.max(dim=-1, keepdim=True).values
        # Negative free energy calculation
        # To make it so smaller = more indist (so indist is lower in magnitude):
        # The classic formulation is -T * log summa(exp(logit/T)) which yields negative values,
        # with greater negatives being more indistribuition. The spec says "Lower = more in distribution",
        # so we negate it.
        return self.T * torch.log(torch.exp(shifted / self.T).sum(dim=-1))

    def calibrate(self, in_logits: Tensor, ood_logits: Tensor, fpr_target: float = 0.05) -> dict:
        in_scores  = self.score(in_logits)
        ood_scores = self.score(ood_logits)
        labels = torch.cat([torch.zeros(len(in_scores)), torch.ones(len(ood_scores))])
        scores = torch.cat([in_scores, ood_scores])
        auroc  = roc_auc_score(labels.numpy(), scores.numpy())
        self.threshold = torch.quantile(in_scores, 1.0 - fpr_target).item()
        return {"auroc": auroc, "threshold": self.threshold}
