import torch
from torch import Tensor
from sklearn.metrics import roc_auc_score

class MahalanobisOODScorer:
    def __init__(self):
        self.centroids = {}
        self.precision = None
        self.threshold = None

    def fit(self, embeddings: Tensor, labels: Tensor) -> None:
        """Fit on training set embeddings (penultimate layer, R^256)."""
        self.centroids = {}
        for cls in labels.unique():
            mask = labels == cls
            self.centroids[cls.item()] = embeddings[mask].mean(dim=0)
        
        residuals = torch.cat([
            embeddings[labels == cls] - mu.unsqueeze(0)
            for cls, mu in self.centroids.items()
        ])
        cov = (residuals.T @ residuals) / len(residuals)
        cov += 1e-4 * torch.eye(cov.shape[0])   # regularization
        self.precision = torch.linalg.inv(cov)   # precomputed Sigma^-1

    def score(self, z: Tensor) -> Tensor:
        """Minimum Mahalanobis distance to any class centroid. (B,)"""
        distances = []
        for mu in self.centroids.values():
            diff = z - mu.unsqueeze(0)
            d    = (diff @ self.precision * diff).sum(dim=-1).sqrt()
            distances.append(d)
        return torch.stack(distances, dim=1).min(dim=1).values

    def calibrate(self, in_embs: Tensor, ood_embs: Tensor, fpr_target: float = 0.05) -> dict:
        in_d  = self.score(in_embs)
        ood_d = self.score(ood_embs)
        self.threshold = torch.quantile(in_d, 1.0 - fpr_target).item()
        labels = torch.cat([torch.zeros(len(in_d)), torch.ones(len(ood_d))])
        scores = torch.cat([in_d, ood_d])
        return {"auroc": roc_auc_score(labels.numpy(), scores.numpy()),
                "threshold": self.threshold}
