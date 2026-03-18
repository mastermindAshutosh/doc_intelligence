import torch
from torch import Tensor
from pathlib import Path
from .energy import EnergyOODScorer
from .mahalanobis import MahalanobisOODScorer

class OODEnsemble:
    def __init__(self, energy: EnergyOODScorer, mahal: MahalanobisOODScorer):
        self.energy = energy
        self.mahal  = mahal

    def is_ood(self, logits: Tensor, embeddings: Tensor) -> Tensor:
        # 1. Energy Score safety
        e_thresh = self.energy.threshold if self.energy.threshold is not None else 1000.0
        e_flag = self.energy.score(logits) > e_thresh
        
        # 2. Mahalanobis Score safety (ensure fitted)
        m_flag = torch.zeros(embeddings.size(0), dtype=torch.bool, device=embeddings.device)
        if hasattr(self.mahal, 'centroids') and self.mahal.centroids and getattr(self.mahal, 'precision', None) is not None:
            m_thresh = self.mahal.threshold if self.mahal.threshold is not None else 1000.0
            m_flag = self.mahal.score(embeddings) > m_thresh
            
        return e_flag | m_flag

    def save(self, path: Path) -> None:
        torch.save({
            "energy_threshold": self.energy.threshold,
            "energy_temperature": self.energy.T,
            "mahal_centroids": self.mahal.centroids,
            "mahal_precision": self.mahal.precision,
            "mahal_threshold": self.mahal.threshold,
        }, path)

    @classmethod
    def load(cls, path: Path, energy_T: float = 1.0) -> "OODEnsemble":
        data = torch.load(path)
        energy = EnergyOODScorer(temperature=data["energy_temperature"])
        energy.threshold = data["energy_threshold"]
        
        mahal = MahalanobisOODScorer()
        mahal.centroids = data["mahal_centroids"]
        mahal.precision = data["mahal_precision"]
        mahal.threshold = data["mahal_threshold"]
        
        return cls(energy, mahal)
