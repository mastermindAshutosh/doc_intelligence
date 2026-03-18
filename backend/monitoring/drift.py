import numpy as np
from scipy.stats import ks_2samp

class ConfidenceDriftDetector:
    def __init__(self, p_value_threshold: float = 0.05):
        self.p_value_threshold = p_value_threshold
        self.reference_distribution = None
        
    def fit(self, reference_confidences: list[float]):
        """Store the baseline validation set confidences."""
        self.reference_distribution = np.array(reference_confidences)
        
    def detect(self, current_confidences: list[float]) -> dict:
        """Run KS test on current window of confidences vs reference."""
        if self.reference_distribution is None:
            raise ValueError("Detector must be fitted with reference data first.")
            
        cur = np.array(current_confidences)
        stat, p_val = ks_2samp(self.reference_distribution, cur)
        
        is_drifting = p_val < self.p_value_threshold
        
        return {
            "drift_detected": bool(is_drifting),
            "ks_statistic": float(stat),
            "p_value": float(p_val)
        }
