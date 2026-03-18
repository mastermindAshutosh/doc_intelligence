from collections import deque
import numpy as np
from backend.classification.calibration import TemperatureScaler

class RollingECEMetric:
    def __init__(self, window_size: int = 1000, n_bins: int = 15):
        self.window_size = window_size
        self.n_bins = n_bins
        # Stores tuples of (probabilities: np.ndarray, matched_label: int)
        self.history = deque(maxlen=window_size)
        self.scaler = TemperatureScaler() # Just to use its ece method
        
    def update(self, probs: np.ndarray, actual_label: int):
        self.history.append((probs, actual_label))
        
    def compute(self) -> float:
        if len(self.history) == 0:
            return 0.0
            
        probs_stack = np.vstack([x[0] for x in self.history])
        labels_stack = np.array([x[1] for x in self.history])
        
        return self.scaler.ece(probs_stack, labels_stack, n_bins=self.n_bins)

from datetime import datetime, timedelta
from backend.schemas import MonitoringSnapshot
from backend.monitoring.drift import ConfidenceDriftDetector

class MetricsStore:
    def __init__(self, window_size: int = 1000):
        # Synthetic reference distribution: mean 0.85, std 0.08 (simulates baseline validation set)
        reference_data = np.random.normal(0.85, 0.08, 100).clip(0, 1).tolist()
        
        self.ece_metric = RollingECEMetric(window_size=window_size)
        self.drift_detector = ConfidenceDriftDetector()
        self.drift_detector.fit(reference_data)
        
        self.history = deque(maxlen=window_size)  # [(timestamp, confidence, prediction, routing)]
        self.class_scores = {} # class -> deque(scores)
        
    def update(self, confidence: float, prediction: str, routing: str, probs: np.ndarray = None, actual_label: int = None):
        now = datetime.now()
        self.history.append((now, confidence, prediction, routing))
        
        if prediction not in self.class_scores:
            self.class_scores[prediction] = deque(maxlen=100)
        self.class_scores[prediction].append(confidence)
        
        if probs is not None and actual_label is not None:
            self.ece_metric.update(probs, actual_label)
            
    def get_snapshot(self) -> MonitoringSnapshot:
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        
        # 1. uncertain_rate_24h
        recent = [x for x in self.history if x[0] > yesterday]
        if not recent:
            uncertain_rate = 0.0
        else:
            # Routing uses ClassificationResponse which has lowercase or Enum
            uncertain_count = sum(1 for x in recent if str(x[3]).lower() in ["human_review", "ood"])
            uncertain_rate = uncertain_count / len(recent)
            
        # 2. confidence_dist
        confidence_dist = {cls: list(scores) for cls, scores in self.class_scores.items()}
        
        # 3. ece_rolling_7d
        ece = self.ece_metric.compute()
        
        # 4. drift_flags
        recent_confidences = [x[1] for x in recent]
        drift_flags = {}
        if len(recent_confidences) >= 10: # Min samples to test
            try:
                res = self.drift_detector.detect(recent_confidences)
                drift_flags = {"overall": res["drift_detected"]}
            except Exception:
                drift_flags = {"overall": False}
        else:
             drift_flags = {"overall": False}
             
        return MonitoringSnapshot(
            timestamp=now,
            uncertain_rate_24h=float(uncertain_rate),
            confidence_dist=confidence_dist,
            ece_rolling_7d=float(ece),
            drift_flags=drift_flags,
            override_rate_7d=0.0, # Stub
            ocr_quality_p10=0.95   # Stub
        )
