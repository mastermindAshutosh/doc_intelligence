import pytest
import numpy as np
import torch
from backend.monitoring.drift import ConfidenceDriftDetector
from backend.monitoring.metrics import RollingECEMetric

def test_drift_detector_no_drift():
    detector = ConfidenceDriftDetector(p_value_threshold=0.05)
    baseline = [0.90, 0.95, 0.88, 0.92, 0.94] * 20
    detector.fit(baseline)
    
    # Test with identical distribution or extremely slight noise
    current = [0.90, 0.95, 0.88, 0.92, 0.94] * 20
    res = detector.detect(current)
    
    assert res["drift_detected"] is False
    assert res["p_value"] > 0.05

def test_drift_detector_detects_drift():
    detector = ConfidenceDriftDetector(p_value_threshold=0.05)
    baseline = [0.95, 0.98, 0.92, 0.96, 0.94] * 20
    detector.fit(baseline)
    
    # Test with drifted (lower) confidence
    current = [0.60, 0.55, 0.70, 0.45, 0.50] * 20
    res = detector.detect(current)
    
    assert res["drift_detected"] is True
    assert res["p_value"] < 0.05

def test_rolling_ece_metric_computes():
    metric = RollingECEMetric(window_size=10, n_bins=5)
    
    # 5 classes
    probs1 = np.array([[0.8, 0.05, 0.05, 0.05, 0.05]]) # Pred 0, high conf
    probs2 = np.array([[0.1, 0.8, 0.05, 0.05, 0.05]])  # Pred 1, high conf
    
    # Correct predictions
    metric.update(probs1, 0)
    metric.update(probs2, 1)
    
    ece = metric.compute()
    assert isinstance(ece, float)
    assert ece >= 0.0

def test_rolling_ece_window_size():
    metric = RollingECEMetric(window_size=2)
    p = np.array([[0.5, 0.5]])
    metric.update(p, 0)
    metric.update(p, 0)
    metric.update(p, 1) # This should evict the first one
    
    assert len(metric.history) == 2
