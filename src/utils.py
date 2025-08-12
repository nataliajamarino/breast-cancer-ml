
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass

@dataclass
class ThresholdResult:
    threshold: float
    recall: float
    precision: float
    f1: float

def find_threshold_for_recall(y_true: np.ndarray, y_proba: np.ndarray, min_recall: float = 0.95) -> ThresholdResult:
    """Find the highest precision threshold that achieves at least `min_recall`.

    Returns a ThresholdResult with the chosen threshold and metrics.

    y_proba must be probability for the positive class.

    """
    # Sort thresholds from high to low precision tendency
    thresholds = np.unique(np.clip(y_proba, 0, 1))
    thresholds.sort()
    best = ThresholdResult(threshold=0.5, recall=0.0, precision=0.0, f1=0.0)
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
        if recall >= min_recall:
            # pick the one with best precision (and then f1) under recall constraint
            if (precision > best.precision) or (precision == best.precision and f1 > best.f1):
                best = ThresholdResult(threshold=float(t), recall=float(recall), precision=float(precision), f1=float(f1))
    return best
