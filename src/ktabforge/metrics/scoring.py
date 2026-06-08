"""Metric utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def safe_roc_auc_score(y_true: pd.Series | np.ndarray, y_score: pd.Series | np.ndarray) -> float:
    """Return ROC AUC, or NaN when a fold does not contain both classes."""

    try:
        if pd.Series(y_true).nunique(dropna=False) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")
