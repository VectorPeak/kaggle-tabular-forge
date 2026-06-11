"""Metric utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, roc_auc_score


@dataclass(frozen=True)
class MetricSpec:
    name: str
    mode: str
    scorer: Callable[[pd.Series | np.ndarray, pd.Series | np.ndarray], float]


def safe_roc_auc_score(y_true: pd.Series | np.ndarray, y_score: pd.Series | np.ndarray) -> float:
    """Return ROC AUC, or NaN when a fold does not contain both classes."""

    try:
        if pd.Series(y_true).nunique(dropna=False) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def safe_rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    try:
        return float(mean_squared_error(y_true, y_pred) ** 0.5)
    except ValueError:
        return float("nan")


def safe_mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    try:
        return float(mean_absolute_error(y_true, y_pred))
    except ValueError:
        return float("nan")


def safe_log_loss(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    try:
        return float(log_loss(y_true, y_pred))
    except ValueError:
        return float("nan")


_METRIC_SPECS = {
    "roc_auc": MetricSpec("roc_auc", "max", safe_roc_auc_score),
    "rmse": MetricSpec("rmse", "min", safe_rmse),
    "mae": MetricSpec("mae", "min", safe_mae),
    "log_loss": MetricSpec("log_loss", "min", safe_log_loss),
}


def get_metric_spec(metric_name: str) -> MetricSpec:
    key = str(metric_name).strip().lower()
    try:
        return _METRIC_SPECS[key]
    except KeyError as exc:
        known = ", ".join(sorted(_METRIC_SPECS))
        raise ValueError(
            f"Unsupported metric_name {metric_name!r}. Known metrics: {known}."
        ) from exc


def score_predictions(
    metric_name: str,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    return get_metric_spec(metric_name).scorer(y_true, y_pred)


def metric_mode(metric_name: str) -> str:
    return get_metric_spec(metric_name).mode


def metric_higher_is_better(metric_name: str) -> bool:
    return metric_mode(metric_name) == "max"


def metric_mode_or_none(metric_name: str | None) -> str | None:
    if metric_name in {None, ""}:
        return None
    return metric_mode(str(metric_name))
