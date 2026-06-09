from __future__ import annotations

import numpy as np
import pandas as pd


def average_predictions(
    predictions: list[pd.Series],
    *,
    method: str,
    experiment_ids: list[str],
    weights: dict[str, float] | None = None,
) -> pd.Series:
    if not predictions:
        raise ValueError("at least one prediction series is required")

    frame = pd.concat([series.reset_index(drop=True) for series in predictions], axis=1)
    frame.columns = experiment_ids
    if method == "simple_average":
        return frame.mean(axis=1)
    if method == "rank_average":
        return frame.rank(method="average", pct=True).mean(axis=1)
    if method == "weighted_average":
        return _weighted_average(frame, experiment_ids=experiment_ids, weights=weights or {})
    raise ValueError(f"Unsupported ensemble method: {method}")


def _weighted_average(
    frame: pd.DataFrame,
    *,
    experiment_ids: list[str],
    weights: dict[str, float],
) -> pd.Series:
    missing = [experiment_id for experiment_id in experiment_ids if experiment_id not in weights]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing weights for candidates: {joined}")

    extra = sorted(set(weights).difference(experiment_ids))
    if extra:
        joined = ", ".join(extra)
        raise ValueError(f"Unexpected weights for candidates: {joined}")

    ordered_weights = pd.Series(
        [weights[experiment_id] for experiment_id in experiment_ids],
        index=experiment_ids,
    )
    if not np.isfinite(ordered_weights.to_numpy()).all():
        raise ValueError("Ensemble weights must be finite")
    if (ordered_weights < 0).any():
        raise ValueError("Ensemble weights must be non-negative")
    total_weight = float(ordered_weights.sum())
    if total_weight <= 0:
        raise ValueError("Ensemble weights must sum to a positive value")
    normalized = ordered_weights / total_weight
    return frame.mul(normalized, axis=1).sum(axis=1)
