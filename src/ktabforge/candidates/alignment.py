from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ktabforge.candidates.pool import Candidate


@dataclass(frozen=True)
class CandidatePredictions:
    candidate: Candidate
    oof: pd.DataFrame
    submission: pd.DataFrame


def load_aligned_predictions(
    candidates: list[Candidate],
    *,
    target: str,
) -> list[CandidatePredictions]:
    loaded = [
        CandidatePredictions(
            candidate=candidate,
            oof=pd.read_parquet(candidate.oof_path),
            submission=pd.read_csv(candidate.test_pred_path),
        )
        for candidate in candidates
    ]
    validate_aligned_predictions(loaded, target=target)
    return loaded


def validate_aligned_predictions(
    predictions: list[CandidatePredictions],
    *,
    target: str,
) -> None:
    if not predictions:
        raise ValueError("candidate pool is empty")

    reference = predictions[0]
    _validate_single_prediction(reference, target=target)
    reference_oof_ids = reference.oof["id"].tolist()
    reference_targets = reference.oof[target].tolist()
    reference_folds = reference.oof["fold"].tolist() if "fold" in reference.oof.columns else None
    reference_test_ids = reference.submission["id"].tolist()

    for prediction in predictions[1:]:
        _validate_single_prediction(prediction, target=target)
        if prediction.oof["id"].tolist() != reference_oof_ids:
            raise ValueError("oof ids do not align")
        if prediction.oof[target].tolist() != reference_targets:
            raise ValueError("oof targets do not align")
        if reference_folds is not None and prediction.oof["fold"].tolist() != reference_folds:
            raise ValueError("oof folds do not align")
        if prediction.submission["id"].tolist() != reference_test_ids:
            raise ValueError("test ids do not align")


def _validate_single_prediction(prediction: CandidatePredictions, *, target: str) -> None:
    required_oof_columns = {"id", target, "prediction"}
    missing_oof = required_oof_columns.difference(prediction.oof.columns)
    if missing_oof:
        joined = ", ".join(sorted(missing_oof))
        raise ValueError(f"oof is missing required columns: {joined}")

    required_submission_columns = {"id", target}
    missing_submission = required_submission_columns.difference(prediction.submission.columns)
    if missing_submission:
        joined = ", ".join(sorted(missing_submission))
        raise ValueError(f"submission is missing required columns: {joined}")

    _validate_probability_series(prediction.oof["prediction"], "oof prediction")
    _validate_probability_series(prediction.submission[target], "test prediction")


def _validate_probability_series(values: pd.Series, label: str) -> None:
    numeric = pd.to_numeric(values, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{label} contains non-finite values")
    if ((numeric < 0) | (numeric > 1)).any():
        raise ValueError(f"{label} is outside [0, 1]")

