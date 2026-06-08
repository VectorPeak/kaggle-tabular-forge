"""Small sklearn logistic-regression OOF baseline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ktabforge.models.registry import run_model_oof


@dataclass(frozen=True)
class BaselineResult:
    """Outputs from an out-of-fold baseline run."""

    oof: pd.DataFrame
    test_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    oof_score: float
    metric_name: str = "roc_auc"
    model_family: str = "logistic_regression"
    model_params: dict[str, object] | None = None


def run_logistic_oof_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    folds: pd.DataFrame,
    target: str,
    id_column: str,
    seed: int = 42,
) -> BaselineResult:
    """Train a logistic-regression OOF baseline and return predictions in memory."""

    result = run_model_oof(
        train=train,
        test=test,
        folds=folds,
        target=target,
        id_column=id_column,
        seed=seed,
        model_family="logistic_regression",
    )
    return BaselineResult(
        oof=result.oof,
        test_predictions=result.test_predictions,
        fold_metrics=result.fold_metrics,
        oof_score=result.oof_score,
        metric_name=result.metric_name,
        model_family=result.model_family,
        model_params=result.model_params,
    )
