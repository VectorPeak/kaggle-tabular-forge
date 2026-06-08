"""Logistic-regression model adapter."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression

from ktabforge.models.base import ModelOOFResult, run_sklearn_classifier_oof

DEFAULT_LOGISTIC_PARAMS: dict[str, Any] = {"max_iter": 1000}


def run_logistic_oof(
    train: pd.DataFrame,
    test: pd.DataFrame,
    folds: pd.DataFrame,
    target: str,
    id_column: str,
    seed: int,
    model_params: dict[str, Any] | None = None,
    model_family: str = "logistic_regression",
) -> ModelOOFResult:
    params = {**DEFAULT_LOGISTIC_PARAMS, **(model_params or {})}
    return run_sklearn_classifier_oof(
        train=train,
        test=test,
        folds=folds,
        target=target,
        id_column=id_column,
        seed=seed,
        model_family=model_family,
        estimator_factory=_build_estimator,
        model_params=params,
        scale_numeric=True,
    )


def _build_estimator(seed: int, params: dict[str, Any]) -> LogisticRegression:
    estimator_params = {"random_state": seed, **params}
    return LogisticRegression(**estimator_params)
