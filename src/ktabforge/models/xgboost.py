"""XGBoost model adapter."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ktabforge.models.base import ModelOOFResult, run_sklearn_classifier_oof

DEFAULT_XGBOOST_PARAMS: dict[str, Any] = {
    "n_estimators": 50,
    "learning_rate": 0.05,
    "max_depth": 3,
    "eval_metric": "logloss",
}


def run_xgboost_oof(
    train: pd.DataFrame,
    test: pd.DataFrame,
    folds: pd.DataFrame,
    target: str,
    id_column: str,
    seed: int,
    model_params: dict[str, Any] | None = None,
    model_family: str = "xgboost",
) -> ModelOOFResult:
    params = {**DEFAULT_XGBOOST_PARAMS, **(model_params or {})}
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
        scale_numeric=False,
    )


def _build_estimator(seed: int, params: dict[str, Any]) -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost model family requires the optional 'xgboost' dependency. "
            "Install kaggle-tabular-forge with the 'gbdt' extra or add xgboost."
        ) from exc

    estimator_params = {"random_state": seed, **params}
    return XGBClassifier(**estimator_params)
