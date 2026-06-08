"""LightGBM model adapter."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ktabforge.models.base import ModelOOFResult, run_sklearn_classifier_oof

DEFAULT_LIGHTGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 50,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "verbosity": -1,
}


def run_lightgbm_oof(
    train: pd.DataFrame,
    test: pd.DataFrame,
    folds: pd.DataFrame,
    target: str,
    id_column: str,
    seed: int,
    model_params: dict[str, Any] | None = None,
    model_family: str = "lightgbm",
) -> ModelOOFResult:
    params = {**DEFAULT_LIGHTGBM_PARAMS, **(model_params or {})}
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
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM model family requires the optional 'lightgbm' dependency. "
            "Install kaggle-tabular-forge with the 'gbdt' extra or add lightgbm."
        ) from exc

    estimator_params = {"random_state": seed, **params}
    return LGBMClassifier(**estimator_params)
