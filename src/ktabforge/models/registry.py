"""Model-family registry and OOF dispatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from ktabforge.models.base import ModelOOFResult
from ktabforge.models.lightgbm import run_lightgbm_oof
from ktabforge.models.logistic import run_logistic_oof
from ktabforge.models.xgboost import run_xgboost_oof

ModelRunner = Callable[..., ModelOOFResult]

MODEL_ALIASES: dict[str, str] = {
    "logistic": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "lightgbm": "lightgbm",
    "lgbm": "lightgbm",
    "xgboost": "xgboost",
    "xgb": "xgboost",
}

MODEL_RUNNERS: dict[str, ModelRunner] = {
    "logistic_regression": run_logistic_oof,
    "lightgbm": run_lightgbm_oof,
    "xgboost": run_xgboost_oof,
}


def run_model_oof(
    train: pd.DataFrame,
    test: pd.DataFrame,
    folds: pd.DataFrame,
    target: str,
    id_column: str,
    seed: int,
    model_family: str,
    model_params: dict[str, Any] | None = None,
) -> ModelOOFResult:
    """Run a configured model family and return unified OOF outputs."""

    canonical_family = normalize_model_family(model_family)
    runner = MODEL_RUNNERS[canonical_family]
    return runner(
        train=train,
        test=test,
        folds=folds,
        target=target,
        id_column=id_column,
        seed=seed,
        model_params=model_params,
        model_family=canonical_family,
    )


def normalize_model_family(model_family: str) -> str:
    key = model_family.strip().lower()
    try:
        return MODEL_ALIASES[key]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_ALIASES))
        raise ValueError(
            f"Unknown model family {model_family!r}. Known families: {known}."
        ) from exc
