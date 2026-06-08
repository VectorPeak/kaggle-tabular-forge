"""Shared OOF model helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ktabforge.features.basic import infer_feature_columns, split_columns_by_dtype
from ktabforge.metrics.scoring import safe_roc_auc_score

EstimatorFactory = Callable[[int, dict[str, Any]], Any]


@dataclass(frozen=True)
class ModelOOFResult:
    """Unified outputs from an out-of-fold model run."""

    oof: pd.DataFrame
    test_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    oof_score: float
    metric_name: str
    model_family: str
    model_params: dict[str, Any]


def run_sklearn_classifier_oof(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    folds: pd.DataFrame | pd.Series,
    target: str,
    id_column: str,
    seed: int,
    model_family: str,
    estimator_factory: EstimatorFactory,
    model_params: dict[str, Any] | None = None,
    scale_numeric: bool = False,
) -> ModelOOFResult:
    """Train a sklearn-compatible classifier with fold-local preprocessing."""

    fold_frame = normalize_folds(folds=folds, train=train, id_column=id_column)
    _validate_model_inputs(train, test, fold_frame, target, id_column)

    params = dict(model_params or {})
    feature_columns = infer_feature_columns(train, target=target, id_column=id_column)
    numeric_columns, categorical_columns = split_columns_by_dtype(train, feature_columns)

    oof_parts: list[pd.DataFrame] = []
    test_fold_predictions: list[np.ndarray] = []
    metric_rows: list[dict[str, float | int | str]] = []

    for fold in sorted(fold_frame["fold"].unique()):
        train_idx = fold_frame.index[fold_frame["fold"] != fold]
        valid_idx = fold_frame.index[fold_frame["fold"] == fold]

        x_train = train.loc[train_idx, feature_columns]
        y_train = train.loc[train_idx, target]
        x_valid = train.loc[valid_idx, feature_columns]
        y_valid = train.loc[valid_idx, target]

        if y_train.nunique(dropna=False) < 2:
            prediction = np.full(len(valid_idx), float(pd.to_numeric(y_train).mean()))
            test_prediction = np.full(len(test), prediction[0])
        else:
            model = build_classifier_pipeline(
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                estimator=estimator_factory(seed, params),
                scale_numeric=scale_numeric,
            )
            model.fit(x_train, y_train)
            prediction = positive_class_probability(model, x_valid)
            test_prediction = positive_class_probability(model, test[feature_columns])

        fold_score = safe_roc_auc_score(y_valid, prediction)
        metric_rows.append(
            {
                "fold": int(fold),
                "metric_name": "roc_auc",
                "roc_auc": fold_score,
                "score": fold_score,
                "rows": int(len(valid_idx)),
            }
        )
        test_fold_predictions.append(test_prediction)

        oof_parts.append(
            pd.DataFrame(
                {
                    id_column: train.loc[valid_idx, id_column].to_numpy(),
                    target: y_valid.to_numpy(),
                    "prediction": prediction,
                    "fold": int(fold),
                },
                index=valid_idx,
            )
        )

    oof = pd.concat(oof_parts).sort_index().reset_index(drop=True)
    test_prediction_mean = np.mean(test_fold_predictions, axis=0)
    test_predictions = pd.DataFrame(
        {id_column: test[id_column].to_numpy(), "prediction": test_prediction_mean}
    )
    fold_metrics = pd.DataFrame(metric_rows)
    oof_score = safe_roc_auc_score(oof[target], oof["prediction"])

    return ModelOOFResult(
        oof=oof,
        test_predictions=test_predictions,
        fold_metrics=fold_metrics,
        oof_score=float(oof_score),
        metric_name="roc_auc",
        model_family=model_family,
        model_params=params,
    )


def build_classifier_pipeline(
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    estimator: Any,
    scale_numeric: bool,
) -> Pipeline:
    transformers = []
    if numeric_columns:
        numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps=numeric_steps), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def normalize_folds(
    *, folds: pd.DataFrame | pd.Series, train: pd.DataFrame, id_column: str
) -> pd.DataFrame:
    """Normalize supported fold containers to an id/fold DataFrame."""

    if isinstance(folds, pd.Series):
        return pd.DataFrame({id_column: train[id_column].to_numpy(), "fold": folds.to_numpy()})
    if isinstance(folds, pd.DataFrame):
        if "fold" not in folds.columns:
            raise ValueError("folds is missing fold column")
        if id_column in folds.columns:
            return folds[[id_column, "fold"]].copy()
        return pd.DataFrame(
            {id_column: train[id_column].to_numpy(), "fold": folds["fold"].to_numpy()}
        )
    raise TypeError("folds must be a pandas DataFrame or Series")


def positive_class_probability(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(frame)
    classifier = model.named_steps["model"]
    positive_index = len(classifier.classes_) - 1
    return probabilities[:, positive_index]


def _validate_model_inputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    folds: pd.DataFrame,
    target: str,
    id_column: str,
) -> None:
    for name, frame in {"train": train, "test": test, "folds": folds}.items():
        if id_column not in frame.columns:
            raise ValueError(f"{name} is missing id column {id_column!r}")
    if target not in train.columns:
        raise ValueError(f"train is missing target column {target!r}")
    if "fold" not in folds.columns:
        raise ValueError("folds is missing fold column")
    if len(train) != len(folds):
        raise ValueError("train and folds must have the same number of rows")
    if not train[id_column].reset_index(drop=True).equals(folds[id_column].reset_index(drop=True)):
        raise ValueError("train and folds ids must match in order")
