from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd


def profile_tabular_frames(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    id_column: str,
) -> dict[str, Any]:
    feature_columns = [column for column in train.columns if column not in {id_column, target}]
    numeric_columns = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(train[column])
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    missing_fraction = {
        column: {
            "train": (
                _round_fraction(train[column].isna().mean())
                if column in train.columns
                else None
            ),
            "test": (
                _round_fraction(test[column].isna().mean())
                if column in test.columns
                else None
            ),
        }
        for column in feature_columns
    }
    categorical_cardinality = {
        column: {
            "train": int(train[column].nunique(dropna=True)),
            "test": int(test[column].nunique(dropna=True)) if column in test.columns else 0,
        }
        for column in categorical_columns
    }
    target_distribution = {
        str(key): int(value)
        for key, value in train[target].value_counts(dropna=False).sort_index().items()
    }

    overlap_ids = set(train[id_column].tolist()).intersection(set(test[id_column].tolist()))

    return {
        "row_count": {"train": int(len(train)), "test": int(len(test))},
        "column_count": {"train": int(train.shape[1]), "test": int(test.shape[1])},
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "duplicate_id_count": {
            "train": int(train[id_column].duplicated().sum()),
            "test": int(test[id_column].duplicated().sum()),
        },
        "target_distribution": target_distribution,
        "missing_fraction": missing_fraction,
        "categorical_cardinality": categorical_cardinality,
        "train_test_id_overlap_count": int(len(overlap_ids)),
    }


def build_feature_backlog(profile: dict[str, Any]) -> list[dict[str, Any]]:
    backlog: list[dict[str, Any]] = []
    categorical_columns = [str(column) for column in profile.get("categorical_columns", [])]
    numeric_columns = [str(column) for column in profile.get("numeric_columns", [])]

    for column in categorical_columns:
        backlog.append(
            _backlog_item(
                feature_id=f"freq-{column}",
                feature_family="frequency_count",
                source_columns=[column],
                hypothesis=f"Frequency encoding on {column} may capture stable density effects.",
            )
        )

    for column in numeric_columns:
        backlog.append(
            _backlog_item(
                feature_id=f"bin-{column}",
                feature_family="binning",
                source_columns=[column],
                hypothesis=f"Binning {column} may expose non-linear thresholds.",
            )
        )

    for left, right in combinations(numeric_columns, 2):
        backlog.append(
            _backlog_item(
                feature_id=f"arith-{left}-{right}",
                feature_family="arithmetic_interactions",
                source_columns=[left, right],
                hypothesis=f"Interactions between {left} and {right} may improve separability.",
            )
        )
        break

    return backlog


def _backlog_item(
    *,
    feature_id: str,
    feature_family: str,
    source_columns: list[str],
    hypothesis: str,
) -> dict[str, Any]:
    return {
        "feature_id": feature_id,
        "feature_family": feature_family,
        "hypothesis": hypothesis,
        "source_columns": source_columns,
        "requires_target": False,
        "fold_safety": "global_safe",
        "leakage_risk": "low",
        "transductive_risk": "low",
        "validation_plan": "compare OOF against baseline via existing run/factory workflow",
        "status": "proposed",
        "review_required": False,
        "owner": "eda_scan",
    }


def _round_fraction(value: float) -> float:
    return round(float(value), 6)
