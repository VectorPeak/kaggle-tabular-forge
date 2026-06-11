from __future__ import annotations

from typing import Any

import pandas as pd


def build_leakage_watchlist(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    id_column: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    if target in test.columns:
        items.append(
            _watch_item(
                rule="target_present_in_test",
                severity="high",
                message=f"test contains target column {target!r}",
                columns=[target],
            )
        )

    duplicate_train_ids = (
        int(train[id_column].duplicated().sum()) if id_column in train.columns else 0
    )
    if duplicate_train_ids > 0:
        items.append(
            _watch_item(
                rule="duplicate_train_ids",
                severity="medium",
                message=f"train has {duplicate_train_ids} duplicated ids",
                columns=[id_column],
            )
        )

    duplicate_test_ids = int(test[id_column].duplicated().sum()) if id_column in test.columns else 0
    if duplicate_test_ids > 0:
        items.append(
            _watch_item(
                rule="duplicate_test_ids",
                severity="medium",
                message=f"test has {duplicate_test_ids} duplicated ids",
                columns=[id_column],
            )
        )

    if id_column in train.columns and id_column in test.columns:
        overlap_count = int(
            len(set(train[id_column].tolist()).intersection(set(test[id_column].tolist())))
        )
        if overlap_count > 0:
            items.append(
                _watch_item(
                    rule="train_test_id_overlap",
                    severity="high",
                    message=f"train/test id sets overlap for {overlap_count} rows",
                    columns=[id_column],
                )
            )

    feature_columns = [column for column in train.columns if column not in {id_column, target}]
    for column in feature_columns:
        missing_fraction = float(train[column].isna().mean())
        if missing_fraction >= 0.5:
            items.append(
                _watch_item(
                    rule="high_missingness_train",
                    severity="low",
                    message=f"train column {column!r} has missing fraction {missing_fraction:.3f}",
                    columns=[column],
                )
            )

    return items


def _watch_item(
    *,
    rule: str,
    severity: str,
    message: str,
    columns: list[str],
) -> dict[str, Any]:
    return {
        "rule": rule,
        "severity": severity,
        "message": message,
        "columns": columns,
    }
