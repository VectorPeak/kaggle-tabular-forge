"""Basic feature-column inference for pandas tabular frames."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def infer_feature_columns(
    train: pd.DataFrame,
    target: str,
    id_column: str,
    extra_exclude: set[str] | None = None,
) -> list[str]:
    """Infer model feature columns by excluding id, target, fold, and any extras."""

    excluded = {target, id_column, "fold"}
    if extra_exclude:
        excluded.update(extra_exclude)
    return [column for column in train.columns if column not in excluded]


def split_columns_by_dtype(frame: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    """Split feature columns into numeric and categorical lists."""

    numeric = [
        column
        for column in columns
        if is_numeric_dtype(frame[column]) and not is_bool_dtype(frame[column])
    ]
    categorical = [column for column in columns if column not in numeric]
    return numeric, categorical
