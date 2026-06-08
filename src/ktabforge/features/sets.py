"""Feature-set resolution for config-driven runs."""

from __future__ import annotations

import pandas as pd

from ktabforge.features.basic import infer_feature_columns, split_columns_by_dtype

FEATURE_SET_ALIASES = {
    "basic": "basic_inferred",
    "basic_inferred": "basic_inferred",
}


def resolve_feature_set(
    feature_set: str,
    train: pd.DataFrame,
    target: str,
    id_column: str,
) -> dict[str, object]:
    """Resolve a lightweight feature-set manifest for tabular smoke runs."""

    canonical = _normalize_feature_set(feature_set)
    columns = infer_feature_columns(train, target=target, id_column=id_column)
    numeric_columns, categorical_columns = split_columns_by_dtype(train, columns)
    return {
        "feature_set": canonical,
        "families": ["basic"],
        "columns": columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "excluded_columns": [id_column, target, "fold"],
    }


def _normalize_feature_set(feature_set: str) -> str:
    key = feature_set.strip().lower()
    try:
        return FEATURE_SET_ALIASES[key]
    except KeyError as exc:
        known = ", ".join(sorted(FEATURE_SET_ALIASES))
        raise ValueError(
            f"Unknown feature set {feature_set!r}. Known feature sets: {known}."
        ) from exc
