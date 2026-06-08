"""Cross-validation split builders."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def build_stratified_folds(
    train: pd.DataFrame,
    target: str,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a copy of train with a deterministic integer fold assignment."""

    if target not in train.columns:
        raise ValueError(f"Target column {target!r} is not present in train")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(train) < n_splits:
        raise ValueError("n_splits cannot exceed the number of training rows")

    folds = train.copy()
    folds["fold"] = -1

    y = train[target]
    value_counts = y.value_counts(dropna=False)
    can_stratify = y.nunique(dropna=False) > 1 and int(value_counts.min()) >= n_splits

    splitter = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if can_stratify
        else KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )

    split_iter = splitter.split(train, y) if can_stratify else splitter.split(train)
    for fold, (_, valid_idx) in enumerate(split_iter):
        folds.iloc[valid_idx, folds.columns.get_loc("fold")] = fold

    folds["fold"] = folds["fold"].astype(int)
    return folds
