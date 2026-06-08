from pathlib import Path

import pandas as pd

from ktabforge.cv.splitters import build_stratified_folds


def test_stratified_folds_are_reproducible_and_cover_each_row_once():
    data_dir = Path(__file__).resolve().parents[1] / "fixtures" / "data" / "churn_tiny"
    train = pd.read_csv(data_dir / "train.csv")

    first = build_stratified_folds(train, target="Churn", n_splits=3, seed=42)
    second = build_stratified_folds(train, target="Churn", n_splits=3, seed=42)

    assert first["fold"].tolist() == second["fold"].tolist()
    assert first["id"].tolist() == train["id"].tolist()
    assert first["fold"].between(0, 2).all()
    assert first["fold"].notna().all()
    assert len(first) == len(train)

