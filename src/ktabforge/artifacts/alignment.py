from __future__ import annotations

import pandas as pd


def build_oof_frame(
    *,
    train: pd.DataFrame,
    id_column: str,
    target: str,
    predictions: pd.Series | list[float],
    folds: pd.Series | list[int],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": train[id_column].to_numpy(),
            target: train[target].to_numpy(),
            "prediction": predictions,
            "fold": folds,
        }
    )


def build_submission_frame(
    *,
    test: pd.DataFrame,
    id_column: str,
    target: str,
    predictions: pd.Series | list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": test[id_column].to_numpy(),
            target: predictions,
        }
    )


def oof_row_count_matches(oof: pd.DataFrame, train: pd.DataFrame) -> bool:
    return len(oof) == len(train)


def submission_row_count_matches(submission: pd.DataFrame, test: pd.DataFrame) -> bool:
    return len(submission) == len(test)
