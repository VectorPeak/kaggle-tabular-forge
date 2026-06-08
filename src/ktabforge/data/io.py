"""I/O helpers for standard Kaggle tabular data layouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TabularFrames:
    """Standard train/test/submission frame bundle."""

    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame


def load_tabular_frames(data_dir: str | Path, target: str, id_column: str) -> TabularFrames:
    """Read train.csv, test.csv, and sample_submission.csv from a data directory."""

    root = Path(data_dir)
    train = pd.read_csv(root / "train.csv")
    test = pd.read_csv(root / "test.csv")
    sample_submission = pd.read_csv(root / "sample_submission.csv")

    audit = audit_tabular_frames(train, test, sample_submission, target, id_column)
    if not audit["ok"]:
        problems = "; ".join(audit["errors"])
        raise ValueError(f"Invalid tabular data layout: {problems}")

    return TabularFrames(train=train, test=test, sample_submission=sample_submission)


def audit_tabular_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    target: str,
    id_column: str,
) -> dict[str, Any]:
    """Return lightweight checks for a standard train/test/submission triplet."""

    errors: list[str] = []
    warnings: list[str] = []

    for name, frame in {
        "train": train,
        "test": test,
        "sample_submission": sample_submission,
    }.items():
        if id_column not in frame.columns:
            errors.append(f"{name} is missing id column {id_column!r}")

    if target not in train.columns:
        errors.append(f"train is missing target column {target!r}")
    if target in test.columns:
        warnings.append(f"test contains target column {target!r}")
    if target not in sample_submission.columns:
        errors.append(f"sample_submission is missing target column {target!r}")

    if id_column in test.columns and id_column in sample_submission.columns:
        if len(test) != len(sample_submission):
            errors.append("test and sample_submission row counts differ")
        elif not test[id_column].reset_index(drop=True).equals(
            sample_submission[id_column].reset_index(drop=True)
        ):
            errors.append("sample_submission ids do not match test ids in order")

    for name, frame in {"train": train, "test": test}.items():
        if id_column in frame.columns and frame[id_column].duplicated().any():
            warnings.append(f"{name} has duplicate ids")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "id": {
            "column": id_column,
            "train_present": id_column in train.columns,
            "test_present": id_column in test.columns,
            "sample_submission_present": id_column in sample_submission.columns,
        },
        "target": {
            "column": target,
            "train_present": target in train.columns,
            "test_present": target in test.columns,
            "sample_submission_present": target in sample_submission.columns,
        },
        "row_count": {
            "train": len(train),
            "test": len(test),
            "sample_submission": len(sample_submission),
        },
        "sample_submission": {
            "columns": list(sample_submission.columns),
            "matches_test_ids": not errors
            and id_column in test.columns
            and id_column in sample_submission.columns,
        },
    }
