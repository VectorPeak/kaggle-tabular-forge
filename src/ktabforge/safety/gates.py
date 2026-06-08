from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SafetyGateResult:
    status: str
    reason: str


def evaluate_artifact_safety(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    oof: pd.DataFrame,
    submission: pd.DataFrame,
) -> SafetyGateResult:
    if len(oof) != len(train):
        return SafetyGateResult(
            status="failed",
            reason=f"OOF row count {len(oof)} does not match train row count {len(train)}.",
        )
    if len(submission) != len(test):
        return SafetyGateResult(
            status="failed",
            reason=(
                f"Submission row count {len(submission)} does not match test row count {len(test)}."
            ),
        )
    return SafetyGateResult(status="completed", reason="OOF and submission row counts match.")
