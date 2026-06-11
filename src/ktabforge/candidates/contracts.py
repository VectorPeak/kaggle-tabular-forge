from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PredictionArtifactMeta:
    competition: str
    metric_name: str
    prediction_type: str
    target: str
    id_column: str
    oof_row_count: int
    test_row_count: int
    oof_checksum: str | None = None
    test_checksum: str | None = None


@dataclass(frozen=True)
class CVProtocolMeta:
    cv_protocol_id: str
    splitter: str
    fold_count: int
    seed: int | None
    oof_safe: bool


@dataclass(frozen=True)
class CandidateRecord:
    experiment_id: str
    model_family: str | None
    oof_score: float | None
    row: dict[str, Any]
    prediction_meta: PredictionArtifactMeta
    cv_protocol: CVProtocolMeta
    oof: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class CandidateCompatibilityRejection:
    experiment_id: str
    reason: str


@dataclass(frozen=True)
class CandidateCompatibilityResult:
    accepted: list[CandidateRecord]
    rejected: list[CandidateCompatibilityRejection]

