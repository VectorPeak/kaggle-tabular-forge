from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Candidate:
    experiment_id: str
    row: dict[str, Any]
    oof_path: Path
    test_pred_path: Path
    oof_score: float | None
    model_family: str | None
    prediction_type: str


@dataclass(frozen=True)
class RejectedCandidate:
    experiment_id: str
    reason: str


@dataclass(frozen=True)
class CandidatePool:
    candidates: list[Candidate]
    rejected: list[RejectedCandidate]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([candidate.row for candidate in self.candidates])


def build_candidate_pool(
    *,
    artifact_root: str | Path,
    competition: str,
    metric_name: str | None = None,
    candidate_ids: list[str] | None = None,
    top_n: int | None = None,
) -> CandidatePool:
    root = Path(artifact_root)
    registry_path = root / "registry" / competition / "experiment_registry.csv"
    if not registry_path.exists():
        return CandidatePool(candidates=[], rejected=[])

    registry = pd.read_csv(registry_path)
    allowed_ids = set(candidate_ids or [])
    candidates: list[Candidate] = []
    rejected: list[RejectedCandidate] = []

    for row in registry.to_dict(orient="records"):
        experiment_id = str(row.get("experiment_id", ""))
        if allowed_ids and experiment_id not in allowed_ids:
            continue

        reason = _rejection_reason(row, root, competition, metric_name)
        if reason:
            rejected.append(RejectedCandidate(experiment_id=experiment_id, reason=reason))
            continue

        oof_path = _resolve_path(row["oof_path"], root)
        test_pred_path = _resolve_path(row["test_pred_path"], root)
        candidates.append(
            Candidate(
                experiment_id=experiment_id,
                row=dict(row),
                oof_path=oof_path,
                test_pred_path=test_pred_path,
                oof_score=_optional_float(row.get("oof_score")),
                model_family=_optional_string(row.get("model_family")),
                prediction_type=str(row.get("prediction_type") or "probability"),
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.oof_score is not None,
            candidate.oof_score if candidate.oof_score is not None else float("-inf"),
        ),
        reverse=True,
    )
    if top_n is not None:
        candidates = candidates[:top_n]
    return CandidatePool(candidates=candidates, rejected=rejected)


def _rejection_reason(
    row: dict[str, Any],
    artifact_root: Path,
    competition: str,
    metric_name: str | None,
) -> str | None:
    row_competition = str(row.get("competition", ""))
    if row_competition and row_competition != competition:
        return "competition does not match"

    if str(row.get("status", "")).lower() != "completed":
        return "status is not completed"

    if metric_name and str(row.get("metric_name", "")) != metric_name:
        return "metric_name does not match"

    if not _path_exists(row.get("oof_path"), artifact_root):
        return "oof_path does not exist"

    if not _path_exists(row.get("test_pred_path"), artifact_root):
        return "test_pred_path does not exist"

    prediction_type = str(row.get("prediction_type") or "probability")
    if prediction_type != "probability":
        return "prediction_type is not probability"

    leakage_risk = str(row.get("leakage_risk") or "low").lower()
    review_status = str(row.get("review_status") or "").lower()
    if leakage_risk in {"high", "blocker"} and review_status not in {"approved", "reviewed"}:
        return "leakage risk is not approved"

    return None


def _path_exists(value: object, artifact_root: Path) -> bool:
    if pd.isna(value):
        return False
    return _resolve_path(value, artifact_root).exists()


def _resolve_path(value: object, artifact_root: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = [
        artifact_root / path,
        artifact_root.parent / path,
        path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return artifact_root / path


def _optional_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _optional_string(value: object) -> str | None:
    if pd.isna(value):
        return None
    return str(value)
