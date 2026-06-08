from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ktabforge.artifacts.layout import ArtifactPaths


@dataclass(frozen=True)
class SmokeEvidenceResult:
    status: str
    oof_score: float | None
    paths: ArtifactPaths
    metric_name: str = "roc_auc"
    reason: str = ""


@dataclass(frozen=True)
class RunContext:
    data_dir: Path
    artifact_root: Path
    competition: str
    experiment_id: str
    target: str
    id_column: str
    n_splits: int
    seed: int
    paths: ArtifactPaths
