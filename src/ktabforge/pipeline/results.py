from __future__ import annotations

from dataclasses import dataclass

from ktabforge.artifacts.layout import ArtifactPaths


@dataclass(frozen=True)
class ExperimentRunResult:
    status: str
    oof_score: float | None
    paths: ArtifactPaths
    metric_name: str = "roc_auc"
    reason: str = ""
