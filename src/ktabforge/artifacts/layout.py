from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    artifact_root: Path
    experiment_dir: Path
    oof_dir: Path
    submission_dir: Path
    registry_dir: Path
    oof_path: Path
    submission_path: Path
    registry_path: Path
    metrics_path: Path
    fold_metrics_path: Path
    config_snapshot_path: Path
    run_manifest_path: Path
    feature_manifest_path: Path
    model_manifest_path: Path
    submission_review_path: Path


def build_artifact_paths(
    artifact_root: str | Path,
    competition: str,
    experiment_id: str,
) -> ArtifactPaths:
    root = Path(artifact_root)
    experiment_dir = root / "experiments" / competition / experiment_id
    oof_dir = root / "oof" / competition / experiment_id
    submission_dir = root / "submissions" / competition / experiment_id
    registry_dir = root / "registry" / competition

    return ArtifactPaths(
        artifact_root=root,
        experiment_dir=experiment_dir,
        oof_dir=oof_dir,
        submission_dir=submission_dir,
        registry_dir=registry_dir,
        oof_path=oof_dir / "oof.parquet",
        submission_path=submission_dir / "submission.csv",
        registry_path=registry_dir / "experiment_registry.csv",
        metrics_path=experiment_dir / "metrics.json",
        fold_metrics_path=experiment_dir / "fold_metrics.csv",
        config_snapshot_path=experiment_dir / "config.yaml",
        run_manifest_path=experiment_dir / "run_manifest.json",
        feature_manifest_path=experiment_dir / "feature_manifest.json",
        model_manifest_path=experiment_dir / "model_manifest.json",
        submission_review_path=experiment_dir / "submission_review.md",
    )


def ensure_new_artifact_layout(paths: ArtifactPaths) -> None:
    existing = [
        path
        for path in (paths.experiment_dir, paths.oof_dir, paths.submission_dir)
        if path.exists()
    ]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Artifact directory already exists for this experiment; refusing to overwrite: "
            f"{joined}"
        )

    paths.experiment_dir.mkdir(parents=True, exist_ok=False)
    paths.oof_dir.mkdir(parents=True, exist_ok=False)
    paths.submission_dir.mkdir(parents=True, exist_ok=False)
    paths.registry_dir.mkdir(parents=True, exist_ok=True)
