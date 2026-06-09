from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from ktabforge.artifacts.layout import build_artifact_paths, ensure_new_artifact_layout
from ktabforge.artifacts.writers import write_csv, write_json, write_markdown, write_parquet
from ktabforge.candidates.alignment import CandidatePredictions, load_aligned_predictions
from ktabforge.candidates.pool import CandidatePool, build_candidate_pool
from ktabforge.ensembles.averaging import average_predictions
from ktabforge.ensembles.config import EnsembleConfig, load_ensemble_config
from ktabforge.metrics.scoring import safe_roc_auc_score
from ktabforge.registry.experiments import append_experiment_registry
from ktabforge.utils.hashing import stable_hash


@dataclass(frozen=True)
class EnsembleRunResult:
    status: str
    oof_score: float
    experiment_id: str
    registry_path: Path
    oof_path: Path
    submission_path: Path


def run_ensemble_from_config(config_path: str | Path) -> EnsembleRunResult:
    config = load_ensemble_config(config_path)
    return run_ensemble(config)


def run_ensemble(config: EnsembleConfig) -> EnsembleRunResult:
    pool = build_candidate_pool(
        artifact_root=config.artifact_root,
        competition=config.competition,
        metric_name=config.metric_name,
        candidate_ids=config.candidate_ids,
        top_n=config.top_n,
    )
    if not pool.candidates:
        raise ValueError("No eligible ensemble candidates found")
    _ensure_requested_candidates_eligible(pool, config.candidate_ids)

    loaded = load_aligned_predictions(pool.candidates, target=config.target)
    oof = _build_ensemble_oof(loaded, config=config)
    submission = _build_ensemble_submission(loaded, config=config)
    fold_metrics = _build_fold_metrics(oof, metric_name=config.metric_name, target=config.target)
    oof_score = float(safe_roc_auc_score(oof[config.target], oof["prediction"]))

    paths = build_artifact_paths(
        config.artifact_root,
        config.competition,
        config.experiment_id,
    )
    ensure_new_artifact_layout(paths)

    write_parquet(paths.oof_path, oof)
    write_csv(paths.submission_path, submission)
    write_csv(paths.fold_metrics_path, fold_metrics)
    write_json(
        paths.metrics_path,
        {
            "metric_name": config.metric_name,
            "oof_score": oof_score,
            "status": "completed",
            "reason": "ensemble completed",
        },
    )
    paths.config_snapshot_path.write_text(
        yaml.safe_dump(config.raw, sort_keys=False),
        encoding="utf-8",
    )

    candidate_pool_path = paths.experiment_dir / "candidate_pool.parquet"
    ensemble_manifest_path = paths.experiment_dir / "ensemble_manifest.json"
    selection_report_path = paths.experiment_dir / "selection_report.md"
    run_manifest_path = paths.run_manifest_path

    write_parquet(candidate_pool_path, pool.to_frame())
    write_json(ensemble_manifest_path, _ensemble_manifest(config, pool, oof_score))
    write_markdown(selection_report_path, _selection_report(pool))
    write_json(
        run_manifest_path,
        {
            "competition": config.competition,
            "experiment_id": config.experiment_id,
            "run_mode": "ensemble",
            "method": config.method,
            "parent_experiment_ids": [candidate.experiment_id for candidate in pool.candidates],
            "status": "completed",
        },
    )

    append_experiment_registry(
        paths.registry_path,
        {
            "experiment_id": config.experiment_id,
            "competition": config.competition,
            "metric_name": config.metric_name,
            "oof_score": oof_score,
            "status": "completed",
            "oof_path": str(paths.oof_path),
            "test_pred_path": str(paths.submission_path),
            "fold_metrics_path": str(paths.fold_metrics_path),
            "model_family": "ensemble",
            "model_preset": config.method,
            "feature_set": "ensemble",
            "config_hash": stable_hash(config.raw),
            "seed": None,
            "run_mode": "ensemble",
            "reason": "ensemble completed",
            "feature_manifest_hash": None,
            "prediction_type": "probability",
            "parent_experiment_ids": ",".join(
                candidate.experiment_id for candidate in pool.candidates
            ),
            "parent_count": len(pool.candidates),
            "ensemble_recipe": config.method,
            "candidate_pool_path": str(candidate_pool_path),
            "ensemble_manifest_path": str(ensemble_manifest_path),
            "selection_report_path": str(selection_report_path),
        },
    )

    return EnsembleRunResult(
        status="completed",
        oof_score=oof_score,
        experiment_id=config.experiment_id,
        registry_path=paths.registry_path,
        oof_path=paths.oof_path,
        submission_path=paths.submission_path,
    )


def _ensure_requested_candidates_eligible(pool: CandidatePool, requested_ids: list[str]) -> None:
    if not requested_ids:
        return
    accepted_ids = {candidate.experiment_id for candidate in pool.candidates}
    missing_or_rejected = [
        experiment_id for experiment_id in requested_ids if experiment_id not in accepted_ids
    ]
    if missing_or_rejected:
        joined = ", ".join(missing_or_rejected)
        raise ValueError(f"requested candidates are not eligible: {joined}")


def _build_ensemble_oof(
    loaded: list[CandidatePredictions],
    *,
    config: EnsembleConfig,
) -> pd.DataFrame:
    reference = loaded[0].oof
    predictions = [item.oof["prediction"] for item in loaded]
    experiment_ids = [item.candidate.experiment_id for item in loaded]
    averaged = average_predictions(
        predictions,
        method=config.method,
        experiment_ids=experiment_ids,
        weights=config.weights,
    )
    frame = pd.DataFrame(
        {
            "id": reference["id"].to_numpy(),
            config.target: reference[config.target].to_numpy(),
            "prediction": averaged.to_numpy(),
        }
    )
    if "fold" in reference.columns:
        frame["fold"] = reference["fold"].to_numpy()
    return frame


def _build_ensemble_submission(
    loaded: list[CandidatePredictions],
    *,
    config: EnsembleConfig,
) -> pd.DataFrame:
    reference = loaded[0].submission
    predictions = [item.submission[config.target] for item in loaded]
    experiment_ids = [item.candidate.experiment_id for item in loaded]
    averaged = average_predictions(
        predictions,
        method=config.method,
        experiment_ids=experiment_ids,
        weights=config.weights,
    )
    return pd.DataFrame({"id": reference["id"].to_numpy(), config.target: averaged.to_numpy()})


def _build_fold_metrics(
    oof: pd.DataFrame,
    *,
    metric_name: str,
    target: str,
) -> pd.DataFrame:
    if "fold" not in oof.columns:
        return pd.DataFrame(columns=["fold", metric_name, "rows"])
    rows = []
    for fold, frame in oof.groupby("fold", sort=True):
        rows.append(
            {
                "fold": fold,
                metric_name: safe_roc_auc_score(frame[target], frame["prediction"]),
                "rows": len(frame),
            }
        )
    return pd.DataFrame(rows)


def _ensemble_manifest(
    config: EnsembleConfig,
    pool: CandidatePool,
    oof_score: float,
) -> dict[str, object]:
    return {
        "experiment_id": config.experiment_id,
        "competition": config.competition,
        "method": config.method,
        "metric_name": config.metric_name,
        "oof_score": oof_score,
        "parents": [candidate.experiment_id for candidate in pool.candidates],
        "weights": config.weights,
        "rejected": [
            {"experiment_id": item.experiment_id, "reason": item.reason}
            for item in pool.rejected
        ],
    }


def _selection_report(pool: CandidatePool) -> str:
    accepted = "\n".join(f"- {candidate.experiment_id}" for candidate in pool.candidates)
    rejected = "\n".join(
        f"- {candidate.experiment_id}: {candidate.reason}" for candidate in pool.rejected
    )
    return (
        "# Candidate Selection Report\n\n"
        "## Accepted\n\n"
        f"{accepted or '- None'}\n\n"
        "## Rejected\n\n"
        f"{rejected or '- None'}\n"
    )
