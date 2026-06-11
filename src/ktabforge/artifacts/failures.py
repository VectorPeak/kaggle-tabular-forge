from __future__ import annotations

from pathlib import Path
from typing import Any

from ktabforge.artifacts.writers import write_json
from ktabforge.metrics.scoring import metric_mode_or_none
from ktabforge.registry.experiments import append_experiment_registry
from ktabforge.utils.time import utc_now_iso


def record_failure(
    *,
    artifact_root: str | Path,
    competition: str,
    experiment_id: str,
    reason: str,
    run_mode: str,
    config_path: str | Path | None = None,
    config_hash: str | None = None,
    metric_name: str | None = None,
    model_family: str | None = None,
    model_preset: str | None = None,
    feature_set: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    root = Path(artifact_root)
    record_path = root / "failures" / competition / f"{experiment_id}.json"
    payload = {
        "competition": competition,
        "experiment_id": experiment_id,
        "status": "failed",
        "reason": reason,
        "run_mode": run_mode,
        "metric_name": metric_name,
        "config_path": str(config_path) if config_path is not None else None,
        "config_hash": config_hash,
        "model_family": model_family,
        "model_preset": model_preset,
        "feature_set": feature_set,
        "created_at": utc_now_iso(),
    }
    if extra:
        payload["extra"] = extra
    write_json(record_path, payload)
    append_experiment_registry(
        root / "registry" / competition / "experiment_registry.csv",
        {
            "experiment_id": experiment_id,
            "competition": competition,
            "metric_name": metric_name,
            "metric_mode": metric_mode_or_none(metric_name),
            "oof_score": None,
            "status": "failed",
            "oof_path": None,
            "test_pred_path": None,
            "fold_metrics_path": None,
            "model_family": model_family,
            "seed": None,
            "run_mode": run_mode,
            "reason": reason,
            "model_preset": model_preset,
            "feature_set": feature_set,
            "config_hash": config_hash,
            "created_at": utc_now_iso(),
            "feature_manifest_hash": None,
            "failure_record_path": str(record_path),
        },
    )
    return record_path
