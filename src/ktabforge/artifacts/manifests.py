from __future__ import annotations

from pathlib import Path
from typing import Any

from ktabforge.artifacts.checksums import sha256_file
from ktabforge.utils.env import collect_environment
from ktabforge.utils.git import collect_git_info
from ktabforge.utils.time import utc_now_iso


def build_run_manifest(
    *,
    competition: str,
    experiment_id: str,
    data_dir: str | Path,
    target: str,
    id_column: str,
    n_splits: int,
    seed: int,
    status: str,
    paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "competition": competition,
        "experiment_id": experiment_id,
        "created_at": utc_now_iso(),
        "data_dir": str(data_dir),
        "target": target,
        "id_column": id_column,
        "n_splits": n_splits,
        "seed": seed,
        "status": status,
        "paths": paths,
        "environment": collect_environment(),
        "git": collect_git_info(),
    }


def build_feature_manifest(
    *,
    train_columns: list[str],
    target: str,
    id_column: str,
    schema_audit: Any = None,
) -> dict[str, Any]:
    excluded = {target, id_column}
    features = [column for column in train_columns if column not in excluded]
    return {
        "id_column": id_column,
        "target": target,
        "features": features,
        "feature_count": len(features),
        "schema_audit": _jsonable(schema_audit),
    }


def build_model_manifest(
    *,
    model_family: str,
    metric_name: str,
    seed: int,
    n_splits: int,
    baseline_result: Any = None,
) -> dict[str, Any]:
    return {
        "model_family": model_family,
        "metric_name": metric_name,
        "seed": seed,
        "n_splits": n_splits,
        "baseline_result_type": (
            type(baseline_result).__name__ if baseline_result is not None else None
        ),
    }


def add_file_checksums(manifest: dict[str, Any], paths: dict[str, str]) -> dict[str, Any]:
    manifest = dict(manifest)
    manifest["checksums"] = {
        name: sha256_file(path)
        for name, path in paths.items()
        if path and Path(path).exists() and Path(path).is_file()
    }
    return manifest


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)
