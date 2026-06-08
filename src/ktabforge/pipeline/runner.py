from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ktabforge.artifacts.layout import build_artifact_paths, ensure_new_artifact_layout
from ktabforge.artifacts.manifests import (
    add_file_checksums,
    build_feature_manifest,
    build_model_manifest,
    build_run_manifest,
)
from ktabforge.artifacts.writers import write_csv, write_json, write_markdown, write_parquet
from ktabforge.config.experiment import ExperimentConfig, load_experiment_config
from ktabforge.pipeline.evidence import (
    _build_oof_from_result,
    _build_submission_from_result,
    _call_with_supported_kwargs,
    _coerce_float,
    _extract,
    _normalize_fold_metrics,
    _normalize_folds,
    _normalize_frames,
    _submission_review,
)
from ktabforge.pipeline.results import ExperimentRunResult
from ktabforge.registry.experiments import append_experiment_registry
from ktabforge.safety.gates import evaluate_artifact_safety
from ktabforge.utils.hashing import stable_hash


def run_experiment_from_config(config_path: str | Path) -> ExperimentRunResult:
    config = load_experiment_config(config_path)
    return run_experiment(config)


def run_experiment(config: ExperimentConfig) -> ExperimentRunResult:
    from ktabforge.cv.splitters import build_stratified_folds
    from ktabforge.data.io import load_tabular_frames

    try:
        from ktabforge.data.schema import audit_tabular_frames
    except ImportError:
        from ktabforge.data.io import audit_tabular_frames

    paths = build_artifact_paths(config.artifact_root, config.competition, config.experiment_id)
    ensure_new_artifact_layout(paths)

    frames = _normalize_frames(
        _call_with_supported_kwargs(
            load_tabular_frames,
            data_dir=config.data_dir,
            target=config.target,
            id_column=config.id_column,
        )
    )
    train = frames["train"]
    test = frames["test"]
    sample_submission = frames.get("sample_submission")

    schema_audit = _call_with_supported_kwargs(
        audit_tabular_frames,
        train=train,
        test=test,
        sample_submission=sample_submission,
        target=config.target,
        id_column=config.id_column,
        frames=frames,
    )
    raw_folds = _call_with_supported_kwargs(
        build_stratified_folds,
        train=train,
        target=config.target,
        id_column=config.id_column,
        n_splits=config.n_splits,
        seed=config.seed,
    )
    folds = _normalize_folds(raw_folds, len(train))

    model_result = _run_configured_model(
        config=config,
        train=train,
        test=test,
        raw_folds=raw_folds,
        folds=folds,
    )

    metric_name = str(_extract(model_result, "metric_name", default="roc_auc"))
    oof_score = _coerce_float(_extract(model_result, "oof_score", default=None))
    oof = _build_oof_from_result(model_result, train, config.id_column, config.target, folds)
    submission = _build_submission_from_result(model_result, test, config.id_column, config.target)
    fold_metrics = _normalize_fold_metrics(model_result, metric_name, oof_score)
    gate = evaluate_artifact_safety(train=train, test=test, oof=oof, submission=submission)

    write_parquet(paths.oof_path, oof)
    write_csv(paths.submission_path, submission)
    write_json(
        paths.metrics_path,
        {
            "metric_name": metric_name,
            "oof_score": oof_score,
            "status": gate.status,
            "reason": gate.reason,
            "config_hash": config.config_hash,
        },
    )
    write_csv(paths.fold_metrics_path, fold_metrics)
    _write_config_snapshot(paths.experiment_dir / "config.yaml", config.raw)

    feature_manifest = build_feature_manifest(
        train_columns=list(train.columns),
        target=config.target,
        id_column=config.id_column,
        schema_audit=schema_audit,
    )
    feature_manifest["feature_set"] = config.feature_set
    feature_manifest["feature_families"] = config.feature_families

    model_manifest = build_model_manifest(
        model_family=config.model_family,
        metric_name=metric_name,
        seed=config.seed,
        n_splits=config.n_splits,
        baseline_result=model_result,
    )
    model_manifest["model_preset"] = config.model_preset
    model_manifest["model_params"] = config.model_params

    write_json(paths.feature_manifest_path, feature_manifest)
    write_json(paths.model_manifest_path, model_manifest)
    write_markdown(paths.submission_review_path, _submission_review(gate.status, gate.reason))

    artifact_file_paths = {
        "oof_path": str(paths.oof_path),
        "submission_path": str(paths.submission_path),
        "metrics_path": str(paths.metrics_path),
        "fold_metrics_path": str(paths.fold_metrics_path),
        "feature_manifest_path": str(paths.feature_manifest_path),
        "model_manifest_path": str(paths.model_manifest_path),
        "config_path": str(paths.experiment_dir / "config.yaml"),
    }
    run_manifest = build_run_manifest(
        competition=config.competition,
        experiment_id=config.experiment_id,
        data_dir=config.data_dir,
        target=config.target,
        id_column=config.id_column,
        n_splits=config.n_splits,
        seed=config.seed,
        status=gate.status,
        paths=artifact_file_paths,
    )
    run_manifest["run_mode"] = config.run_mode
    run_manifest["config_hash"] = config.config_hash
    write_json(paths.run_manifest_path, add_file_checksums(run_manifest, artifact_file_paths))

    registry_extras = {
        "model_preset": config.model_preset,
        "feature_set": config.feature_set,
        "config_hash": config.config_hash,
    }
    append_experiment_registry(
        paths.registry_path,
        {
            "experiment_id": config.experiment_id,
            "competition": config.competition,
            "metric_name": metric_name,
            "oof_score": oof_score,
            "status": gate.status,
            "oof_path": str(paths.oof_path),
            "test_pred_path": str(paths.submission_path),
            "fold_metrics_path": str(paths.fold_metrics_path),
            "model_family": config.model_family,
            "model_preset": config.model_preset,
            "feature_set": config.feature_set,
            "config_hash": config.config_hash,
            "seed": config.seed,
            "run_mode": config.run_mode,
            "reason": gate.reason,
            "feature_manifest_hash": stable_hash(feature_manifest),
        },
    )
    _ensure_registry_extra_columns(paths.registry_path, config.experiment_id, registry_extras)

    return ExperimentRunResult(
        status=gate.status,
        oof_score=oof_score,
        paths=paths,
        metric_name=metric_name,
        reason=gate.reason,
    )


def _run_configured_model(
    *,
    config: ExperimentConfig,
    train: pd.DataFrame,
    test: pd.DataFrame,
    raw_folds: Any,
    folds: pd.Series,
) -> Any:
    runner = _load_model_runner(config.model_family)
    return _call_with_supported_kwargs(
        runner,
        model_family=config.model_family,
        family=config.model_family,
        model_preset=config.model_preset,
        preset=config.model_preset,
        model_params=config.model_params,
        params=config.model_params,
        train=train,
        test=test,
        target=config.target,
        id_column=config.id_column,
        folds=raw_folds,
        fold=folds,
        n_splits=config.n_splits,
        seed=config.seed,
        feature_set=config.feature_set,
        feature_families=config.feature_families,
        run_mode=config.run_mode,
    )


def _load_model_runner(model_family: str) -> Any:
    try:
        from ktabforge.models.registry import run_model_oof

        return run_model_oof
    except ImportError:
        if model_family == "logistic_regression":
            from ktabforge.models.baseline import run_logistic_oof_baseline

            return run_logistic_oof_baseline
        raise


def _write_config_snapshot(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _ensure_registry_extra_columns(
    registry_path: Path,
    experiment_id: str,
    extras: dict[str, Any],
) -> None:
    if not registry_path.exists():
        return
    frame = pd.read_csv(registry_path)
    if frame.empty or "experiment_id" not in frame.columns:
        return
    for column, value in extras.items():
        if column not in frame.columns:
            frame[column] = None
        frame.loc[frame["experiment_id"] == experiment_id, column] = value
    frame.to_csv(registry_path, index=False)
