from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ktabforge.artifacts.alignment import build_oof_frame, build_submission_frame
from ktabforge.artifacts.layout import build_artifact_paths, ensure_new_artifact_layout
from ktabforge.artifacts.manifests import (
    add_file_checksums,
    build_feature_manifest,
    build_model_manifest,
    build_run_manifest,
)
from ktabforge.artifacts.writers import write_csv, write_json, write_markdown, write_parquet
from ktabforge.pipeline.run_context import RunContext, SmokeEvidenceResult
from ktabforge.registry.experiments import append_experiment_registry
from ktabforge.safety.gates import evaluate_artifact_safety
from ktabforge.safety.leakage import smoke_leakage_review
from ktabforge.utils.hashing import stable_hash


def run_smoke_evidence(
    *,
    data_dir: str | Path,
    artifact_root: str | Path,
    competition: str,
    experiment_id: str,
    target: str,
    id_column: str,
    n_splits: int,
    seed: int,
) -> SmokeEvidenceResult:
    from ktabforge.cv.splitters import build_stratified_folds
    from ktabforge.data.io import load_tabular_frames
    try:
        from ktabforge.data.schema import audit_tabular_frames
    except ImportError:
        from ktabforge.data.io import audit_tabular_frames
    from ktabforge.models.baseline import run_logistic_oof_baseline

    paths = build_artifact_paths(artifact_root, competition, experiment_id)
    ensure_new_artifact_layout(paths)

    context = RunContext(
        data_dir=Path(data_dir),
        artifact_root=Path(artifact_root),
        competition=competition,
        experiment_id=experiment_id,
        target=target,
        id_column=id_column,
        n_splits=n_splits,
        seed=seed,
        paths=paths,
    )

    frames = _normalize_frames(
        _call_with_supported_kwargs(
            load_tabular_frames,
            data_dir=Path(data_dir),
            target=target,
            id_column=id_column,
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
        target=target,
        id_column=id_column,
        frames=frames,
    )
    raw_folds = _call_with_supported_kwargs(
        build_stratified_folds,
        train=train,
        target=target,
        id_column=id_column,
        n_splits=n_splits,
        seed=seed,
    )
    folds = _normalize_folds(raw_folds, len(train))

    baseline_result = _call_with_supported_kwargs(
        run_logistic_oof_baseline,
        train=train,
        test=test,
        target=target,
        id_column=id_column,
        folds=raw_folds,
        fold=folds,
        n_splits=n_splits,
        seed=seed,
    )

    metric_name = str(_extract(baseline_result, "metric_name", default="roc_auc"))
    oof_score = _coerce_float(_extract(baseline_result, "oof_score", default=None))
    oof = _build_oof_from_result(baseline_result, train, id_column, target, folds)
    submission = _build_submission_from_result(baseline_result, test, id_column, target)
    fold_metrics = _normalize_fold_metrics(baseline_result, metric_name, oof_score)

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
        },
    )
    write_csv(paths.fold_metrics_path, fold_metrics)

    feature_manifest = build_feature_manifest(
        train_columns=list(train.columns),
        target=target,
        id_column=id_column,
        schema_audit=schema_audit,
    )
    model_manifest = build_model_manifest(
        model_family="logistic_regression",
        metric_name=metric_name,
        seed=seed,
        n_splits=n_splits,
        baseline_result=baseline_result,
    )
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
    }
    run_manifest = build_run_manifest(
        competition=competition,
        experiment_id=experiment_id,
        data_dir=data_dir,
        target=target,
        id_column=id_column,
        n_splits=n_splits,
        seed=seed,
        status=gate.status,
        paths=artifact_file_paths,
    )
    write_json(paths.run_manifest_path, add_file_checksums(run_manifest, artifact_file_paths))

    append_experiment_registry(
        paths.registry_path,
        {
            "experiment_id": experiment_id,
            "competition": competition,
            "metric_name": metric_name,
            "oof_score": oof_score,
            "status": gate.status,
            "oof_path": str(paths.oof_path),
            "test_pred_path": str(paths.submission_path),
            "fold_metrics_path": str(paths.fold_metrics_path),
            "model_family": "logistic_regression",
            "seed": seed,
            "run_mode": "smoke",
            "reason": gate.reason,
            "feature_manifest_hash": stable_hash(feature_manifest),
        },
    )

    _ = context
    return SmokeEvidenceResult(
        status=gate.status,
        oof_score=oof_score,
        paths=paths,
        metric_name=metric_name,
        reason=gate.reason,
    )


def _call_with_supported_kwargs(function: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(function)
    has_var_keyword = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if has_var_keyword:
        return function(**kwargs)
    supported = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return function(**supported)


def _normalize_frames(value: Any) -> dict[str, pd.DataFrame]:
    sample_submission = None
    if isinstance(value, dict):
        train = value.get("train")
        test = value.get("test")
        sample_submission = value.get("sample_submission")
    elif isinstance(value, tuple):
        train = value[0] if len(value) > 0 else None
        test = value[1] if len(value) > 1 else None
        sample_submission = value[2] if len(value) > 2 else None
    else:
        train = _extract(value, "train", default=None)
        test = _extract(value, "test", default=None)
        sample_submission = _extract(value, "sample_submission", default=None)

    if not isinstance(train, pd.DataFrame) or not isinstance(test, pd.DataFrame):
        raise TypeError(
            "load_tabular_frames must provide pandas DataFrame values for train and test."
        )
    frames = {"train": train, "test": test}
    if isinstance(sample_submission, pd.DataFrame):
        frames["sample_submission"] = sample_submission
    return frames


def _normalize_folds(value: Any, row_count: int) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reset_index(drop=True).astype(int)
    if isinstance(value, pd.DataFrame):
        if "fold" in value.columns:
            return value["fold"].reset_index(drop=True).astype(int)
        if "fold_id" in value.columns:
            return value["fold_id"].reset_index(drop=True).astype(int)
    if isinstance(value, np.ndarray):
        return pd.Series(value, dtype="int64")
    if isinstance(value, list | tuple):
        if len(value) == row_count and all(np.isscalar(item) for item in value):
            return pd.Series(value, dtype="int64")
        folds = pd.Series(np.full(row_count, -1), dtype="int64")
        for fold_id, split in enumerate(value):
            valid_idx = split[1] if isinstance(split, tuple) and len(split) >= 2 else split
            folds.iloc[list(valid_idx)] = fold_id
        return folds
    raise TypeError("build_stratified_folds must return fold ids or split indices.")


def _build_oof_from_result(
    result: Any,
    train: pd.DataFrame,
    id_column: str,
    target: str,
    folds: pd.Series,
) -> pd.DataFrame:
    existing = _extract_any(result, ["oof", "oof_frame", "oof_predictions_frame"], default=None)
    if isinstance(existing, pd.DataFrame):
        if {"id", target, "prediction", "fold"}.issubset(existing.columns):
            return existing[["id", target, "prediction", "fold"]]
        predictions = _extract_prediction_column(existing)
        fold_values = existing["fold"] if "fold" in existing.columns else folds
        return build_oof_frame(
            train=train,
            id_column=id_column,
            target=target,
            predictions=predictions,
            folds=fold_values,
        )

    predictions = _extract_any(
        result,
        ["oof_predictions", "oof_pred", "oof_proba", "train_predictions", "predictions"],
        default=None,
    )
    if predictions is None and isinstance(result, tuple) and len(result) > 0:
        predictions = result[0]
    return build_oof_frame(
        train=train,
        id_column=id_column,
        target=target,
        predictions=_clip_predictions(predictions, len(train), "OOF"),
        folds=folds,
    )


def _build_submission_from_result(
    result: Any,
    test: pd.DataFrame,
    id_column: str,
    target: str,
) -> pd.DataFrame:
    existing = _extract_any(
        result,
        ["submission", "submission_frame", "test_predictions_frame"],
        default=None,
    )
    if isinstance(existing, pd.DataFrame):
        if {"id", target}.issubset(existing.columns):
            return existing[["id", target]]
        predictions = _extract_prediction_column(existing)
        return build_submission_frame(
            test=test,
            id_column=id_column,
            target=target,
            predictions=predictions,
        )

    predictions = _extract_any(
        result,
        ["test_predictions", "test_pred", "test_proba", "submission_predictions"],
        default=None,
    )
    if predictions is None and isinstance(result, tuple) and len(result) > 1:
        predictions = result[1]
    if isinstance(predictions, pd.DataFrame):
        predictions = _extract_prediction_column(predictions)
    return build_submission_frame(
        test=test,
        id_column=id_column,
        target=target,
        predictions=_clip_predictions(predictions, len(test), "test"),
    )


def _normalize_fold_metrics(result: Any, metric_name: str, oof_score: float | None) -> pd.DataFrame:
    fold_metrics = _extract(result, "fold_metrics", default=None)
    if fold_metrics is None and isinstance(result, tuple) and len(result) > 2:
        fold_metrics = result[2]
    if isinstance(fold_metrics, pd.DataFrame):
        return fold_metrics
    if isinstance(fold_metrics, list):
        return pd.DataFrame(fold_metrics)
    return pd.DataFrame([{"fold": "overall", "metric_name": metric_name, "score": oof_score}])


def _extract_prediction_column(frame: pd.DataFrame) -> pd.Series:
    for column in ("prediction", "pred", "probability", "proba", "target"):
        if column in frame.columns:
            return frame[column]
    raise ValueError("Prediction DataFrame must include a prediction-like column.")


def _clip_predictions(predictions: Any, expected_len: int, label: str) -> pd.Series:
    if predictions is None:
        raise ValueError(f"{label} predictions were not returned by baseline runner.")
    series = pd.Series(np.asarray(predictions).reshape(-1))
    if len(series) != expected_len:
        raise ValueError(
            f"{label} prediction length {len(series)} does not match expected "
            f"length {expected_len}."
        )
    return series.astype(float).clip(0.0, 1.0)


def _extract(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _extract_any(value: Any, names: list[str], default: Any) -> Any:
    for name in names:
        result = _extract(value, name, default=None)
        if result is not None:
            return result
    return default


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _submission_review(status: str, reason: str) -> str:
    review = smoke_leakage_review()
    return (
        "# Submission Review\n\n"
        f"- status: {status}\n"
        f"- artifact gate: {reason}\n"
        f"- leakage risk: {review['risk']}\n"
        f"- note: {review['statement']}\n"
        "- kaggle submit: not performed\n"
    )
