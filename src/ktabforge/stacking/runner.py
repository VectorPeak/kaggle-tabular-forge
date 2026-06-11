from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from ktabforge.artifacts.layout import build_artifact_paths, ensure_new_artifact_layout
from ktabforge.artifacts.manifests import add_file_checksums, build_run_manifest
from ktabforge.artifacts.writers import write_csv, write_json, write_markdown, write_parquet
from ktabforge.candidates import (
    CandidateCompatibilityRejection,
    CandidateRecord,
    CVProtocolMeta,
    PredictionArtifactMeta,
    evaluate_candidate_compatibility,
)
from ktabforge.candidates.pool import CandidatePool, build_candidate_pool
from ktabforge.metrics.scoring import metric_mode, score_predictions
from ktabforge.registry.experiments import append_experiment_registry
from ktabforge.safety.gates import evaluate_artifact_safety
from ktabforge.stacking.config import StackingPreflightConfig, load_stacking_config
from ktabforge.utils.hashing import stable_hash


@dataclass(frozen=True)
class StackingPreflightResult:
    status: str
    experiment_id: str
    accepted_count: int
    rejected_count: int
    selection_report_path: Path
    stacking_manifest_path: Path
    stack_oof_path: Path
    stack_test_path: Path


@dataclass(frozen=True)
class StackRunResult:
    status: str
    experiment_id: str
    oof_score: float
    metric_name: str
    oof_path: Path
    submission_path: Path
    stack_oof_path: Path
    stack_test_path: Path
    selection_report_path: Path
    stacking_manifest_path: Path
    run_manifest_path: Path


@dataclass(frozen=True)
class _SelectedCandidates:
    accepted: list[CandidateRecord]
    rejected: list[CandidateCompatibilityRejection]


@dataclass(frozen=True)
class _TrainedStack:
    oof: pd.DataFrame
    submission: pd.DataFrame
    fold_metrics: pd.DataFrame
    oof_score: float


def run_stacking_preflight(config_path: str | Path) -> StackingPreflightResult:
    config = load_stacking_config(config_path)
    return run_preflight(config)


def run_stack_from_config(config_path: str | Path) -> StackRunResult:
    config = load_stacking_config(config_path)
    return run_stack(config)


def run_preflight(config: StackingPreflightConfig) -> StackingPreflightResult:
    selection = _select_candidates(config)
    paths = build_artifact_paths(
        config.artifact_root,
        config.competition,
        config.experiment_id,
    )
    ensure_new_artifact_layout(paths)

    stack_oof = _build_stack_oof(
        selection.accepted,
        target=config.target,
        id_column=config.id_column,
    )
    stack_test = _build_stack_test(selection.accepted, id_column=config.id_column)
    stack_oof_path = paths.oof_dir / "stack_oof.parquet"
    stack_test_path = paths.submission_dir / "stack_test.parquet"
    selection_report_path = paths.experiment_dir / "selection_report.md"
    stacking_manifest_path = paths.experiment_dir / "stacking_manifest.json"

    write_parquet(stack_oof_path, stack_oof)
    write_parquet(stack_test_path, stack_test)
    _write_config_snapshot(paths.config_snapshot_path, config.raw)
    write_markdown(
        selection_report_path,
        _selection_report(
            accepted=selection.accepted,
            rejected=selection.rejected,
            config=config,
        ),
    )
    write_json(
        stacking_manifest_path,
        _stacking_manifest(
            config=config,
            accepted=selection.accepted,
            rejected=selection.rejected,
            stack_oof_path=stack_oof_path,
            stack_test_path=stack_test_path,
            stack_oof=stack_oof,
            stack_test=stack_test,
            status="prepared",
        ),
    )

    return StackingPreflightResult(
        status="prepared",
        experiment_id=config.experiment_id,
        accepted_count=len(selection.accepted),
        rejected_count=len(selection.rejected),
        selection_report_path=selection_report_path,
        stacking_manifest_path=stacking_manifest_path,
        stack_oof_path=stack_oof_path,
        stack_test_path=stack_test_path,
    )


def run_stack(config: StackingPreflightConfig) -> StackRunResult:
    if config.stacker_method == "preflight_only":
        raise ValueError("stacker.method=preflight_only cannot produce a completed stack run")

    selection = _select_candidates(config)
    stacker_seed = _resolved_stacker_seed(config)
    paths = build_artifact_paths(
        config.artifact_root,
        config.competition,
        config.experiment_id,
    )
    ensure_new_artifact_layout(paths)

    stack_oof = _build_stack_oof(
        selection.accepted,
        target=config.target,
        id_column=config.id_column,
    )
    stack_test = _build_stack_test(selection.accepted, id_column=config.id_column)
    trained = _train_oof_safe_stack(stack_oof=stack_oof, stack_test=stack_test, config=config)
    gate = evaluate_artifact_safety(
        train=stack_oof[[config.id_column, config.target]],
        test=stack_test[[config.id_column]],
        oof=trained.oof,
        submission=trained.submission,
    )

    stack_oof_path = paths.oof_dir / "stack_oof.parquet"
    stack_test_path = paths.submission_dir / "stack_test.parquet"
    selection_report_path = paths.experiment_dir / "selection_report.md"
    stacking_manifest_path = paths.experiment_dir / "stacking_manifest.json"

    write_parquet(stack_oof_path, stack_oof)
    write_parquet(stack_test_path, stack_test)
    write_parquet(paths.oof_path, trained.oof)
    write_csv(paths.submission_path, trained.submission)
    write_csv(paths.fold_metrics_path, trained.fold_metrics)
    write_json(
        paths.metrics_path,
        {
            "metric_name": config.metric_name,
            "metric_mode": metric_mode(config.metric_name),
            "oof_score": trained.oof_score,
            "status": gate.status,
            "reason": gate.reason,
            "stacker_method": config.stacker_method,
            "config_hash": stable_hash(config.raw),
        },
    )
    _write_config_snapshot(paths.config_snapshot_path, config.raw)
    write_markdown(
        selection_report_path,
        _selection_report(
            accepted=selection.accepted,
            rejected=selection.rejected,
            config=config,
        ),
    )
    write_json(
        stacking_manifest_path,
        _stacking_manifest(
            config=config,
            accepted=selection.accepted,
            rejected=selection.rejected,
            stack_oof_path=stack_oof_path,
            stack_test_path=stack_test_path,
            stack_oof=stack_oof,
            stack_test=stack_test,
            status=gate.status,
            oof_path=paths.oof_path,
            submission_path=paths.submission_path,
            oof_score=trained.oof_score,
        ),
    )

    artifact_file_paths = {
        "oof_path": str(paths.oof_path),
        "submission_path": str(paths.submission_path),
        "metrics_path": str(paths.metrics_path),
        "fold_metrics_path": str(paths.fold_metrics_path),
        "config_path": str(paths.config_snapshot_path),
        "stack_oof_path": str(stack_oof_path),
        "stack_test_path": str(stack_test_path),
        "selection_report_path": str(selection_report_path),
        "stacking_manifest_path": str(stacking_manifest_path),
    }
    run_manifest = build_run_manifest(
        competition=config.competition,
        experiment_id=config.experiment_id,
        data_dir=config.artifact_root,
        target=config.target,
        id_column=config.id_column,
        n_splits=_stack_fold_count(stack_oof),
        seed=stacker_seed,
        status=gate.status,
        paths=artifact_file_paths,
    )
    run_manifest["run_mode"] = "stack"
    run_manifest["metric_name"] = config.metric_name
    run_manifest["stacker_method"] = config.stacker_method
    run_manifest["parent_experiment_ids"] = [
        candidate.experiment_id for candidate in selection.accepted
    ]
    run_manifest["config_hash"] = stable_hash(config.raw)
    write_json(paths.run_manifest_path, add_file_checksums(run_manifest, artifact_file_paths))

    append_experiment_registry(
        paths.registry_path,
        {
            "experiment_id": config.experiment_id,
            "competition": config.competition,
            "metric_name": config.metric_name,
            "metric_mode": metric_mode(config.metric_name),
            "oof_score": trained.oof_score,
            "status": gate.status,
            "oof_path": str(paths.oof_path),
            "test_pred_path": str(paths.submission_path),
            "fold_metrics_path": str(paths.fold_metrics_path),
            "model_family": "stacking",
            "model_preset": config.stacker_method,
            "feature_set": "stacking",
            "config_hash": stable_hash(config.raw),
            "seed": stacker_seed,
            "run_mode": "stack",
            "reason": gate.reason,
            "feature_manifest_hash": None,
            "prediction_type": "probability",
            "parent_experiment_ids": ",".join(
                candidate.experiment_id for candidate in selection.accepted
            ),
            "parent_count": len(selection.accepted),
            "stack_oof_path": str(stack_oof_path),
            "stack_test_path": str(stack_test_path),
            "stacking_manifest_path": str(stacking_manifest_path),
            "selection_report_path": str(selection_report_path),
        },
    )

    return StackRunResult(
        status=gate.status,
        experiment_id=config.experiment_id,
        oof_score=trained.oof_score,
        metric_name=config.metric_name,
        oof_path=paths.oof_path,
        submission_path=paths.submission_path,
        stack_oof_path=stack_oof_path,
        stack_test_path=stack_test_path,
        selection_report_path=selection_report_path,
        stacking_manifest_path=stacking_manifest_path,
        run_manifest_path=paths.run_manifest_path,
    )


def _select_candidates(config: StackingPreflightConfig) -> _SelectedCandidates:
    pool = build_candidate_pool(
        artifact_root=config.artifact_root,
        competition=config.competition,
        metric_name=config.metric_name,
        candidate_ids=config.candidate_ids,
        top_n=config.top_n,
    )
    records = [_build_candidate_record(candidate, config=config) for candidate in pool.candidates]
    compatibility = evaluate_candidate_compatibility(
        records,
        target=config.target,
        id_column=config.id_column,
        min_parents=1,
    )

    accepted = list(compatibility.accepted)
    rejected = [
        CandidateCompatibilityRejection(
            experiment_id=item.experiment_id,
            reason=item.reason,
        )
        for item in pool.rejected
    ] + list(compatibility.rejected)

    missing_ids = _missing_candidate_ids(config.candidate_ids, pool)
    rejected.extend(
        CandidateCompatibilityRejection(
            experiment_id=experiment_id,
            reason="candidate_id was requested but not found in registry",
        )
        for experiment_id in missing_ids
    )

    if config.max_parents is not None and len(accepted) > config.max_parents:
        overflow = accepted[config.max_parents :]
        rejected.extend(
            CandidateCompatibilityRejection(
                experiment_id=candidate.experiment_id,
                reason=f"trimmed by max_parents={config.max_parents}",
            )
            for candidate in overflow
        )
        accepted = accepted[: config.max_parents]

    if len(accepted) < config.min_parents:
        raise ValueError(
            f"stack-preflight requires at least {config.min_parents} compatible parents; "
            f"found {len(accepted)}"
        )
    return _SelectedCandidates(accepted=accepted, rejected=rejected)


def _build_candidate_record(
    candidate: Any,
    *,
    config: StackingPreflightConfig,
) -> CandidateRecord:
    oof = pd.read_parquet(candidate.oof_path).reset_index(drop=True)
    test = pd.read_csv(candidate.test_pred_path).reset_index(drop=True)

    return CandidateRecord(
        experiment_id=candidate.experiment_id,
        model_family=candidate.model_family,
        oof_score=candidate.oof_score,
        row=dict(candidate.row),
        prediction_meta=PredictionArtifactMeta(
            competition=_string_or_default(candidate.row.get("competition"), config.competition),
            metric_name=_string_or_default(candidate.row.get("metric_name"), config.metric_name),
            prediction_type=_string_or_default(
                candidate.row.get("prediction_type"),
                "probability",
            ),
            target=_string_or_default(candidate.row.get("target"), config.target),
            id_column=_string_or_default(candidate.row.get("id_column"), config.id_column),
            oof_row_count=len(oof),
            test_row_count=len(test),
        ),
        cv_protocol=CVProtocolMeta(
            cv_protocol_id=_cv_protocol_id(candidate.row, oof),
            splitter=_string_or_default(candidate.row.get("splitter"), "unknown"),
            fold_count=_fold_count(candidate.row, oof),
            seed=_optional_int(candidate.row.get("seed")),
            oof_safe=_optional_bool(candidate.row.get("oof_safe"), default=True),
        ),
        oof=oof,
        test=test,
    )


def _missing_candidate_ids(config_candidate_ids: list[str], pool: CandidatePool) -> list[str]:
    if not config_candidate_ids:
        return []
    seen = {candidate.experiment_id for candidate in pool.candidates}
    seen.update(item.experiment_id for item in pool.rejected)
    return [experiment_id for experiment_id in config_candidate_ids if experiment_id not in seen]


def _build_stack_oof(
    candidates: list[CandidateRecord],
    *,
    target: str,
    id_column: str,
) -> pd.DataFrame:
    reference = candidates[0].oof.reset_index(drop=True)
    frame = pd.DataFrame(
        {
            id_column: reference[id_column].to_numpy(),
            target: reference[target].to_numpy(),
        }
    )
    if "fold" in reference.columns:
        frame["fold"] = reference["fold"].to_numpy()
    for candidate in candidates:
        frame[candidate.experiment_id] = candidate.oof["prediction"].to_numpy()
    return frame


def _build_stack_test(
    candidates: list[CandidateRecord],
    *,
    id_column: str,
) -> pd.DataFrame:
    reference = candidates[0].test.reset_index(drop=True)
    frame = pd.DataFrame({id_column: reference[id_column].to_numpy()})
    for candidate in candidates:
        target_column = candidate.prediction_meta.target
        frame[candidate.experiment_id] = candidate.test[target_column].to_numpy()
    return frame


def _train_oof_safe_stack(
    *,
    stack_oof: pd.DataFrame,
    stack_test: pd.DataFrame,
    config: StackingPreflightConfig,
) -> _TrainedStack:
    if "fold" not in stack_oof.columns:
        raise ValueError("stack runtime requires a fold column in stack_oof for OOF-safe training")

    feature_columns = [
        column
        for column in stack_oof.columns
        if column not in {config.id_column, config.target, "fold"}
    ]
    if not feature_columns:
        raise ValueError("stack runtime requires at least one parent prediction column")

    train_x = stack_oof[feature_columns].reset_index(drop=True)
    train_y = stack_oof[config.target].reset_index(drop=True)
    folds = stack_oof["fold"].reset_index(drop=True)
    test_x = stack_test[feature_columns].reset_index(drop=True)

    oof_predictions = pd.Series(np.zeros(len(stack_oof), dtype=float))
    fold_rows: list[dict[str, object]] = []
    seed = _resolved_stacker_seed(config)

    for fold in sorted(folds.drop_duplicates().tolist()):
        valid_mask = folds == fold
        fit_x = train_x.loc[~valid_mask]
        fit_y = train_y.loc[~valid_mask]
        valid_x = train_x.loc[valid_mask]
        valid_y = train_y.loc[valid_mask]
        model = _fit_stacker(
            method=config.stacker_method,
            params=config.stacker_params,
            seed=seed,
            features=fit_x,
            target=fit_y,
        )
        fold_predictions = _predict_stacker(model, valid_x).clip(0.0, 1.0)
        oof_predictions.loc[valid_mask] = fold_predictions.to_numpy()
        fold_rows.append(
            {
                "fold": fold,
                "metric_name": config.metric_name,
                "score": score_predictions(config.metric_name, valid_y, fold_predictions),
                "rows": int(valid_mask.sum()),
            }
        )

    final_model = _fit_stacker(
        method=config.stacker_method,
        params=config.stacker_params,
        seed=seed,
        features=train_x,
        target=train_y,
    )
    test_predictions = _predict_stacker(final_model, test_x).clip(0.0, 1.0)
    oof_score = float(score_predictions(config.metric_name, train_y, oof_predictions))

    oof = pd.DataFrame(
        {
            config.id_column: stack_oof[config.id_column].to_numpy(),
            config.target: train_y.to_numpy(),
            "prediction": oof_predictions.to_numpy(),
            "fold": folds.to_numpy(),
        }
    )
    submission = pd.DataFrame(
        {
            config.id_column: stack_test[config.id_column].to_numpy(),
            config.target: test_predictions.to_numpy(),
        }
    )
    return _TrainedStack(
        oof=oof,
        submission=submission,
        fold_metrics=pd.DataFrame(fold_rows),
        oof_score=oof_score,
    )


def _fit_stacker(
    *,
    method: str,
    params: dict[str, Any],
    seed: int,
    features: pd.DataFrame,
    target: pd.Series,
) -> Any:
    target_values = target.reset_index(drop=True)
    if target_values.nunique(dropna=False) < 2:
        return {"type": "constant", "value": float(target_values.iloc[0])}

    method_key = method.strip().lower()
    estimator_params = dict(params)
    estimator_params.pop("seed", None)
    if method_key == "logistic_regression":
        return LogisticRegression(
            random_state=seed,
            **estimator_params,
        ).fit(features, target_values)
    if method_key == "ridge_classifier":
        return RidgeClassifier(**estimator_params).fit(features, target_values)
    raise ValueError(
        "Unsupported stacker.method "
        f"{method!r}. Known methods: logistic_regression, ridge_classifier."
    )


def _predict_stacker(model: Any, features: pd.DataFrame) -> pd.Series:
    if isinstance(model, dict) and model.get("type") == "constant":
        return pd.Series(np.full(len(features), float(model["value"])), dtype=float)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        return pd.Series(probabilities[:, -1], dtype=float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        return pd.Series(_sigmoid(scores), dtype=float)
    predictions = np.asarray(model.predict(features), dtype=float)
    return pd.Series(predictions, dtype=float)


def _selection_report(
    *,
    accepted: list[CandidateRecord],
    rejected: list[CandidateCompatibilityRejection],
    config: StackingPreflightConfig,
) -> str:
    accepted_lines = [
        "| experiment_id | model_family | oof_score | fold_count | cv_protocol_id |",
        "| --- | --- | --- | --- | --- |",
    ]
    for candidate in accepted:
        row = (
            "| {experiment_id} | {model_family} | {oof_score} | {fold_count} | "
            "{cv_protocol_id} |"
        ).format(
            experiment_id=candidate.experiment_id,
            model_family=candidate.model_family or "",
            oof_score="" if candidate.oof_score is None else candidate.oof_score,
            fold_count=candidate.cv_protocol.fold_count,
            cv_protocol_id=candidate.cv_protocol.cv_protocol_id,
        )
        accepted_lines.append(row)

    rejected_lines = [
        "| experiment_id | reason |",
        "| --- | --- |",
    ]
    for item in rejected:
        rejected_lines.append(
            f"| {item.experiment_id} | {_escape_markdown_cell(item.reason)} |"
        )

    return (
        "# Stacking Preflight Selection Report\n\n"
        f"- experiment_id: `{config.experiment_id}`\n"
        f"- competition: `{config.competition}`\n"
        f"- metric_name: `{config.metric_name}`\n"
        f"- min_parents: `{config.min_parents}`\n"
        f"- max_parents: `{config.max_parents}`\n"
        f"- stacker_method: `{config.stacker_method}`\n\n"
        "## Accepted Candidates\n\n"
        f"{chr(10).join(accepted_lines)}\n\n"
        "## Rejected Candidates\n\n"
        f"{chr(10).join(rejected_lines)}\n"
    )


def _stacking_manifest(
    *,
    config: StackingPreflightConfig,
    accepted: list[CandidateRecord],
    rejected: list[CandidateCompatibilityRejection],
    stack_oof_path: Path,
    stack_test_path: Path,
    stack_oof: pd.DataFrame,
    stack_test: pd.DataFrame,
    status: str,
    oof_path: Path | None = None,
    submission_path: Path | None = None,
    oof_score: float | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": status,
        "experiment_id": config.experiment_id,
        "competition": config.competition,
        "target": config.target,
        "id_column": config.id_column,
        "metric_name": config.metric_name,
        "stacker_method": config.stacker_method,
        "stacker_params": config.stacker_params,
        "accepted_candidate_ids": [candidate.experiment_id for candidate in accepted],
        "rejected_candidates": [
            {"experiment_id": item.experiment_id, "reason": item.reason}
            for item in rejected
        ],
        "stack_oof_path": str(stack_oof_path),
        "stack_test_path": str(stack_test_path),
        "stack_oof_columns": stack_oof.columns.tolist(),
        "stack_test_columns": stack_test.columns.tolist(),
        "config_hash": stable_hash(config.raw),
    }
    if oof_path is not None:
        payload["oof_path"] = str(oof_path)
    if submission_path is not None:
        payload["submission_path"] = str(submission_path)
    if oof_score is not None:
        payload["oof_score"] = oof_score
    return payload


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _write_config_snapshot(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _cv_protocol_id(row: dict[str, Any], oof: pd.DataFrame) -> str:
    explicit = row.get("cv_protocol_id")
    if explicit is not None and str(explicit) != "" and not pd.isna(explicit):
        return str(explicit)
    fold_count = _fold_count(row, oof)
    seed = _optional_int(row.get("seed"))
    return f"inferred-folds{fold_count}-seed{seed if seed is not None else 'na'}"


def _fold_count(row: dict[str, Any], oof: pd.DataFrame) -> int:
    explicit = _optional_int(row.get("fold_count"))
    if explicit is not None:
        return explicit
    if "fold" not in oof.columns:
        return 0
    return int(oof["fold"].nunique(dropna=False))


def _stack_fold_count(stack_oof: pd.DataFrame) -> int:
    if "fold" not in stack_oof.columns:
        return 0
    return int(stack_oof["fold"].nunique(dropna=False))


def _resolved_stacker_seed(config: StackingPreflightConfig) -> int:
    seed = _optional_int(config.stacker_params.get("seed"))
    if seed is not None:
        return seed
    return 42


def _optional_int(value: object) -> int | None:
    if value is None or str(value) == "" or pd.isna(value):
        return None
    return int(value)


def _optional_bool(value: object, *, default: bool) -> bool:
    if value is None or (not isinstance(value, bool) and pd.isna(value)):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return default


def _string_or_default(value: object, default: str) -> str:
    if value is None or str(value) == "" or pd.isna(value):
        return default
    return str(value)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))
