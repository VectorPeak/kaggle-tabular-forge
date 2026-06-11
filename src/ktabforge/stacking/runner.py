from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ktabforge.artifacts.layout import build_artifact_paths, ensure_new_artifact_layout
from ktabforge.artifacts.writers import write_json, write_markdown, write_parquet
from ktabforge.candidates import (
    CandidateCompatibilityRejection,
    CandidateRecord,
    CVProtocolMeta,
    PredictionArtifactMeta,
    evaluate_candidate_compatibility,
)
from ktabforge.candidates.pool import CandidatePool, build_candidate_pool
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


def run_stacking_preflight(config_path: str | Path) -> StackingPreflightResult:
    config = load_stacking_config(config_path)
    return run_preflight(config)


def run_preflight(config: StackingPreflightConfig) -> StackingPreflightResult:
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
        min_parents=config.min_parents,
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

    paths = build_artifact_paths(
        config.artifact_root,
        config.competition,
        config.experiment_id,
    )
    ensure_new_artifact_layout(paths)

    stack_oof = _build_stack_oof(accepted, target=config.target, id_column=config.id_column)
    stack_test = _build_stack_test(accepted, id_column=config.id_column)

    stack_oof_path = paths.oof_dir / "stack_oof.parquet"
    stack_test_path = paths.submission_dir / "stack_test.parquet"
    selection_report_path = paths.experiment_dir / "selection_report.md"
    stacking_manifest_path = paths.experiment_dir / "stacking_manifest.json"

    write_parquet(stack_oof_path, stack_oof)
    write_parquet(stack_test_path, stack_test)
    paths.config_snapshot_path.write_text(
        yaml.safe_dump(config.raw, sort_keys=False),
        encoding="utf-8",
    )
    write_markdown(
        selection_report_path,
        _selection_report(
            accepted=accepted,
            rejected=rejected,
            config=config,
        ),
    )
    write_json(
        stacking_manifest_path,
        {
            "status": "prepared",
            "experiment_id": config.experiment_id,
            "competition": config.competition,
            "target": config.target,
            "id_column": config.id_column,
            "metric_name": config.metric_name,
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
        },
    )

    return StackingPreflightResult(
        status="prepared",
        experiment_id=config.experiment_id,
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        selection_report_path=selection_report_path,
        stacking_manifest_path=stacking_manifest_path,
        stack_oof_path=stack_oof_path,
        stack_test_path=stack_test_path,
    )


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
        accepted_lines.append(
            row
        )

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
        f"- max_parents: `{config.max_parents}`\n\n"
        "## Accepted Candidates\n\n"
        f"{chr(10).join(accepted_lines)}\n\n"
        "## Rejected Candidates\n\n"
        f"{chr(10).join(rejected_lines)}\n"
    )


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


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
