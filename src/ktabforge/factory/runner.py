from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ktabforge.artifacts.writers import write_csv, write_json
from ktabforge.config.matrix import MatrixConfig, load_matrix_config
from ktabforge.factory.matrix import ExpandedExperiment, MatrixPlan, expand_matrix_config
from ktabforge.factory.report import write_candidate_report
from ktabforge.factory.results import FactoryRunResult
from ktabforge.pipeline.results import ExperimentRunResult

ExperimentRunner = Callable[[Path], ExperimentRunResult]


def run_factory_from_config(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    max_runs: int | None = None,
    experiment_runner: ExperimentRunner | None = None,
) -> FactoryRunResult:
    config = load_matrix_config(config_path)
    return run_factory(
        config,
        dry_run=dry_run,
        max_runs=max_runs,
        experiment_runner=experiment_runner,
    )


def run_factory(
    config: MatrixConfig,
    *,
    dry_run: bool = False,
    max_runs: int | None = None,
    experiment_runner: ExperimentRunner | None = None,
) -> FactoryRunResult:
    runner = experiment_runner or _default_experiment_runner
    plan = expand_matrix_config(config, max_runs=max_runs)
    factory_dir = config.artifact_root / "factory" / config.competition / config.factory_id
    expanded_dir = factory_dir / "expanded_configs"
    expanded_dir.mkdir(parents=True, exist_ok=True)

    plan_rows = _write_expanded_configs(plan, config=config, expanded_dir=expanded_dir)
    plan_path = write_csv(factory_dir / "matrix_plan.csv", pd.DataFrame(plan_rows))
    write_json(factory_dir / "matrix_config.json", config.raw)

    if dry_run:
        summary = pd.DataFrame(
            [
                {
                    **row,
                    "status": "planned",
                    "metric_name": config.report_metric_name,
                    "oof_score": None,
                    "reason": "dry run",
                    "error_type": None,
                    "error_message": None,
                    "duration_seconds": 0.0,
                    "registry_path": None,
                    "oof_path": None,
                    "test_pred_path": None,
                }
                for row in plan_rows
            ]
        )
    else:
        summary = _execute_plan(
            plan=plan,
            plan_rows=plan_rows,
            runner=runner,
            continue_on_error=config.continue_on_error,
            metric_name=config.report_metric_name,
        )

    summary_path = write_csv(factory_dir / "run_summary.csv", summary)
    summary_json_path = write_json(
        factory_dir / "run_summary.json",
        {"runs": summary.to_dict(orient="records")},
    )
    report_path = write_candidate_report(
        factory_dir / "candidate_report.md",
        factory_id=config.factory_id,
        metric_name=config.report_metric_name,
        summary=summary,
        top_n=config.report_top_n,
    )

    completed_count = int((summary["status"] == "completed").sum()) if not summary.empty else 0
    failed_count = int((summary["status"] == "failed").sum()) if not summary.empty else 0
    status = "planned" if dry_run else ("failed" if failed_count else "completed")
    return FactoryRunResult(
        status=status,
        completed_count=completed_count,
        failed_count=failed_count,
        planned_count=len(plan.runs),
        factory_dir=factory_dir,
        plan_path=plan_path,
        summary_path=summary_path,
        summary_json_path=summary_json_path,
        report_path=report_path,
    )


def _write_expanded_configs(
    plan: MatrixPlan,
    *,
    config: MatrixConfig,
    expanded_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in plan.runs:
        config_path = expanded_dir / f"{run.experiment_id}.yaml"
        config_path.write_text(yaml.safe_dump(run.payload, sort_keys=False), encoding="utf-8")
        rows.append(_plan_row(run, config=config, config_path=config_path))
    return rows


def _plan_row(
    run: ExpandedExperiment,
    *,
    config: MatrixConfig,
    config_path: Path,
) -> dict[str, Any]:
    experiment = run.payload.get("experiment", {})
    model = run.payload.get("model", {})
    features = run.payload.get("features", {})
    return {
        "factory_id": config.factory_id,
        "experiment_id": run.experiment_id,
        "status": "planned",
        "config_path": str(config_path),
        "config_hash": run.config_hash,
        "model_family": model.get("family"),
        "model_preset": model.get("preset"),
        "feature_set": features.get("feature_set"),
        "seed": experiment.get("seed"),
        "axis_values": yaml.safe_dump(run.axis_values, sort_keys=True).strip(),
    }


def _execute_plan(
    *,
    plan: MatrixPlan,
    plan_rows: list[dict[str, Any]],
    runner: ExperimentRunner,
    continue_on_error: bool,
    metric_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, (_run, plan_row) in enumerate(zip(plan.runs, plan_rows, strict=True)):
        start = time.monotonic()
        try:
            result = runner(Path(plan_row["config_path"]))
            rows.append(
                {
                    **plan_row,
                    "status": result.status,
                    "metric_name": result.metric_name,
                    "oof_score": result.oof_score,
                    "reason": result.reason,
                    "error_type": None,
                    "error_message": None,
                    "duration_seconds": round(time.monotonic() - start, 6),
                    "registry_path": str(result.paths.registry_path),
                    "oof_path": str(result.paths.oof_path),
                    "test_pred_path": str(result.paths.submission_path),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **plan_row,
                    "status": "failed",
                    "metric_name": metric_name,
                    "oof_score": None,
                    "reason": str(exc),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "duration_seconds": round(time.monotonic() - start, 6),
                    "registry_path": None,
                    "oof_path": None,
                    "test_pred_path": None,
                }
            )
            if not continue_on_error:
                for skipped_row in plan_rows[index + 1 :]:
                    rows.append(
                        {
                            **skipped_row,
                            "status": "skipped",
                            "metric_name": metric_name,
                            "oof_score": None,
                            "reason": "stopped after previous failure",
                            "error_type": None,
                            "error_message": None,
                            "duration_seconds": 0.0,
                            "registry_path": None,
                            "oof_path": None,
                            "test_pred_path": None,
                        }
                    )
                break
    return pd.DataFrame(rows)


def _default_experiment_runner(config_path: Path) -> ExperimentRunResult:
    from ktabforge.pipeline.runner import run_experiment_from_config

    return run_experiment_from_config(config_path)
