from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ktabforge.factory.runner import run_factory_from_config
from ktabforge.pipeline.results import ExperimentRunResult


def _base_experiment_payload(repo_root: Path, artifact_root: Path) -> dict:
    return {
        "experiment": {
            "competition": "churn_tiny",
            "hypothesis": "Matrix generated candidate.",
            "seed": 42,
            "run_mode": "smoke",
            "tags": ["p4", "matrix"],
        },
        "data": {
            "data_dir": str(repo_root / "tests" / "fixtures" / "data" / "churn_tiny"),
            "artifact_root": str(artifact_root),
            "target": "Churn",
            "id_column": "id",
        },
        "validation": {
            "strategy": "stratified_kfold",
            "n_splits": 3,
            "shuffle": True,
            "seed": 42,
        },
        "features": {
            "feature_set": "basic_inferred",
            "families": ["basic"],
        },
        "model": {
            "family": "logistic_regression",
            "preset": "baseline",
            "params": {"max_iter": 1000},
        },
    }


def _write_matrix_config(
    path: Path,
    *,
    repo_root: Path,
    artifact_root: Path,
    axes: dict[str, list[object]] | None = None,
    max_runs: int | None = None,
    continue_on_error: bool = True,
) -> Path:
    payload = {
        "factory": {
            "factory_id": "p04-test-matrix",
            "competition": "churn_tiny",
            "artifact_root": str(artifact_root),
            "continue_on_error": continue_on_error,
            "max_runs": max_runs,
        },
        "base_experiment": _base_experiment_payload(repo_root, artifact_root),
        "matrix": {
            "experiment_id_template": "p04-{model.family}-seed{experiment.seed}",
            "axes": axes or {"experiment.seed": [11, 22]},
        },
        "report": {
            "metric_name": "roc_auc",
            "top_n": 10,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _fake_result(
    status: str = "completed",
    score: float = 0.7,
    metric_name: str = "roc_auc",
) -> ExperimentRunResult:
    paths = type(
        "Paths",
        (),
        {
            "registry_path": Path("registry.csv"),
            "oof_path": Path("oof.parquet"),
            "submission_path": Path("submission.csv"),
        },
    )()
    return ExperimentRunResult(
        status=status,
        oof_score=score,
        paths=paths,
        metric_name=metric_name,
        reason="ok",
    )


def test_candidate_factory_dry_run_writes_plan_report_without_p2_artifacts(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    artifact_root = tmp_path / "artifacts"
    config_path = _write_matrix_config(
        tmp_path / "matrix.yaml",
        repo_root=repo_root,
        artifact_root=artifact_root,
    )

    result = run_factory_from_config(config_path, dry_run=True)

    assert result.status == "planned"
    assert result.completed_count == 0
    assert result.failed_count == 0
    assert result.plan_path.exists()
    assert result.report_path.exists()
    assert len(list((result.factory_dir / "expanded_configs").glob("*.yaml"))) == 2
    assert not (artifact_root / "oof").exists()
    plan = pd.read_csv(result.plan_path)
    assert plan["status"].tolist() == ["planned", "planned"]


def test_candidate_factory_calls_runner_and_records_successful_runs(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    artifact_root = tmp_path / "artifacts"
    config_path = _write_matrix_config(
        tmp_path / "matrix.yaml",
        repo_root=repo_root,
        artifact_root=artifact_root,
    )
    calls: list[Path] = []

    def fake_runner(config_path: Path) -> ExperimentRunResult:
        calls.append(config_path)
        return _fake_result(score=0.81)

    result = run_factory_from_config(config_path, experiment_runner=fake_runner)

    assert result.status == "completed"
    assert result.completed_count == 2
    assert result.failed_count == 0
    assert len(calls) == 2
    summary = pd.read_csv(result.summary_path)
    assert summary["status"].tolist() == ["completed", "completed"]
    assert summary["oof_score"].tolist() == [0.81, 0.81]
    assert "Recommended ensemble candidate_ids" in result.report_path.read_text(encoding="utf-8")


def test_candidate_factory_records_failed_runs_and_continues(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    artifact_root = tmp_path / "artifacts"
    config_path = _write_matrix_config(
        tmp_path / "matrix.yaml",
        repo_root=repo_root,
        artifact_root=artifact_root,
        axes={"experiment.seed": [11, 22, 33]},
    )
    calls: list[str] = []

    def fake_runner(config_path: Path) -> ExperimentRunResult:
        payload: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        experiment_id = payload["experiment"]["experiment_id"]
        calls.append(experiment_id)
        if experiment_id.endswith("seed22"):
            raise RuntimeError("boom | bad\nline")
        return _fake_result(score=0.7)

    result = run_factory_from_config(config_path, experiment_runner=fake_runner)

    assert result.status == "failed"
    assert result.completed_count == 2
    assert result.failed_count == 1
    assert calls == [
        "p04-logistic_regression-seed11",
        "p04-logistic_regression-seed22",
        "p04-logistic_regression-seed33",
    ]
    summary = pd.read_csv(result.summary_path)
    failed = summary.loc[summary["status"] == "failed"].iloc[0]
    assert failed["error_type"] == "RuntimeError"
    assert "boom" in failed["error_message"]
    report_text = result.report_path.read_text(encoding="utf-8")
    assert (
        "| experiment_id | config_hash | config_path | error_type | error_message |"
        in report_text
    )
    assert "boom \\| bad<br>line" in report_text


def test_candidate_factory_respects_max_runs_before_calling_runner(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    artifact_root = tmp_path / "artifacts"
    config_path = _write_matrix_config(
        tmp_path / "matrix.yaml",
        repo_root=repo_root,
        artifact_root=artifact_root,
        axes={
            "model.family": ["logistic_regression", "lightgbm"],
            "experiment.seed": [11, 22],
        },
        max_runs=2,
    )
    calls: list[Path] = []

    def fake_runner(config_path: Path) -> ExperimentRunResult:
        calls.append(config_path)
        return _fake_result()

    result = run_factory_from_config(config_path, experiment_runner=fake_runner)

    assert len(calls) == 2
    assert result.completed_count == 2
    summary = pd.read_csv(result.summary_path)
    assert len(summary) == 2


def test_candidate_factory_marks_remaining_runs_skipped_when_stopping_on_failure(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    artifact_root = tmp_path / "artifacts"
    config_path = _write_matrix_config(
        tmp_path / "matrix.yaml",
        repo_root=repo_root,
        artifact_root=artifact_root,
        axes={"experiment.seed": [11, 22, 33]},
        continue_on_error=False,
    )
    calls: list[str] = []

    def fake_runner(config_path: Path) -> ExperimentRunResult:
        payload: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        experiment_id = payload["experiment"]["experiment_id"]
        calls.append(experiment_id)
        if experiment_id.endswith("seed22"):
            raise RuntimeError("boom")
        return _fake_result(score=0.7)

    result = run_factory_from_config(config_path, experiment_runner=fake_runner)

    assert result.status == "failed"
    assert calls == ["p04-logistic_regression-seed11", "p04-logistic_regression-seed22"]
    summary = pd.read_csv(result.summary_path)
    assert summary["status"].tolist() == ["completed", "failed", "skipped"]
    skipped = summary.loc[summary["status"] == "skipped"].iloc[0]
    assert skipped["reason"] == "stopped after previous failure"


def test_candidate_factory_report_respects_minimize_metric_order(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    artifact_root = tmp_path / "artifacts"
    config_path = _write_matrix_config(
        tmp_path / "matrix.yaml",
        repo_root=repo_root,
        artifact_root=artifact_root,
        axes={"experiment.seed": [11, 22]},
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["report"]["metric_name"] = "log_loss"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    scores = {
        "p04-logistic_regression-seed11": 0.42,
        "p04-logistic_regression-seed22": 0.21,
    }

    def fake_runner(config_path: Path) -> ExperimentRunResult:
        run_payload: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        experiment_id = run_payload["experiment"]["experiment_id"]
        return _fake_result(score=scores[experiment_id], metric_name="log_loss")

    result = run_factory_from_config(config_path, experiment_runner=fake_runner)

    report_text = result.report_path.read_text(encoding="utf-8")
    assert report_text.index("p04-logistic_regression-seed22") < report_text.index(
        "p04-logistic_regression-seed11"
    )
