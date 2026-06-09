from pathlib import Path

import yaml
from typer.testing import CliRunner

from ktabforge.cli import app


def _write_matrix_config(path: Path, *, repo_root: Path, artifact_root: Path) -> Path:
    payload = {
        "factory": {
            "factory_id": "p04-cli-matrix",
            "competition": "churn_tiny",
            "artifact_root": str(artifact_root),
            "continue_on_error": True,
            "max_runs": 3,
        },
        "base_experiment": {
            "experiment": {
                "competition": "churn_tiny",
                "hypothesis": "CLI matrix candidate.",
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
        },
        "matrix": {
            "experiment_id_template": "p04-{model.family}-seed{experiment.seed}",
            "axes": {
                "experiment.seed": [11, 22, 33],
            },
        },
        "report": {
            "metric_name": "roc_auc",
            "top_n": 10,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_cli_matrix_dry_run_prints_planned_runs_and_report_path(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    artifact_root = tmp_path / "artifacts"
    config_path = _write_matrix_config(
        tmp_path / "matrix.yaml",
        repo_root=repo_root,
        artifact_root=artifact_root,
    )
    runner = CliRunner()

    result = runner.invoke(app, ["factory", "--config", str(config_path), "--dry-run"])

    assert result.exit_code == 0, result.stdout
    assert "planned" in result.stdout
    assert "p04-logistic_regression-seed11" in result.stdout
    assert "candidate_report.md" in result.stdout


def test_cli_matrix_respects_max_runs_option(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    artifact_root = tmp_path / "artifacts"
    config_path = _write_matrix_config(
        tmp_path / "matrix.yaml",
        repo_root=repo_root,
        artifact_root=artifact_root,
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["factory", "--config", str(config_path), "--dry-run", "--max-runs", "2"],
    )

    assert result.exit_code == 0, result.stdout
    plan_path = artifact_root / "factory" / "churn_tiny" / "p04-cli-matrix" / "matrix_plan.csv"
    assert plan_path.exists()
    assert len(plan_path.read_text(encoding="utf-8").splitlines()) == 3


def test_cli_matrix_rejects_non_positive_max_runs(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    artifact_root = tmp_path / "artifacts"
    config_path = _write_matrix_config(
        tmp_path / "matrix.yaml",
        repo_root=repo_root,
        artifact_root=artifact_root,
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["factory", "--config", str(config_path), "--dry-run", "--max-runs", "0"],
    )

    assert result.exit_code != 0
    assert "max_runs must be greater than 0" in result.stdout
