from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from ktabforge.cli import app
from ktabforge.config.experiment import load_experiment_config
from ktabforge.pipeline.runner import run_experiment_from_config
from ktabforge.reports.compare import compare_experiments


def _write_experiment_config(
    path: Path,
    *,
    repo_root: Path,
    artifact_root: Path,
    experiment_id: str,
    model_family: str = "logistic_regression",
) -> Path:
    payload = {
        "experiment": {
            "experiment_id": experiment_id,
            "competition": "churn_tiny",
            "seed": 42,
            "run_mode": "smoke",
            "tags": ["pytest", "p2"],
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
            "family": model_family,
            "preset": "baseline",
            "params": {"max_iter": 1000},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_experiment_config_loader_exposes_runtime_fields(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    config_path = _write_experiment_config(
        tmp_path / "experiment.yaml",
        repo_root=repo_root,
        artifact_root=tmp_path / "artifacts",
        experiment_id="p2-config-loader",
    )

    config = load_experiment_config(config_path)

    assert config.experiment_id == "p2-config-loader"
    assert config.competition == "churn_tiny"
    assert config.model_family == "logistic_regression"
    assert config.feature_set == "basic_inferred"
    assert config.config_hash


def test_config_driven_run_writes_artifacts_config_snapshot_and_registry(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    config_path = _write_experiment_config(
        tmp_path / "experiment.yaml",
        repo_root=repo_root,
        artifact_root=tmp_path / "artifacts",
        experiment_id="p2-config-run",
    )

    result = run_experiment_from_config(config_path)

    assert result.status == "completed"
    assert result.paths.oof_path.exists()
    assert result.paths.submission_path.exists()
    assert (result.paths.experiment_dir / "config.yaml").exists()

    registry = pd.read_csv(result.paths.registry_path)
    row = registry.loc[registry["experiment_id"] == "p2-config-run"].iloc[0]
    assert row["model_family"] == "logistic_regression"
    assert row["model_preset"] == "baseline"
    assert row["feature_set"] == "basic_inferred"
    assert row["config_hash"]
    assert row["status"] == "completed"


def test_cli_run_and_compare_use_registry_evidence(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    artifact_root = tmp_path / "artifacts"
    config_path = _write_experiment_config(
        tmp_path / "experiment.yaml",
        repo_root=repo_root,
        artifact_root=artifact_root,
        experiment_id="p2-cli-run",
    )
    runner = CliRunner()

    run_result = runner.invoke(app, ["run", "--config", str(config_path)])
    assert run_result.exit_code == 0, run_result.stdout
    assert "completed" in run_result.stdout.lower()

    compare_result = runner.invoke(
        app,
        [
            "compare",
            "--artifact-root",
            str(artifact_root),
            "--competition",
            "churn_tiny",
        ],
    )
    assert compare_result.exit_code == 0, compare_result.stdout
    assert "p2-cli-run" in compare_result.stdout
    assert "oof_score" in compare_result.stdout

    comparison = compare_experiments(artifact_root=artifact_root, competition="churn_tiny")
    assert comparison.iloc[0]["experiment_id"] == "p2-cli-run"

