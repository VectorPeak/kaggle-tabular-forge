from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from ktabforge.cli import app


def _write_candidate(
    *,
    artifact_root: Path,
    competition: str,
    experiment_id: str,
    oof_predictions: list[float],
    test_predictions: list[float],
    model_family: str,
    oof_ids: list[int] | None = None,
    folds: list[int] | None = None,
) -> dict[str, object]:
    oof_ids = oof_ids or [1, 2, 3, 4]
    folds = folds or [0, 0, 1, 1]
    oof_path = artifact_root / "oof" / competition / experiment_id / "oof.parquet"
    test_path = artifact_root / "submissions" / competition / experiment_id / "submission.csv"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": oof_ids,
            "Churn": [0, 1, 0, 1],
            "prediction": oof_predictions,
            "fold": folds,
        }
    ).to_parquet(oof_path, index=False)
    pd.DataFrame({"id": [101, 102], "Churn": test_predictions}).to_csv(test_path, index=False)
    return {
        "experiment_id": experiment_id,
        "competition": competition,
        "metric_name": "roc_auc",
        "metric_mode": "max",
        "oof_score": 0.75,
        "status": "completed",
        "oof_path": str(oof_path),
        "test_pred_path": str(test_path),
        "fold_metrics_path": "",
        "model_family": model_family,
        "seed": 42,
        "run_mode": "run",
        "reason": "ok",
        "model_preset": "baseline",
        "feature_set": "basic",
        "config_hash": experiment_id,
        "created_at": "2026-06-11T00:00:00Z",
        "feature_manifest_hash": experiment_id,
        "prediction_type": "probability",
    }


def _write_registry(artifact_root: Path, competition: str, rows: list[dict[str, object]]) -> None:
    registry_path = artifact_root / "registry" / competition / "experiment_registry.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(registry_path, index=False)


def _write_stacking_config(path: Path, *, artifact_root: Path) -> Path:
    payload = {
        "stacking": {
            "experiment_id": "p05-stack-preflight",
            "competition": "churn_tiny",
            "artifact_root": str(artifact_root),
            "target": "Churn",
            "id_column": "id",
            "metric_name": "roc_auc",
            "candidate_ids": ["candidate-a", "candidate-b", "candidate-c"],
            "top_n": 3,
            "max_parents": 2,
            "min_parents": 2,
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_cli_stack_preflight_writes_selection_report_and_stack_matrices(tmp_path):
    artifact_root = tmp_path / "artifacts"
    competition = "churn_tiny"
    rows = [
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-a",
            oof_predictions=[0.1, 0.8, 0.3, 0.9],
            test_predictions=[0.2, 0.7],
            model_family="lightgbm",
        ),
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-b",
            oof_predictions=[0.2, 0.7, 0.4, 0.8],
            test_predictions=[0.3, 0.6],
            model_family="xgboost",
        ),
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-c",
            oof_predictions=[0.4, 0.6, 0.5, 0.7],
            test_predictions=[0.45, 0.55],
            model_family="catboost",
            folds=[1, 1, 0, 0],
        ),
    ]
    _write_registry(artifact_root, competition, rows)
    config_path = _write_stacking_config(tmp_path / "stacking.yaml", artifact_root=artifact_root)
    runner = CliRunner()

    result = runner.invoke(app, ["stack-preflight", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    assert "prepared" in result.stdout.lower()
    experiment_dir = artifact_root / "experiments" / competition / "p05-stack-preflight"
    stack_oof_path = (
        artifact_root / "oof" / competition / "p05-stack-preflight" / "stack_oof.parquet"
    )
    stack_test_path = (
        artifact_root / "submissions" / competition / "p05-stack-preflight" / "stack_test.parquet"
    )
    assert (experiment_dir / "selection_report.md").exists()
    assert (experiment_dir / "stacking_manifest.json").exists()
    assert stack_oof_path.exists()
    assert stack_test_path.exists()

    stack_oof = pd.read_parquet(stack_oof_path)
    stack_test = pd.read_parquet(stack_test_path)
    assert stack_oof.columns.tolist() == ["id", "Churn", "fold", "candidate-a", "candidate-b"]
    assert stack_test.columns.tolist() == ["id", "candidate-a", "candidate-b"]

    report = (experiment_dir / "selection_report.md").read_text(encoding="utf-8")
    assert "candidate-a" in report
    assert "candidate-b" in report
    assert "candidate-c" in report
    assert "fold" in report.lower()

    registry = pd.read_csv(artifact_root / "registry" / competition / "experiment_registry.csv")
    assert registry["experiment_id"].tolist() == ["candidate-a", "candidate-b", "candidate-c"]
