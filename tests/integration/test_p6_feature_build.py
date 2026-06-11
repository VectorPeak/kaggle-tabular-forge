from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from ktabforge.features.pipeline import run_feature_build_from_config


def _write_feature_config(path: Path, *, repo_root: Path, artifact_root: Path) -> Path:
    payload = {
        "feature_build": {
            "feature_build_id": "p06-feature-smoke",
            "competition": "churn_tiny",
            "artifact_root": str(artifact_root),
        },
        "data": {
            "data_dir": str(repo_root / "tests" / "fixtures" / "data" / "churn_tiny"),
            "target": "Churn",
            "id_column": "id",
        },
        "transforms": [
            {
                "type": "frequency",
                "columns": ["Contract", "InternetService"],
                "mode": "count",
            },
            {
                "type": "binning",
                "columns": ["tenure", "MonthlyCharges"],
                "bins": 4,
                "strategy": "quantile",
            },
            {
                "type": "arithmetic_interactions",
                "pairs": [
                    {
                        "left": "MonthlyCharges",
                        "right": "tenure",
                        "operations": ["mul", "div"],
                    }
                ],
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_feature_build_writes_expected_artifacts_and_preserves_core_columns(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    artifact_root = tmp_path / "artifacts"
    config_path = _write_feature_config(
        tmp_path / "feature_build.yaml",
        repo_root=repo_root,
        artifact_root=artifact_root,
    )

    result = run_feature_build_from_config(config_path)

    assert result.status == "completed"
    assert result.feature_build_id == "p06-feature-smoke"

    feature_dir = artifact_root / "features" / "churn_tiny" / "p06-feature-smoke"
    train_features_path = feature_dir / "train_features.parquet"
    test_features_path = feature_dir / "test_features.parquet"
    manifest_path = feature_dir / "feature_build_manifest.json"
    schema_path = feature_dir / "feature_schema.json"
    report_path = feature_dir / "feature_build_report.md"

    assert train_features_path.exists()
    assert test_features_path.exists()
    assert manifest_path.exists()
    assert schema_path.exists()
    assert report_path.exists()

    train_features = pd.read_parquet(train_features_path)
    test_features = pd.read_parquet(test_features_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")

    raw_train = pd.read_csv(repo_root / "tests" / "fixtures" / "data" / "churn_tiny" / "train.csv")
    raw_test = pd.read_csv(repo_root / "tests" / "fixtures" / "data" / "churn_tiny" / "test.csv")

    assert len(train_features) == len(raw_train)
    assert len(test_features) == len(raw_test)
    assert train_features.columns[:2].tolist() == ["id", "Churn"]
    assert test_features.columns[0] == "id"
    assert "Churn" not in test_features.columns
    assert "Contract__count" in train_features.columns
    assert "tenure__bin" in train_features.columns
    assert "MonthlyCharges__mul__tenure" in train_features.columns
    assert manifest["status"] == "completed"
    assert manifest["feature_build_id"] == "p06-feature-smoke"
    assert manifest["train_row_count"] == len(raw_train)
    assert manifest["test_row_count"] == len(raw_test)
    assert manifest["generated_feature_count"] == len(feature_schema["features"])
    assert any(item["feature_name"] == "Contract__count" for item in feature_schema["features"])
    assert "frequency" in report
    assert "binning" in report
    assert "arithmetic_interactions" in report
