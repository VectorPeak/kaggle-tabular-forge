from __future__ import annotations

import json
from pathlib import Path

import yaml

from ktabforge.eda.profiling import build_feature_backlog, profile_tabular_frames
from ktabforge.eda.runner import run_eda_scan_from_config


def test_profile_tabular_frames_summarizes_rows_missingness_and_target_distribution():
    import pandas as pd

    train = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
            "num_a": [1.0, None, 3.0, 4.0],
            "num_b": [10.0, 20.0, 30.0, 40.0],
            "cat_a": ["x", "y", "x", None],
        }
    )
    test = pd.DataFrame(
        {
            "id": [5, 6],
            "num_a": [5.0, None],
            "num_b": [50.0, 60.0],
            "cat_a": ["z", "x"],
        }
    )

    profile = profile_tabular_frames(train=train, test=test, target="target", id_column="id")

    assert profile["row_count"] == {"train": 4, "test": 2}
    assert profile["feature_columns"] == ["num_a", "num_b", "cat_a"]
    assert profile["numeric_columns"] == ["num_a", "num_b"]
    assert profile["categorical_columns"] == ["cat_a"]
    assert profile["duplicate_id_count"] == {"train": 0, "test": 0}
    assert profile["target_distribution"] == {"0": 2, "1": 2}
    assert profile["missing_fraction"]["num_a"]["train"] == 0.25
    assert profile["missing_fraction"]["cat_a"]["train"] == 0.25


def test_build_feature_backlog_produces_minimal_seed_families():
    import pandas as pd

    train = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "target": [0, 1, 0],
            "num_a": [1.0, 2.0, 3.0],
            "num_b": [10.0, 20.0, 30.0],
            "cat_a": ["x", "y", "x"],
        }
    )
    test = pd.DataFrame(
        {
            "id": [4, 5],
            "num_a": [4.0, 5.0],
            "num_b": [40.0, 50.0],
            "cat_a": ["z", "x"],
        }
    )

    profile = profile_tabular_frames(train=train, test=test, target="target", id_column="id")
    backlog = build_feature_backlog(profile)

    families = {item["feature_family"] for item in backlog}

    assert "frequency_count" in families
    assert "binning" in families
    assert "arithmetic_interactions" in families
    assert all("feature_id" in item for item in backlog)
    assert all(item["status"] == "proposed" for item in backlog)


def test_run_eda_scan_from_config_writes_expected_artifacts(tmp_path: Path):
    data_dir = tmp_path / "data"
    artifact_root = tmp_path / "artifacts"
    data_dir.mkdir()

    fixture_dir = (
        Path(__file__).resolve().parents[1] / "fixtures" / "data" / "churn_tiny"
    )
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        (data_dir / name).write_text(
            (fixture_dir / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    config_path = tmp_path / "eda.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "eda": {
                    "eda_id": "p6-eda-unit",
                    "competition": "churn_tiny",
                    "artifact_root": str(artifact_root),
                },
                "data": {
                    "data_dir": str(data_dir),
                    "target": "Churn",
                    "id_column": "id",
                },
                "scan": {
                    "profile_train": True,
                    "profile_test": True,
                    "include_drift": True,
                    "include_leakage_watchlist": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = run_eda_scan_from_config(config_path)

    artifact_dir = artifact_root / "eda_findings" / "churn_tiny" / "p6-eda-unit"
    assert result.status == "completed"
    assert result.eda_id == "p6-eda-unit"
    assert result.artifact_dir == artifact_dir
    assert (artifact_dir / "eda_manifest.json").exists()
    assert (artifact_dir / "eda_summary.md").exists()
    assert (artifact_dir / "leakage_watchlist.json").exists()
    assert (artifact_dir / "feature_backlog.json").exists()

    manifest = json.loads((artifact_dir / "eda_manifest.json").read_text(encoding="utf-8"))
    backlog = json.loads((artifact_dir / "feature_backlog.json").read_text(encoding="utf-8"))

    assert manifest["eda_id"] == "p6-eda-unit"
    assert manifest["competition"] == "churn_tiny"
    assert "profiling" in manifest["focus"]
    assert "leakage" in manifest["focus"]
    assert any(item["feature_family"] == "frequency_count" for item in backlog["items"])
