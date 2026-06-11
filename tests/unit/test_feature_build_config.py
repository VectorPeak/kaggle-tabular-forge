from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ktabforge.config.schema import validate_config_file
from ktabforge.features.pipeline import load_feature_build_config


def _schema_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "schemas"
        / "feature_pipeline.schema.json"
    )


def _write_config(
    tmp_path: Path,
    *,
    feature_build_overrides: dict[str, object] | None = None,
    data_overrides: dict[str, object] | None = None,
    transforms: list[dict[str, object]] | None = None,
) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    payload = {
        "feature_build": {
            "feature_build_id": "p06-feature-config-test",
            "competition": "churn_tiny",
            "artifact_root": str(tmp_path / "artifacts"),
        },
        "data": {
            "data_dir": str(repo_root / "tests" / "fixtures" / "data" / "churn_tiny"),
            "target": "Churn",
            "id_column": "id",
        },
        "transforms": transforms
        or [
            {
                "type": "frequency",
                "columns": ["Contract"],
                "mode": "count",
            }
        ],
    }
    if feature_build_overrides:
        payload["feature_build"].update(feature_build_overrides)
    if data_overrides:
        payload["data"].update(data_overrides)
    config_path = tmp_path / "feature_build.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_load_feature_build_config_parses_valid_bundle(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        transforms=[
            {"type": "frequency", "columns": ["Contract"], "mode": "frequency"},
            {"type": "binning", "columns": ["tenure"], "bins": 4, "strategy": "quantile"},
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
    )

    config = load_feature_build_config(config_path)

    assert config.feature_build_id == "p06-feature-config-test"
    assert config.competition == "churn_tiny"
    assert config.target == "Churn"
    assert config.id_column == "id"
    assert [transform.type for transform in config.transforms] == [
        "frequency",
        "binning",
        "arithmetic_interactions",
    ]


def test_feature_build_schema_rejects_unknown_fields(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        transforms=[
            {
                "type": "frequency",
                "columns": ["Contract"],
                "mode": "count",
                "typo_option": True,
            }
        ],
    )

    result = validate_config_file(config_path, _schema_path())

    assert result.valid is False
    assert any("transforms.0" in error for error in result.errors)


def test_load_feature_build_config_rejects_unsafe_feature_build_id(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        feature_build_overrides={"feature_build_id": "../escape"},
    )

    with pytest.raises(ValueError, match="feature_build_id"):
        load_feature_build_config(config_path)


def test_load_feature_build_config_rejects_non_positive_bins(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        transforms=[
            {"type": "binning", "columns": ["tenure"], "bins": 0, "strategy": "quantile"},
        ],
    )

    with pytest.raises(ValueError, match="transforms.0"):
        load_feature_build_config(config_path)
