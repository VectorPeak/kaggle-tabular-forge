from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ktabforge.config.schema import validate_config_file
from ktabforge.stacking.config import load_stacking_config


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "schemas" / "stacking.schema.json"


def _write_config(
    tmp_path: Path,
    *,
    stacking_overrides: dict[str, object] | None = None,
) -> Path:
    payload = {
        "stacking": {
            "experiment_id": "p05-stack-config-test",
            "competition": "churn_tiny",
            "artifact_root": str(tmp_path / "artifacts"),
            "target": "Churn",
            "id_column": "id",
            "metric_name": "roc_auc",
            "candidate_ids": ["candidate-a", "candidate-b"],
            "top_n": 2,
            "max_parents": 2,
            "min_parents": 2,
            "selection": {
                "strategy": "score_desc",
                "max_pairwise_corr": 0.99,
                "min_gain": 0.0,
                "report_top_k_pairs": 10,
            },
            "stacker": {
                "method": "logistic_regression",
                "params": {"C": 1.0, "max_iter": 200},
            },
        }
    }
    if stacking_overrides:
        payload["stacking"].update(stacking_overrides)
    config_path = tmp_path / "stacking.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.mark.parametrize(
    ("stacking_overrides", "expected_message"),
    [
        (
            {"selection": {"strategy": "score_desc", "max_pairwise_corr": float("nan")}},
            "selection.max_pairwise_corr",
        ),
        (
            {"selection": {"strategy": "score_desc", "max_pairwise_corr": float("inf")}},
            "selection.max_pairwise_corr",
        ),
        (
            {"selection": {"strategy": "hill_climb_greedy", "min_gain": float("nan")}},
            "selection.min_gain",
        ),
        (
            {"selection": {"strategy": "hill_climb_greedy", "min_gain": float("inf")}},
            "selection.min_gain",
        ),
        (
            {"stacker": {"method": "logistic_regression", "params": {"C": float("nan")}}},
            "stacker.params.C",
        ),
        (
            {"stacker": {"method": "logistic_regression", "params": {"C": float("inf")}}},
            "stacker.params.C",
        ),
    ],
)
def test_load_stacking_config_rejects_non_finite_values(
    tmp_path: Path,
    stacking_overrides: dict[str, object],
    expected_message: str,
):
    config_path = _write_config(tmp_path, stacking_overrides=stacking_overrides)

    with pytest.raises(ValueError, match=expected_message):
        load_stacking_config(config_path)


@pytest.mark.parametrize(
    ("stacking_overrides", "expected_path"),
    [
        ({"unexpected_root_flag": True}, "stacking"),
        (
            {
                "selection": {
                    "strategy": "score_desc",
                    "max_pairwise_corr": 0.99,
                    "typo_threshold": 0.1,
                }
            },
            "stacking.selection",
        ),
        (
            {
                "stacker": {
                    "method": "logistic_regression",
                    "params": {"C": 1.0},
                    "typo_param": True,
                }
            },
            "stacking.stacker",
        ),
    ],
)
def test_stacking_schema_rejects_unknown_fields(
    tmp_path: Path,
    stacking_overrides: dict[str, object],
    expected_path: str,
):
    config_path = _write_config(tmp_path, stacking_overrides=stacking_overrides)

    result = validate_config_file(config_path, _schema_path())

    assert result.valid is False
    assert any(expected_path in error for error in result.errors)
    assert any("Additional properties are not allowed" in error for error in result.errors)


def test_stacking_schema_rejects_unknown_stacker_method(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        stacking_overrides={
            "stacker": {"method": "totally_real_meta_model", "params": {"C": 1.0}}
        },
    )

    result = validate_config_file(config_path, _schema_path())

    assert result.valid is False
    assert any("stacking.stacker.method" in error for error in result.errors)


def test_load_stacking_config_allows_preflight_only_only_for_preflight_runtime(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        stacking_overrides={"stacker": {"method": "preflight_only", "params": {}}},
    )

    config = load_stacking_config(config_path, runtime="preflight")

    assert config.stacker_method == "preflight_only"
    with pytest.raises(ValueError, match="preflight_only"):
        load_stacking_config(config_path, runtime="stack")


def test_load_stacking_config_rejects_hill_climb_with_max_pairwise_corr(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        stacking_overrides={
            "selection": {
                "strategy": "hill_climb_greedy",
                "min_gain": 0.0,
                "max_pairwise_corr": 0.99,
            }
        },
    )

    with pytest.raises(ValueError, match="max_pairwise_corr"):
        load_stacking_config(config_path)


@pytest.mark.parametrize(
    ("stacking_overrides", "expected_message"),
    [
        ({"max_parents": 1, "min_parents": 2}, "max_parents"),
        ({"top_n": 1, "min_parents": 2}, "top_n"),
    ],
)
def test_load_stacking_config_rejects_invalid_parent_relationships(
    tmp_path: Path,
    stacking_overrides: dict[str, object],
    expected_message: str,
):
    config_path = _write_config(tmp_path, stacking_overrides=stacking_overrides)

    with pytest.raises(ValueError, match=expected_message):
        load_stacking_config(config_path)
