from pathlib import Path

import yaml

from ktabforge.config.matrix import load_matrix_config
from ktabforge.config.schema import validate_config_file
from ktabforge.factory.matrix import expand_matrix_config


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
    axes: dict[str, list[object]],
    experiment_id_template: str = "p04-{model.family}-seed{experiment.seed}",
    max_runs: int | None = None,
    factory_id: str = "p04-test-matrix",
    competition: str = "churn_tiny",
) -> Path:
    payload = {
        "factory": {
            "factory_id": factory_id,
            "competition": competition,
            "artifact_root": str(artifact_root),
            "continue_on_error": True,
            "max_runs": max_runs,
        },
        "base_experiment": _base_experiment_payload(repo_root, artifact_root),
        "matrix": {
            "experiment_id_template": experiment_id_template,
            "axes": axes,
        },
        "report": {
            "metric_name": "roc_auc",
            "top_n": 10,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_matrix_expansion_builds_cartesian_product_with_stable_ids(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    config_path = _write_matrix_config(
        tmp_path / "matrix.yaml",
        repo_root=repo_root,
        artifact_root=tmp_path / "artifacts",
        axes={
            "model.family": ["logistic_regression", "lightgbm"],
            "experiment.seed": [11, 22],
        },
    )

    matrix_config = load_matrix_config(config_path)
    plan = expand_matrix_config(matrix_config)

    assert [run.experiment_id for run in plan.runs] == [
        "p04-logistic_regression-seed11",
        "p04-logistic_regression-seed22",
        "p04-lightgbm-seed11",
        "p04-lightgbm-seed22",
    ]
    assert plan.runs[0].payload["model"]["family"] == "logistic_regression"
    assert plan.runs[2].payload["model"]["family"] == "lightgbm"
    assert plan.runs[1].payload["experiment"]["seed"] == 22


def test_matrix_expansion_preserves_base_config_fields_not_in_matrix(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    artifact_root = tmp_path / "artifacts"
    config_path = _write_matrix_config(
        tmp_path / "matrix.yaml",
        repo_root=repo_root,
        artifact_root=artifact_root,
        axes={"experiment.seed": [11]},
    )

    plan = expand_matrix_config(load_matrix_config(config_path))
    payload = plan.runs[0].payload

    assert payload["data"]["artifact_root"] == str(artifact_root)
    assert payload["data"]["target"] == "Churn"
    assert payload["validation"]["n_splits"] == 3
    assert payload["features"]["feature_set"] == "basic_inferred"
    assert payload["model"]["params"]["max_iter"] == 1000


def test_matrix_expansion_uses_factory_artifact_root_and_competition(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    artifact_root = tmp_path / "artifacts"
    config_path = _write_matrix_config(
        tmp_path / "matrix.yaml",
        repo_root=repo_root,
        artifact_root=artifact_root,
        axes={"experiment.seed": [11]},
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["base_experiment"]["experiment"]["competition"] = "../escape"
    payload["base_experiment"]["data"]["artifact_root"] = "../escape"
    payload["base_experiment"]["outputs"] = {
        "experiment_dir": "../escape",
        "oof_dir": "../escape",
        "submission_dir": "../escape",
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    plan = expand_matrix_config(load_matrix_config(config_path))
    expanded = plan.runs[0].payload

    assert expanded["experiment"]["competition"] == "churn_tiny"
    assert expanded["data"]["artifact_root"] == str(artifact_root)
    assert expanded["outputs"]["experiment_dir"].endswith(
        "artifacts/experiments/churn_tiny/p04-logistic_regression-seed11"
    )
    assert expanded["outputs"]["oof_dir"].endswith(
        "artifacts/oof/churn_tiny/p04-logistic_regression-seed11"
    )


def test_matrix_expansion_rejects_duplicate_experiment_ids(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    config_path = _write_matrix_config(
        tmp_path / "matrix.yaml",
        repo_root=repo_root,
        artifact_root=tmp_path / "artifacts",
        axes={"experiment.seed": [11, 22]},
        experiment_id_template="p04-duplicate",
    )

    try:
        expand_matrix_config(load_matrix_config(config_path))
    except ValueError as exc:
        assert "Duplicate experiment_id" in str(exc)
    else:
        raise AssertionError("duplicate experiment ids should be rejected")


def test_max_runs_limits_expanded_plan_in_matrix_order(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    config_path = _write_matrix_config(
        tmp_path / "matrix.yaml",
        repo_root=repo_root,
        artifact_root=tmp_path / "artifacts",
        axes={
            "model.family": ["logistic_regression", "lightgbm"],
            "experiment.seed": [11, 22],
        },
        max_runs=3,
    )

    plan = expand_matrix_config(load_matrix_config(config_path))

    assert [run.experiment_id for run in plan.runs] == [
        "p04-logistic_regression-seed11",
        "p04-logistic_regression-seed22",
        "p04-lightgbm-seed11",
    ]


def test_matrix_expansion_rejects_duplicate_ids_before_max_runs_truncation(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    config_path = _write_matrix_config(
        tmp_path / "matrix.yaml",
        repo_root=repo_root,
        artifact_root=tmp_path / "artifacts",
        axes={"experiment.seed": [11, 22]},
        experiment_id_template="p04-duplicate",
        max_runs=1,
    )

    try:
        expand_matrix_config(load_matrix_config(config_path))
    except ValueError as exc:
        assert "Duplicate experiment_id" in str(exc)
    else:
        raise AssertionError("duplicate ids in the full matrix should be rejected")


def test_matrix_config_rejects_unsafe_factory_id_and_competition(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    for index, kwargs in enumerate(
        [
            {"factory_id": "../escape"},
            {"factory_id": "bad\\id"},
            {"competition": "../other"},
            {"competition": "bad/name"},
        ]
    ):
        config_path = _write_matrix_config(
            tmp_path / f"matrix-{index}.yaml",
            repo_root=repo_root,
            artifact_root=tmp_path / "artifacts",
            axes={"experiment.seed": [11]},
            **kwargs,
        )

        try:
            load_matrix_config(config_path)
        except ValueError as exc:
            assert "Unsafe" in str(exc)
        else:
            raise AssertionError("unsafe factory path segments should be rejected")


def test_matrix_config_rejects_unsafe_axis_paths(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    for index, axes in enumerate(
        [
            {"data.artifact_root": ["../escape"]},
            {"experiment.competition": ["../escape"]},
            {"experiment.experiment_id": ["manual-id"]},
            {"outputs.oof_dir": ["../escape"]},
            {"model..family": ["logistic_regression"]},
        ]
    ):
        config_path = _write_matrix_config(
            tmp_path / f"matrix-axis-{index}.yaml",
            repo_root=repo_root,
            artifact_root=tmp_path / "artifacts",
            axes=axes,
        )

        try:
            load_matrix_config(config_path)
        except ValueError as exc:
            assert "Unsafe matrix axis path" in str(exc)
        else:
            raise AssertionError("unsafe matrix axis path should be rejected")


def test_matrix_config_requires_strict_boolean_and_positive_top_n(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    config_path = _write_matrix_config(
        tmp_path / "matrix.yaml",
        repo_root=repo_root,
        artifact_root=tmp_path / "artifacts",
        axes={"experiment.seed": [11]},
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["factory"]["continue_on_error"] = "false"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    try:
        load_matrix_config(config_path)
    except TypeError as exc:
        assert "continue_on_error" in str(exc)
    else:
        raise AssertionError("string booleans should be rejected")

    payload["factory"]["continue_on_error"] = False
    payload["report"]["top_n"] = 0
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    try:
        load_matrix_config(config_path)
    except ValueError as exc:
        assert "report.top_n" in str(exc)
    else:
        raise AssertionError("non-positive report.top_n should be rejected")


def test_matrix_expansion_generates_experiment_schema_valid_payload(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    config_path = _write_matrix_config(
        tmp_path / "matrix.yaml",
        repo_root=repo_root,
        artifact_root=tmp_path / "artifacts",
        axes={"experiment.seed": [11]},
    )

    run = expand_matrix_config(load_matrix_config(config_path)).runs[0]
    expanded_path = tmp_path / "expanded.yaml"
    expanded_path.write_text(yaml.safe_dump(run.payload, sort_keys=False), encoding="utf-8")
    result = validate_config_file(
        expanded_path,
        repo_root / "configs" / "schemas" / "experiment.schema.json",
    )

    assert result.valid is True
    assert result.errors == []
