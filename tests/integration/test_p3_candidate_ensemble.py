from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from ktabforge.candidates.pool import build_candidate_pool
from ktabforge.cli import app
from ktabforge.ensembles.runner import run_ensemble_from_config


def _write_candidate(
    *,
    artifact_root: Path,
    competition: str,
    experiment_id: str,
    oof_predictions: list[float],
    test_predictions: list[float],
    oof_ids: list[int] | None = None,
    test_ids: list[int] | None = None,
    status: str = "completed",
    score: float = 0.7,
    model_family: str = "logistic_regression",
) -> dict[str, object]:
    oof_ids = oof_ids or [1, 2, 3, 4]
    test_ids = test_ids or [101, 102]
    oof_path = artifact_root / "oof" / competition / experiment_id / "oof.parquet"
    submission_path = (
        artifact_root / "submissions" / competition / experiment_id / "submission.csv"
    )
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": oof_ids,
            "Churn": [0, 1, 0, 1],
            "prediction": oof_predictions,
            "fold": [0, 0, 1, 1],
        }
    ).to_parquet(oof_path, index=False)
    pd.DataFrame({"id": test_ids, "Churn": test_predictions}).to_csv(
        submission_path,
        index=False,
    )
    return {
        "experiment_id": experiment_id,
        "competition": competition,
        "metric_name": "roc_auc",
        "oof_score": score,
        "status": status,
        "oof_path": str(oof_path),
        "test_pred_path": str(submission_path),
        "fold_metrics_path": "",
        "model_family": model_family,
        "seed": 42,
        "run_mode": "smoke",
        "reason": "ok",
        "model_preset": "baseline",
        "feature_set": "basic",
        "config_hash": experiment_id,
        "created_at": "2026-06-09",
        "feature_manifest_hash": experiment_id,
        "prediction_type": "probability",
    }


def _write_registry(artifact_root: Path, competition: str, rows: list[dict[str, object]]) -> Path:
    registry_path = artifact_root / "registry" / competition / "experiment_registry.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(registry_path, index=False)
    return registry_path


def _write_ensemble_config(
    path: Path,
    *,
    artifact_root: Path,
    method: str = "simple_average",
) -> Path:
    payload = {
        "ensemble": {
            "experiment_id": "p03-ensemble-smoke",
            "competition": "churn_tiny",
            "artifact_root": str(artifact_root),
            "target": "Churn",
            "id_column": "id",
            "metric_name": "roc_auc",
            "method": method,
            "candidate_ids": ["candidate-a", "candidate-b"],
            "weights": {"candidate-a": 0.25, "candidate-b": 0.75},
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_two_candidate_registry(artifact_root: Path) -> str:
    competition = "churn_tiny"
    rows = [
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-a",
            oof_predictions=[0.1, 0.8, 0.3, 0.9],
            test_predictions=[0.2, 0.7],
            score=0.72,
        ),
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-b",
            oof_predictions=[0.2, 0.7, 0.4, 0.8],
            test_predictions=[0.3, 0.6],
            score=0.71,
            model_family="lightgbm",
        ),
    ]
    _write_registry(artifact_root, competition, rows)
    return competition


def test_candidate_pool_filters_completed_candidates_with_prediction_paths(tmp_path):
    artifact_root = tmp_path / "artifacts"
    competition = "churn_tiny"
    accepted = _write_candidate(
        artifact_root=artifact_root,
        competition=competition,
        experiment_id="candidate-a",
        oof_predictions=[0.1, 0.8, 0.3, 0.9],
        test_predictions=[0.2, 0.7],
    )
    rejected = _write_candidate(
        artifact_root=artifact_root,
        competition=competition,
        experiment_id="candidate-failed",
        oof_predictions=[0.2, 0.7, 0.4, 0.8],
        test_predictions=[0.3, 0.6],
        status="failed",
    )
    missing_path = dict(accepted)
    missing_path["experiment_id"] = "candidate-missing"
    missing_path["oof_path"] = str(artifact_root / "missing.parquet")
    _write_registry(artifact_root, competition, [accepted, rejected, missing_path])

    pool = build_candidate_pool(
        artifact_root=artifact_root,
        competition=competition,
        metric_name="roc_auc",
    )

    assert [candidate.experiment_id for candidate in pool.candidates] == ["candidate-a"]
    rejected_by_id = {item.experiment_id: item.reason for item in pool.rejected}
    assert "status is not completed" in rejected_by_id["candidate-failed"]
    assert "oof_path does not exist" in rejected_by_id["candidate-missing"]


def test_ensemble_runner_blocks_misaligned_test_predictions(tmp_path):
    artifact_root = tmp_path / "artifacts"
    competition = "churn_tiny"
    rows = [
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-a",
            oof_predictions=[0.1, 0.8, 0.3, 0.9],
            test_predictions=[0.2, 0.7],
        ),
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-b",
            oof_predictions=[0.2, 0.7, 0.4, 0.8],
            test_predictions=[0.3, 0.6],
            test_ids=[102, 101],
        ),
    ]
    _write_registry(artifact_root, competition, rows)
    config_path = _write_ensemble_config(tmp_path / "ensemble.yaml", artifact_root=artifact_root)

    try:
        run_ensemble_from_config(config_path)
    except ValueError as exc:
        assert "test ids do not align" in str(exc)
    else:
        raise AssertionError("misaligned test predictions should block ensemble creation")


def test_ensemble_runner_blocks_missing_explicit_candidate_id(tmp_path):
    artifact_root = tmp_path / "artifacts"
    competition = "churn_tiny"
    row = _write_candidate(
        artifact_root=artifact_root,
        competition=competition,
        experiment_id="candidate-a",
        oof_predictions=[0.1, 0.8, 0.3, 0.9],
        test_predictions=[0.2, 0.7],
    )
    _write_registry(artifact_root, competition, [row])
    config_path = _write_ensemble_config(tmp_path / "ensemble.yaml", artifact_root=artifact_root)

    try:
        run_ensemble_from_config(config_path)
    except ValueError as exc:
        assert "requested candidates are not eligible" in str(exc)
        assert "candidate-b" in str(exc)
    else:
        raise AssertionError("missing explicit candidate id should block ensemble creation")


def test_weighted_average_rejects_negative_weights(tmp_path):
    artifact_root = tmp_path / "artifacts"
    competition = "churn_tiny"
    rows = [
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-a",
            oof_predictions=[0.1, 0.8, 0.3, 0.9],
            test_predictions=[0.2, 0.7],
        ),
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-b",
            oof_predictions=[0.2, 0.7, 0.4, 0.8],
            test_predictions=[0.3, 0.6],
        ),
    ]
    _write_registry(artifact_root, competition, rows)
    config_path = _write_ensemble_config(
        tmp_path / "ensemble.yaml",
        artifact_root=artifact_root,
        method="weighted_average",
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["ensemble"]["weights"] = {"candidate-a": -0.5, "candidate-b": 1.5}
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    try:
        run_ensemble_from_config(config_path)
    except ValueError as exc:
        assert "Ensemble weights must be non-negative" in str(exc)
    else:
        raise AssertionError("negative weights should block ensemble creation")


def test_weighted_average_rejects_missing_extra_and_non_finite_weights(tmp_path):
    cases = [
        ({"candidate-a": 1.0}, "Missing weights for candidates"),
        (
            {"candidate-a": 0.5, "candidate-b": 0.5, "candidate-c": 0.1},
            "Unexpected weights for candidates",
        ),
        ({"candidate-a": 0.5, "candidate-b": float("inf")}, "Ensemble weights must be finite"),
    ]
    for index, (weights, expected_message) in enumerate(cases):
        artifact_root = tmp_path / f"artifacts-{index}"
        _write_two_candidate_registry(artifact_root)
        config_path = _write_ensemble_config(
            tmp_path / f"ensemble-{index}.yaml",
            artifact_root=artifact_root,
            method="weighted_average",
        )
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload["ensemble"]["weights"] = weights
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

        try:
            run_ensemble_from_config(config_path)
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError(f"{expected_message} should block ensemble creation")


def test_ensemble_runner_blocks_explicit_candidate_rejected_by_gate(tmp_path):
    artifact_root = tmp_path / "artifacts"
    competition = "churn_tiny"
    accepted = _write_candidate(
        artifact_root=artifact_root,
        competition=competition,
        experiment_id="candidate-a",
        oof_predictions=[0.1, 0.8, 0.3, 0.9],
        test_predictions=[0.2, 0.7],
    )
    rejected = _write_candidate(
        artifact_root=artifact_root,
        competition=competition,
        experiment_id="candidate-b",
        oof_predictions=[0.2, 0.7, 0.4, 0.8],
        test_predictions=[0.3, 0.6],
        status="failed",
    )
    _write_registry(artifact_root, competition, [accepted, rejected])
    config_path = _write_ensemble_config(tmp_path / "ensemble.yaml", artifact_root=artifact_root)

    try:
        run_ensemble_from_config(config_path)
    except ValueError as exc:
        assert "requested candidates are not eligible" in str(exc)
        assert "candidate-b" in str(exc)
    else:
        raise AssertionError("explicit rejected candidate should block ensemble creation")


def test_rank_average_cli_writes_rank_averaged_predictions(tmp_path):
    artifact_root = tmp_path / "artifacts"
    competition = _write_two_candidate_registry(artifact_root)
    config_path = _write_ensemble_config(
        tmp_path / "ensemble.yaml",
        artifact_root=artifact_root,
        method="rank_average",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["ensemble", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    oof = pd.read_parquet(
        artifact_root / "oof" / competition / "p03-ensemble-smoke" / "oof.parquet"
    )
    submission = pd.read_csv(
        artifact_root / "submissions" / competition / "p03-ensemble-smoke" / "submission.csv"
    )
    assert oof["prediction"].round(6).tolist() == [0.25, 0.75, 0.5, 1.0]
    assert submission["Churn"].round(6).tolist() == [0.5, 1.0]


def test_weighted_average_cli_writes_weighted_predictions(tmp_path):
    artifact_root = tmp_path / "artifacts"
    competition = _write_two_candidate_registry(artifact_root)
    config_path = _write_ensemble_config(
        tmp_path / "ensemble.yaml",
        artifact_root=artifact_root,
        method="weighted_average",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["ensemble", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    oof = pd.read_parquet(
        artifact_root / "oof" / competition / "p03-ensemble-smoke" / "oof.parquet"
    )
    submission = pd.read_csv(
        artifact_root / "submissions" / competition / "p03-ensemble-smoke" / "submission.csv"
    )
    assert oof["prediction"].round(6).tolist() == [0.175, 0.725, 0.375, 0.825]
    assert submission["Churn"].round(6).tolist() == [0.275, 0.625]


def test_cli_ensemble_writes_oof_submission_manifest_and_registry(tmp_path):
    artifact_root = tmp_path / "artifacts"
    competition = "churn_tiny"
    rows = [
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-a",
            oof_predictions=[0.1, 0.8, 0.3, 0.9],
            test_predictions=[0.2, 0.7],
            score=0.72,
        ),
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-b",
            oof_predictions=[0.2, 0.7, 0.4, 0.8],
            test_predictions=[0.3, 0.6],
            score=0.71,
            model_family="lightgbm",
        ),
    ]
    _write_registry(artifact_root, competition, rows)
    config_path = _write_ensemble_config(tmp_path / "ensemble.yaml", artifact_root=artifact_root)
    runner = CliRunner()

    result = runner.invoke(app, ["ensemble", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    assert "completed" in result.stdout.lower()
    ensemble_dir = artifact_root / "experiments" / competition / "p03-ensemble-smoke"
    oof_path = artifact_root / "oof" / competition / "p03-ensemble-smoke" / "oof.parquet"
    submission_path = (
        artifact_root / "submissions" / competition / "p03-ensemble-smoke" / "submission.csv"
    )
    assert oof_path.exists()
    assert submission_path.exists()
    assert (ensemble_dir / "ensemble_manifest.json").exists()
    assert (ensemble_dir / "selection_report.md").exists()

    oof = pd.read_parquet(oof_path)
    submission = pd.read_csv(submission_path)
    assert oof["prediction"].round(6).tolist() == [0.15, 0.75, 0.35, 0.85]
    assert submission["Churn"].round(6).tolist() == [0.25, 0.65]

    registry = pd.read_csv(artifact_root / "registry" / competition / "experiment_registry.csv")
    ensemble_row = registry.loc[registry["experiment_id"] == "p03-ensemble-smoke"].iloc[0]
    assert ensemble_row["model_family"] == "ensemble"
    assert ensemble_row["model_preset"] == "simple_average"
    assert ensemble_row["parent_experiment_ids"] == "candidate-a,candidate-b"


def test_ensemble_config_rejects_unsafe_experiment_id(tmp_path):
    artifact_root = tmp_path / "artifacts"
    _write_two_candidate_registry(artifact_root)
    config_path = _write_ensemble_config(tmp_path / "ensemble.yaml", artifact_root=artifact_root)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["ensemble"]["experiment_id"] = "../escape"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    try:
        run_ensemble_from_config(config_path)
    except ValueError as exc:
        assert "experiment_id" in str(exc)
    else:
        raise AssertionError("unsafe experiment id should be rejected")
