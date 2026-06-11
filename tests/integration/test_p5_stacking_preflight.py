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
    id_column: str = "id",
    target: str = "Churn",
    seed: int = 42,
    target_values: list[int] | None = None,
    metric_name: str = "roc_auc",
) -> dict[str, object]:
    oof_ids = oof_ids or [1, 2, 3, 4]
    folds = folds or [0, 0, 1, 1]
    target_values = target_values or [0, 1, 0, 1]
    oof_path = artifact_root / "oof" / competition / experiment_id / "oof.parquet"
    test_path = artifact_root / "submissions" / competition / experiment_id / "submission.csv"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            id_column: oof_ids,
            target: target_values,
            "prediction": oof_predictions,
            "fold": folds,
        }
    ).to_parquet(oof_path, index=False)
    pd.DataFrame({id_column: [101, 102], target: test_predictions}).to_csv(test_path, index=False)
    return {
        "experiment_id": experiment_id,
        "competition": competition,
        "metric_name": metric_name,
        "metric_mode": "min" if metric_name in {"log_loss", "rmse", "mae"} else "max",
        "oof_score": 0.75,
        "status": "completed",
        "oof_path": str(oof_path),
        "test_pred_path": str(test_path),
        "fold_metrics_path": "",
        "model_family": model_family,
        "seed": seed,
        "run_mode": "run",
        "reason": "ok",
        "model_preset": "baseline",
        "feature_set": "basic",
        "config_hash": experiment_id,
        "created_at": "2026-06-11T00:00:00Z",
        "feature_manifest_hash": experiment_id,
        "prediction_type": "probability",
        "id_column": id_column,
        "target": target,
    }


def _write_registry(artifact_root: Path, competition: str, rows: list[dict[str, object]]) -> None:
    registry_path = artifact_root / "registry" / competition / "experiment_registry.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(registry_path, index=False)


def _write_stacking_config(
    path: Path,
    *,
    artifact_root: Path,
    experiment_id: str = "p05-stack-preflight",
    competition: str = "churn_tiny",
    target: str = "Churn",
    id_column: str = "id",
    metric_name: str = "roc_auc",
    candidate_ids: list[str] | None = None,
    top_n: int = 3,
    max_parents: int = 2,
    min_parents: int = 2,
    stacker_method: str = "logistic_regression",
    stacker_params: dict[str, object] | None = None,
    selection: dict[str, object] | None = None,
) -> Path:
    stacking_payload = {
        "experiment_id": experiment_id,
        "competition": competition,
        "artifact_root": str(artifact_root),
        "target": target,
        "id_column": id_column,
        "metric_name": metric_name,
        "candidate_ids": candidate_ids or ["candidate-a", "candidate-b", "candidate-c"],
        "top_n": top_n,
        "max_parents": max_parents,
        "min_parents": min_parents,
        "stacker": {
            "method": stacker_method,
            "params": stacker_params or {"C": 1.0, "max_iter": 200},
        },
    }
    if selection is not None:
        stacking_payload["selection"] = selection

    payload = {"stacking": stacking_payload}
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


def test_cli_stack_writes_completed_oof_submission_and_registry(tmp_path):
    artifact_root = tmp_path / "artifacts"
    competition = "churn_tiny"
    rows = [
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-a",
            oof_predictions=[0.05, 0.9, 0.2, 0.85],
            test_predictions=[0.15, 0.8],
            model_family="lightgbm",
        ),
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-b",
            oof_predictions=[0.2, 0.75, 0.35, 0.8],
            test_predictions=[0.3, 0.65],
            model_family="xgboost",
        ),
    ]
    _write_registry(artifact_root, competition, rows)
    config_path = _write_stacking_config(tmp_path / "stacking.yaml", artifact_root=artifact_root)
    runner = CliRunner()

    result = runner.invoke(app, ["stack", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    assert "completed" in result.stdout.lower()

    experiment_dir = artifact_root / "experiments" / competition / "p05-stack-preflight"
    oof_path = artifact_root / "oof" / competition / "p05-stack-preflight" / "oof.parquet"
    submission_path = (
        artifact_root / "submissions" / competition / "p05-stack-preflight" / "submission.csv"
    )
    stack_oof_path = (
        artifact_root / "oof" / competition / "p05-stack-preflight" / "stack_oof.parquet"
    )
    stack_test_path = (
        artifact_root / "submissions" / competition / "p05-stack-preflight" / "stack_test.parquet"
    )

    assert oof_path.exists()
    assert submission_path.exists()
    assert stack_oof_path.exists()
    assert stack_test_path.exists()
    assert (experiment_dir / "stacking_manifest.json").exists()
    assert (experiment_dir / "selection_report.md").exists()
    assert (experiment_dir / "fold_metrics.csv").exists()
    assert (experiment_dir / "metrics.json").exists()
    assert (experiment_dir / "run_manifest.json").exists()

    oof = pd.read_parquet(oof_path)
    submission = pd.read_csv(submission_path)
    assert oof.columns.tolist() == ["id", "Churn", "prediction", "fold"]
    assert submission.columns.tolist() == ["id", "Churn"]
    assert len(oof) == 4
    assert len(submission) == 2
    assert oof["prediction"].between(0, 1).all()
    assert submission["Churn"].between(0, 1).all()

    registry = pd.read_csv(artifact_root / "registry" / competition / "experiment_registry.csv")
    stack_row = registry.loc[registry["experiment_id"] == "p05-stack-preflight"].iloc[0]
    assert stack_row["status"] == "completed"
    assert stack_row["model_family"] == "stacking"
    assert stack_row["model_preset"] == "logistic_regression"
    assert stack_row["parent_experiment_ids"] == "candidate-a,candidate-b"


def test_cli_stack_honors_custom_id_column_and_stacker_seed(tmp_path):
    artifact_root = tmp_path / "artifacts"
    competition = "churn_tiny"
    rows = [
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-a",
            oof_predictions=[0.1, 0.85, 0.25, 0.9],
            test_predictions=[0.2, 0.75],
            model_family="lightgbm",
            id_column="customer_id",
            seed=11,
        ),
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-b",
            oof_predictions=[0.2, 0.8, 0.35, 0.82],
            test_predictions=[0.3, 0.7],
            model_family="xgboost",
            id_column="customer_id",
            seed=11,
        ),
    ]
    _write_registry(artifact_root, competition, rows)
    config_path = _write_stacking_config(
        tmp_path / "stacking_custom_id.yaml",
        artifact_root=artifact_root,
        experiment_id="p05-stack-custom-id",
        id_column="customer_id",
        candidate_ids=["candidate-a", "candidate-b"],
        stacker_params={"C": 0.5, "max_iter": 200, "seed": 777},
    )
    runner = CliRunner()

    result = runner.invoke(app, ["stack", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    oof_path = artifact_root / "oof" / competition / "p05-stack-custom-id" / "oof.parquet"
    submission_path = (
        artifact_root / "submissions" / competition / "p05-stack-custom-id" / "submission.csv"
    )
    run_manifest_path = (
        artifact_root / "experiments" / competition / "p05-stack-custom-id" / "run_manifest.json"
    )
    oof = pd.read_parquet(oof_path)
    submission = pd.read_csv(submission_path)
    manifest = yaml.safe_load(run_manifest_path.read_text(encoding="utf-8"))

    assert oof.columns.tolist() == ["customer_id", "Churn", "prediction", "fold"]
    assert submission.columns.tolist() == ["customer_id", "Churn"]
    assert manifest["id_column"] == "customer_id"
    assert manifest["seed"] == 777

    registry = pd.read_csv(artifact_root / "registry" / competition / "experiment_registry.csv")
    stack_row = registry.loc[registry["experiment_id"] == "p05-stack-custom-id"].iloc[0]
    assert int(stack_row["seed"]) == 777


def test_cli_stack_rejects_preflight_only_stacker(tmp_path):
    artifact_root = tmp_path / "artifacts"
    competition = "churn_tiny"
    rows = [
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-a",
            oof_predictions=[0.05, 0.9, 0.2, 0.85],
            test_predictions=[0.15, 0.8],
            model_family="lightgbm",
        ),
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-b",
            oof_predictions=[0.2, 0.75, 0.35, 0.8],
            test_predictions=[0.3, 0.65],
            model_family="xgboost",
        ),
    ]
    _write_registry(artifact_root, competition, rows)
    config_path = _write_stacking_config(
        tmp_path / "stacking_preflight_only.yaml",
        artifact_root=artifact_root,
        stacker_method="preflight_only",
        stacker_params={},
        candidate_ids=["candidate-a", "candidate-b"],
    )
    runner = CliRunner()

    result = runner.invoke(app, ["stack", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "preflight_only" in result.stdout


def test_cli_stack_preflight_diversity_selection_filters_high_correlation_candidate(tmp_path):
    artifact_root = tmp_path / "artifacts"
    competition = "churn_tiny"
    target_values = [0, 1, 0, 1, 0, 1]
    oof_ids = [1, 2, 3, 4, 5, 6]
    folds = [0, 0, 1, 1, 2, 2]
    rows = [
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-a",
            oof_predictions=[0.10, 0.90, 0.20, 0.80, 0.15, 0.85],
            test_predictions=[0.20, 0.80],
            model_family="lightgbm",
            oof_ids=oof_ids,
            folds=folds,
            target_values=target_values,
        ),
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-b",
            oof_predictions=[0.11, 0.89, 0.21, 0.79, 0.14, 0.86],
            test_predictions=[0.21, 0.79],
            model_family="xgboost",
            oof_ids=oof_ids,
            folds=folds,
            target_values=target_values,
        ),
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-c",
            oof_predictions=[0.30, 0.78, 0.25, 0.72, 0.45, 0.88],
            test_predictions=[0.35, 0.75],
            model_family="catboost",
            oof_ids=oof_ids,
            folds=folds,
            target_values=target_values,
        ),
    ]
    _write_registry(artifact_root, competition, rows)
    config_path = _write_stacking_config(
        tmp_path / "stacking_diversity.yaml",
        artifact_root=artifact_root,
        experiment_id="p05-stack-diversity",
        candidate_ids=["candidate-a", "candidate-b", "candidate-c"],
        selection={
            "strategy": "diversity_greedy",
            "max_pairwise_corr": 0.995,
            "report_top_k_pairs": 5,
        },
    )
    runner = CliRunner()

    result = runner.invoke(app, ["stack-preflight", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    stack_oof_path = (
        artifact_root / "oof" / competition / "p05-stack-diversity" / "stack_oof.parquet"
    )
    report_path = (
        artifact_root
        / "experiments"
        / competition
        / "p05-stack-diversity"
        / "selection_report.md"
    )
    manifest_path = (
        artifact_root
        / "experiments"
        / competition
        / "p05-stack-diversity"
        / "stacking_manifest.json"
    )

    stack_oof = pd.read_parquet(stack_oof_path)
    report = report_path.read_text(encoding="utf-8")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert stack_oof.columns.tolist() == ["id", "Churn", "fold", "candidate-a", "candidate-c"]
    assert "candidate-b" in report
    assert "max_pairwise_corr" in report
    assert "Top Correlated Pairs" in report
    assert "candidate-a" in report and "candidate-b" in report
    assert manifest["accepted_candidate_ids"] == ["candidate-a", "candidate-c"]
    assert manifest["selection_strategy"] == "diversity_greedy"
    assert manifest["selection_max_pairwise_corr"] == 0.995


def test_cli_stack_preflight_hill_climb_selection_reports_trace(tmp_path):
    artifact_root = tmp_path / "artifacts"
    competition = "churn_tiny"
    target_values = [0, 0, 0, 1, 1, 1]
    oof_ids = [1, 2, 3, 4, 5, 6]
    folds = [0, 0, 1, 1, 2, 2]
    rows = [
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-a",
            oof_predictions=[0.3875, 0.1827, 0.1269, 0.7535, 0.6218, 0.8165],
            test_predictions=[0.30, 0.75],
            model_family="lightgbm",
            oof_ids=oof_ids,
            folds=folds,
            target_values=target_values,
            metric_name="log_loss",
        ),
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-b",
            oof_predictions=[0.3976, 0.2670, 0.3975, 0.8758, 0.8026, 0.8676],
            test_predictions=[0.35, 0.82],
            model_family="xgboost",
            oof_ids=oof_ids,
            folds=folds,
            target_values=target_values,
            metric_name="log_loss",
        ),
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-c",
            oof_predictions=[0.4453, 0.2794, 0.6253, 0.7271, 0.7707, 0.5872],
            test_predictions=[0.40, 0.70],
            model_family="catboost",
            oof_ids=oof_ids,
            folds=folds,
            target_values=target_values,
            metric_name="log_loss",
        ),
    ]
    _write_registry(artifact_root, competition, rows)
    config_path = _write_stacking_config(
        tmp_path / "stacking_hill_climb.yaml",
        artifact_root=artifact_root,
        experiment_id="p05-stack-hill-climb",
        metric_name="log_loss",
        candidate_ids=["candidate-a", "candidate-b", "candidate-c"],
        selection={
            "strategy": "hill_climb_greedy",
            "min_gain": 0.0,
            "report_top_k_pairs": 5,
        },
    )
    runner = CliRunner()

    result = runner.invoke(app, ["stack-preflight", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    stack_oof_path = (
        artifact_root / "oof" / competition / "p05-stack-hill-climb" / "stack_oof.parquet"
    )
    report_path = (
        artifact_root
        / "experiments"
        / competition
        / "p05-stack-hill-climb"
        / "selection_report.md"
    )
    manifest_path = (
        artifact_root
        / "experiments"
        / competition
        / "p05-stack-hill-climb"
        / "stacking_manifest.json"
    )

    stack_oof = pd.read_parquet(stack_oof_path)
    report = report_path.read_text(encoding="utf-8")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert stack_oof.columns.tolist() == ["id", "Churn", "fold", "candidate-a", "candidate-b"]
    assert "Hill Climb Trace" in report
    assert "candidate-c" in report
    assert "min_gain" in report
    assert manifest["accepted_candidate_ids"] == ["candidate-a", "candidate-b"]
    assert manifest["selection_strategy"] == "hill_climb_greedy"
    assert [step["experiment_id"] for step in manifest["hill_climb_trace"]] == [
        "candidate-a",
        "candidate-b",
    ]


def test_cli_stack_preflight_top_n_is_applied_after_compatibility_and_selection(tmp_path):
    artifact_root = tmp_path / "artifacts"
    competition = "churn_tiny"
    target_values = [0, 1, 0, 1, 0, 1]
    oof_ids = [1, 2, 3, 4, 5, 6]
    folds = [0, 0, 1, 1, 2, 2]
    rows = [
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-a",
            oof_predictions=[0.10, 0.90, 0.20, 0.80, 0.15, 0.85],
            test_predictions=[0.20, 0.80],
            model_family="lightgbm",
            oof_ids=oof_ids,
            folds=folds,
            target_values=target_values,
        ),
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-b",
            oof_predictions=[0.11, 0.89, 0.21, 0.79, 0.14, 0.86],
            test_predictions=[0.21, 0.79],
            model_family="xgboost",
            oof_ids=oof_ids,
            folds=folds,
            target_values=target_values,
        ),
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-c",
            oof_predictions=[0.30, 0.78, 0.25, 0.72, 0.45, 0.88],
            test_predictions=[0.35, 0.75],
            model_family="catboost",
            oof_ids=oof_ids,
            folds=folds,
            target_values=target_values,
        ),
    ]
    _write_registry(artifact_root, competition, rows)
    config_path = _write_stacking_config(
        tmp_path / "stacking_diversity_top_n.yaml",
        artifact_root=artifact_root,
        experiment_id="p05-stack-diversity-top-n",
        candidate_ids=["candidate-a", "candidate-b", "candidate-c"],
        top_n=2,
        max_parents=2,
        min_parents=2,
        selection={
            "strategy": "diversity_greedy",
            "max_pairwise_corr": 0.995,
            "report_top_k_pairs": 5,
        },
    )
    runner = CliRunner()

    result = runner.invoke(app, ["stack-preflight", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    stack_oof_path = (
        artifact_root / "oof" / competition / "p05-stack-diversity-top-n" / "stack_oof.parquet"
    )
    manifest_path = (
        artifact_root
        / "experiments"
        / competition
        / "p05-stack-diversity-top-n"
        / "stacking_manifest.json"
    )
    stack_oof = pd.read_parquet(stack_oof_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert stack_oof.columns.tolist() == ["id", "Churn", "fold", "candidate-a", "candidate-c"]
    assert manifest["accepted_candidate_ids"] == ["candidate-a", "candidate-c"]
    rejected_by_id = {
        item["experiment_id"]: item["reason"] for item in manifest["rejected_candidates"]
    }
    assert "candidate-b" in rejected_by_id
    assert "max_pairwise_corr" in rejected_by_id["candidate-b"]


def test_cli_stack_preflight_does_not_mark_top_n_trimmed_candidate_as_missing_from_registry(
    tmp_path,
):
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
    ]
    _write_registry(artifact_root, competition, rows)
    config_path = _write_stacking_config(
        tmp_path / "stacking_top_n_explicit_ids.yaml",
        artifact_root=artifact_root,
        experiment_id="p05-stack-top-n-explicit-ids",
        candidate_ids=["candidate-a", "candidate-b"],
        top_n=1,
        max_parents=1,
        min_parents=1,
    )
    runner = CliRunner()

    result = runner.invoke(app, ["stack-preflight", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    manifest_path = (
        artifact_root
        / "experiments"
        / competition
        / "p05-stack-top-n-explicit-ids"
        / "stacking_manifest.json"
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    rejected_reasons = [item["reason"] for item in manifest["rejected_candidates"]]
    assert all("not found in registry" not in reason for reason in rejected_reasons)


def test_cli_stack_preflight_hill_climb_respects_max_parents_without_post_trim(tmp_path):
    artifact_root = tmp_path / "artifacts"
    competition = "churn_tiny"
    target_values = [0, 0, 0, 1, 1, 1]
    oof_ids = [1, 2, 3, 4, 5, 6]
    folds = [0, 0, 1, 1, 2, 2]
    rows = [
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-a",
            oof_predictions=[0.3875, 0.1827, 0.1269, 0.7535, 0.6218, 0.8165],
            test_predictions=[0.30, 0.75],
            model_family="lightgbm",
            oof_ids=oof_ids,
            folds=folds,
            target_values=target_values,
            metric_name="log_loss",
        ),
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-b",
            oof_predictions=[0.3976, 0.2670, 0.3975, 0.8758, 0.8026, 0.8676],
            test_predictions=[0.35, 0.82],
            model_family="xgboost",
            oof_ids=oof_ids,
            folds=folds,
            target_values=target_values,
            metric_name="log_loss",
        ),
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-c",
            oof_predictions=[0.4453, 0.2794, 0.6253, 0.7271, 0.7707, 0.5872],
            test_predictions=[0.40, 0.70],
            model_family="catboost",
            oof_ids=oof_ids,
            folds=folds,
            target_values=target_values,
            metric_name="log_loss",
        ),
    ]
    _write_registry(artifact_root, competition, rows)
    config_path = _write_stacking_config(
        tmp_path / "stacking_hill_climb_trimmed.yaml",
        artifact_root=artifact_root,
        experiment_id="p05-stack-hill-climb-trimmed",
        metric_name="log_loss",
        candidate_ids=["candidate-a", "candidate-b", "candidate-c"],
        max_parents=1,
        min_parents=1,
        selection={
            "strategy": "hill_climb_greedy",
            "min_gain": 0.0,
            "report_top_k_pairs": 5,
        },
    )
    runner = CliRunner()

    result = runner.invoke(app, ["stack-preflight", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    stack_oof_path = (
        artifact_root / "oof" / competition / "p05-stack-hill-climb-trimmed" / "stack_oof.parquet"
    )
    manifest_path = (
        artifact_root
        / "experiments"
        / competition
        / "p05-stack-hill-climb-trimmed"
        / "stacking_manifest.json"
    )

    stack_oof = pd.read_parquet(stack_oof_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert stack_oof.columns.tolist() == ["id", "Churn", "fold", "candidate-a"]
    assert manifest["accepted_candidate_ids"] == ["candidate-a"]
    assert [step["experiment_id"] for step in manifest["hill_climb_trace"]] == ["candidate-a"]
    rejected_by_id = {
        item["experiment_id"]: item["reason"] for item in manifest["rejected_candidates"]
    }
    assert all("trimmed by max_parents" not in reason for reason in rejected_by_id.values())


def test_cli_stack_preflight_rejects_hill_climb_with_max_pairwise_corr_in_reporting(
    tmp_path,
):
    artifact_root = tmp_path / "artifacts"
    competition = "churn_tiny"
    target_values = [0, 0, 0, 1, 1, 1]
    oof_ids = [1, 2, 3, 4, 5, 6]
    folds = [0, 0, 1, 1, 2, 2]
    rows = [
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-a",
            oof_predictions=[0.3875, 0.1827, 0.1269, 0.7535, 0.6218, 0.8165],
            test_predictions=[0.30, 0.75],
            model_family="lightgbm",
            oof_ids=oof_ids,
            folds=folds,
            target_values=target_values,
            metric_name="log_loss",
        ),
        _write_candidate(
            artifact_root=artifact_root,
            competition=competition,
            experiment_id="candidate-b",
            oof_predictions=[0.3976, 0.2670, 0.3975, 0.8758, 0.8026, 0.8676],
            test_predictions=[0.35, 0.82],
            model_family="xgboost",
            oof_ids=oof_ids,
            folds=folds,
            target_values=target_values,
            metric_name="log_loss",
        ),
    ]
    _write_registry(artifact_root, competition, rows)
    config_path = _write_stacking_config(
        tmp_path / "stacking_hill_climb_with_corr.yaml",
        artifact_root=artifact_root,
        experiment_id="p05-stack-hill-climb-with-corr",
        metric_name="log_loss",
        candidate_ids=["candidate-a", "candidate-b"],
        min_parents=1,
        max_parents=2,
        selection={
            "strategy": "hill_climb_greedy",
            "min_gain": 0.0,
            "max_pairwise_corr": 0.99,
            "report_top_k_pairs": 5,
        },
    )
    runner = CliRunner()

    result = runner.invoke(app, ["stack-preflight", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "max_pairwise_corr" in result.stdout
    assert "hill_climb_greedy" in result.stdout
