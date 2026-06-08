from pathlib import Path

import pandas as pd

from ktabforge.pipeline.evidence import run_smoke_evidence


def test_smoke_evidence_pipeline_writes_oof_submission_and_registry(tmp_path):
    data_dir = Path(__file__).resolve().parents[1] / "fixtures" / "data" / "churn_tiny"

    result = run_smoke_evidence(
        data_dir=data_dir,
        artifact_root=tmp_path / "artifacts",
        competition="churn_tiny",
        experiment_id="pytest-smoke",
        target="Churn",
        id_column="id",
        n_splits=3,
        seed=42,
    )

    assert result.status == "completed"
    assert result.oof_score is not None
    assert result.paths.experiment_dir.exists()
    assert result.paths.oof_path.exists()
    assert result.paths.submission_path.exists()
    assert result.paths.registry_path.exists()

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    oof = pd.read_parquet(result.paths.oof_path)
    submission = pd.read_csv(result.paths.submission_path)
    registry = pd.read_csv(result.paths.registry_path)

    assert len(oof) == len(train)
    assert set(oof.columns) == {"id", "Churn", "prediction", "fold"}
    assert oof["prediction"].between(0, 1).all()
    assert submission["id"].tolist() == test["id"].tolist()
    assert submission["Churn"].between(0, 1).all()
    assert registry.loc[0, "experiment_id"] == "pytest-smoke"
    assert registry.loc[0, "status"] == "completed"

