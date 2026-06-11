from pathlib import Path

from ktabforge.config.schema import validate_config_file


def test_competition_example_matches_schema():
    repo_root = Path(__file__).resolve().parents[2]

    result = validate_config_file(
        repo_root / "configs" / "competition.example.yaml",
        repo_root / "configs" / "schemas" / "competition.schema.json",
    )

    assert result.valid is True
    assert result.errors == []


def test_experiment_example_matches_schema():
    repo_root = Path(__file__).resolve().parents[2]

    result = validate_config_file(
        repo_root / "configs" / "experiment.example.yaml",
        repo_root / "configs" / "schemas" / "experiment.schema.json",
    )

    assert result.valid is True
    assert result.errors == []


def test_ensemble_example_matches_schema():
    repo_root = Path(__file__).resolve().parents[2]

    result = validate_config_file(
        repo_root / "configs" / "ensembles" / "p03_candidate_ensemble.example.yaml",
        repo_root / "configs" / "schemas" / "ensemble.schema.json",
    )

    assert result.valid is True
    assert result.errors == []


def test_stacking_example_matches_schema():
    repo_root = Path(__file__).resolve().parents[2]

    result = validate_config_file(
        repo_root / "configs" / "stacking.example.yaml",
        repo_root / "configs" / "schemas" / "stacking.schema.json",
    )

    assert result.valid is True
    assert result.errors == []


def test_eda_example_matches_schema():
    repo_root = Path(__file__).resolve().parents[2]

    result = validate_config_file(
        repo_root / "configs" / "eda" / "p06_churn_eda.example.yaml",
        repo_root / "configs" / "schemas" / "eda.schema.json",
    )

    assert result.valid is True
    assert result.errors == []


def test_proposal_example_matches_schema():
    repo_root = Path(__file__).resolve().parents[2]

    result = validate_config_file(
        repo_root / "configs" / "proposals" / "p06_freq_count.example.yaml",
        repo_root / "configs" / "schemas" / "proposal.schema.json",
    )

    assert result.valid is True
    assert result.errors == []


def test_feature_pipeline_example_matches_schema():
    repo_root = Path(__file__).resolve().parents[2]

    result = validate_config_file(
        repo_root / "configs" / "features" / "p06_freq_bundle.example.yaml",
        repo_root / "configs" / "schemas" / "feature_pipeline.schema.json",
    )

    assert result.valid is True
    assert result.errors == []
