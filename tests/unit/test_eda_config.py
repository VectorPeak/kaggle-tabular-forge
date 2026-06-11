from pathlib import Path

import yaml

from ktabforge.eda.config import load_eda_config


def test_load_eda_config_returns_typed_fields(tmp_path):
    config_path = tmp_path / "eda.yaml"
    payload = {
        "eda": {
            "eda_id": "p06-churn-eda",
            "competition": "churn_tiny",
            "artifact_root": "artifacts",
            "focus": ["drift", "leakage"],
        },
        "data": {
            "data_dir": "data/churn_tiny/raw",
            "target": "Churn",
            "id_column": "id",
        },
        "scan": {
            "profile_train": True,
            "profile_test": True,
            "include_drift": True,
            "include_leakage_watchlist": True,
        },
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = load_eda_config(config_path)

    assert config.eda_id == "p06-churn-eda"
    assert config.competition == "churn_tiny"
    assert config.artifact_root == Path("artifacts")
    assert config.data_dir == Path("data/churn_tiny/raw")
    assert config.target == "Churn"
    assert config.id_column == "id"
    assert config.focus == ["drift", "leakage"]
    assert config.profile_train is True
    assert config.profile_test is True
    assert config.include_drift is True
    assert config.include_leakage_watchlist is True
    assert config.config_path == config_path
