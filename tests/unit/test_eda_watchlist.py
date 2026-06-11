from __future__ import annotations

import json
from pathlib import Path

import yaml

from ktabforge.eda.runner import run_eda_scan_from_config
from ktabforge.eda.watchlist import build_leakage_watchlist


def test_build_leakage_watchlist_flags_target_in_test_duplicate_ids_and_id_overlap():
    import pandas as pd

    train = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "target": [0, 1, 0],
            "feature": [10.0, 20.0, 30.0],
        }
    )
    test = pd.DataFrame(
        {
            "id": [2, 3, 3],
            "target": [0, 0, 0],
            "feature": [11.0, 22.0, 33.0],
        }
    )

    watchlist = build_leakage_watchlist(train=train, test=test, target="target", id_column="id")
    reasons = {item["rule"] for item in watchlist}

    assert "target_present_in_test" in reasons
    assert "duplicate_train_ids" in reasons
    assert "duplicate_test_ids" in reasons
    assert "train_test_id_overlap" in reasons


def test_run_eda_scan_writes_watchlist_json_payload(tmp_path: Path):
    data_dir = tmp_path / "data"
    artifact_root = tmp_path / "artifacts"
    data_dir.mkdir()

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"

    train_path.write_text(
        "id,Churn,feature\n1,0,10\n1,1,20\n2,0,30\n",
        encoding="utf-8",
    )
    test_path.write_text(
        "id,Churn,feature\n2,0,15\n3,0,25\n3,0,35\n",
        encoding="utf-8",
    )
    sample_path.write_text(
        "id,Churn\n2,0.0\n3,0.0\n3,0.0\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "eda.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "eda": {
                    "eda_id": "p6-watchlist-unit",
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
                    "include_drift": False,
                    "include_leakage_watchlist": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = run_eda_scan_from_config(config_path)

    watchlist_path = result.artifact_dir / "leakage_watchlist.json"
    payload = json.loads(watchlist_path.read_text(encoding="utf-8"))
    rules = {item["rule"] for item in payload["items"]}

    assert payload["eda_id"] == "p6-watchlist-unit"
    assert payload["competition"] == "churn_tiny"
    assert "target_present_in_test" in rules
    assert "duplicate_train_ids" in rules
