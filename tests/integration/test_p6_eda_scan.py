import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from ktabforge.cli import app


def _write_eda_config(path: Path, *, repo_root: Path, artifact_root: Path) -> Path:
    payload = {
        "eda": {
            "eda_id": "p06-eda-smoke",
            "competition": "churn_tiny",
            "artifact_root": str(artifact_root),
            "focus": [
                "drift",
                "leakage",
                "categorical_cardinality",
            ],
        },
        "data": {
            "data_dir": str(repo_root / "tests" / "fixtures" / "data" / "churn_tiny"),
            "target": "Churn",
            "id_column": "id",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_cli_eda_scan_writes_expected_artifacts_and_backlog(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    artifact_root = tmp_path / "artifacts"
    config_path = _write_eda_config(
        tmp_path / "eda.yaml",
        repo_root=repo_root,
        artifact_root=artifact_root,
    )
    runner = CliRunner()

    result = runner.invoke(app, ["eda", "scan", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    assert "completed" in result.stdout.lower()
    assert "p06-eda-smoke" in result.stdout

    eda_dir = artifact_root / "eda_findings" / "churn_tiny" / "p06-eda-smoke"
    manifest_path = eda_dir / "eda_manifest.json"
    summary_path = eda_dir / "eda_summary.md"
    watchlist_path = eda_dir / "leakage_watchlist.json"
    backlog_path = eda_dir / "feature_backlog.json"

    assert manifest_path.exists()
    assert summary_path.exists()
    assert watchlist_path.exists()
    assert backlog_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["eda_id"] == "p06-eda-smoke"
    assert manifest["competition"] == "churn_tiny"
    assert manifest["focus"] == ["drift", "leakage", "categorical_cardinality"]

    backlog = json.loads(backlog_path.read_text(encoding="utf-8"))
    feature_families = {item["feature_family"] for item in backlog["items"]}
    assert "frequency_count" in feature_families
    assert "binning" in feature_families
    assert "arithmetic_interactions" in feature_families
