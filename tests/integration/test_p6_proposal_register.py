import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from ktabforge.cli import app


def _write_proposal_config(path: Path, *, artifact_root: Path) -> Path:
    payload = {
        "proposal": {
            "proposal_id": "p06-freq-count",
            "competition": "churn_tiny",
            "artifact_root": str(artifact_root),
            "owner": "eda_scan",
        },
        "feature": {
            "feature_id": "freq-contract",
            "feature_family": "frequency_count",
            "source_columns": ["Contract"],
            "hypothesis": "Frequency encoding on Contract may expose density effects.",
            "validation_plan": "compare OOF against baseline",
            "requires_target": False,
            "fold_safety": "global_safe",
            "leakage_risk": "low",
            "transductive_risk": "low",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_cli_proposal_register_writes_manifest_and_config_snapshot(tmp_path):
    artifact_root = tmp_path / "artifacts"
    config_path = _write_proposal_config(tmp_path / "proposal.yaml", artifact_root=artifact_root)
    runner = CliRunner()

    result = runner.invoke(app, ["proposal", "register", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    assert "registered" in result.stdout.lower()
    assert "p06-freq-count" in result.stdout

    artifact_dir = artifact_root / "proposals" / "churn_tiny" / "p06-freq-count"
    manifest_path = artifact_dir / "proposal_manifest.json"
    snapshot_path = artifact_dir / "proposal_config.yaml"

    assert manifest_path.exists()
    assert snapshot_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "registered"
    assert manifest["proposal_id"] == "p06-freq-count"
    assert manifest["competition"] == "churn_tiny"
    assert manifest["feature_id"] == "freq-contract"
    assert manifest["feature_family"] == "frequency_count"
    assert manifest["artifact_dir"] == str(artifact_dir)
    assert snapshot["proposal"]["proposal_id"] == "p06-freq-count"
    assert snapshot["feature"]["feature_id"] == "freq-contract"
