from pathlib import Path

import yaml

from ktabforge.proposals.config import load_proposal_config


def test_load_proposal_config_returns_typed_fields(tmp_path):
    config_path = tmp_path / "proposal.yaml"
    payload = {
        "proposal": {
            "proposal_id": "p06-freq-count",
            "competition": "churn_tiny",
            "artifact_root": "artifacts",
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
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = load_proposal_config(config_path)

    assert config.proposal_id == "p06-freq-count"
    assert config.competition == "churn_tiny"
    assert config.artifact_root == Path("artifacts")
    assert config.owner == "eda_scan"
    assert config.feature_id == "freq-contract"
    assert config.feature_family == "frequency_count"
    assert config.source_columns == ["Contract"]
    assert config.hypothesis.startswith("Frequency encoding")
    assert config.validation_plan == "compare OOF against baseline"
    assert config.requires_target is False
    assert config.fold_safety == "global_safe"
    assert config.leakage_risk == "low"
    assert config.transductive_risk == "low"
    assert config.config_path == config_path
