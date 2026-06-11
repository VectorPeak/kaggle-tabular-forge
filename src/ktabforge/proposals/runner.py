from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ktabforge.artifacts.writers import write_json
from ktabforge.proposals.config import ProposalConfig, load_proposal_config
from ktabforge.utils.hashing import stable_hash
from ktabforge.utils.time import utc_now_iso


@dataclass(frozen=True)
class ProposalResult:
    status: str
    proposal_id: str
    competition: str
    artifact_dir: Path
    manifest_path: Path | None = None
    config_snapshot_path: Path | None = None


def validate_proposal_config(config_path: str | Path) -> ProposalResult:
    config = load_proposal_config(config_path)
    artifact_dir = _artifact_dir(config)
    return ProposalResult(
        status="valid",
        proposal_id=config.proposal_id,
        competition=config.competition,
        artifact_dir=artifact_dir,
    )


def register_proposal_from_config(config_path: str | Path) -> ProposalResult:
    config = load_proposal_config(config_path)
    artifact_dir = _artifact_dir(config)
    if artifact_dir.exists():
        raise FileExistsError(
            f"Proposal artifact directory already exists; refusing to overwrite: {artifact_dir}"
        )
    artifact_dir.mkdir(parents=True, exist_ok=False)

    manifest_path = artifact_dir / "proposal_manifest.json"
    config_snapshot_path = artifact_dir / "proposal_config.yaml"

    write_json(
        manifest_path,
        {
            "status": "registered",
            "proposal_id": config.proposal_id,
            "competition": config.competition,
            "artifact_dir": str(artifact_dir),
            "config_path": str(config.config_path),
            "config_hash": stable_hash(config.raw),
            "created_at": utc_now_iso(),
            "owner": config.owner,
            "feature_id": config.feature_id,
            "feature_family": config.feature_family,
            "source_columns": config.source_columns,
            "hypothesis": config.hypothesis,
            "validation_plan": config.validation_plan,
            "requires_target": config.requires_target,
            "fold_safety": config.fold_safety,
            "leakage_risk": config.leakage_risk,
            "transductive_risk": config.transductive_risk,
        },
    )
    config_snapshot_path.write_text(
        yaml.safe_dump(config.raw, sort_keys=False),
        encoding="utf-8",
    )

    return ProposalResult(
        status="registered",
        proposal_id=config.proposal_id,
        competition=config.competition,
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        config_snapshot_path=config_snapshot_path,
    )


def _artifact_dir(config: ProposalConfig) -> Path:
    return config.artifact_root / "proposals" / config.competition / config.proposal_id
