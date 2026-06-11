from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ktabforge.config.loader import load_yaml_file
from ktabforge.config.safety import safe_path_segment
from ktabforge.config.schema import bundled_schema_path, require_loaded_config_valid


@dataclass(frozen=True)
class ProposalConfig:
    proposal_id: str
    competition: str
    artifact_root: Path
    owner: str | None
    feature_id: str
    feature_family: str
    source_columns: list[str]
    hypothesis: str
    validation_plan: str
    requires_target: bool
    fold_safety: str
    leakage_risk: str
    transductive_risk: str
    config_path: Path
    raw: dict[str, object]


def load_proposal_config(config_path: str | Path) -> ProposalConfig:
    path = Path(config_path)
    payload = load_yaml_file(path)
    if not isinstance(payload, dict):
        raise TypeError("Proposal config must be a YAML mapping.")
    require_loaded_config_valid(payload, bundled_schema_path("proposal.schema.json"))

    proposal = _mapping(payload, "proposal")
    feature = _mapping(payload, "feature")

    return ProposalConfig(
        proposal_id=safe_path_segment(
            _required_string(proposal, "proposal_id"),
            field="proposal_id",
        ),
        competition=safe_path_segment(
            _required_string(proposal, "competition"),
            field="competition",
        ),
        artifact_root=Path(_required_string(proposal, "artifact_root")),
        owner=_optional_string(proposal.get("owner")),
        feature_id=_required_string(feature, "feature_id"),
        feature_family=_required_string(feature, "feature_family"),
        source_columns=[str(item) for item in feature.get("source_columns", [])],
        hypothesis=_required_string(feature, "hypothesis"),
        validation_plan=_required_string(feature, "validation_plan"),
        requires_target=_required_bool(feature.get("requires_target")),
        fold_safety=_required_string(feature, "fold_safety"),
        leakage_risk=_required_string(feature, "leakage_risk"),
        transductive_risk=_required_string(feature, "transductive_risk"),
        config_path=path,
        raw=payload,
    )


def _mapping(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"Proposal config section {key!r} must be a mapping.")
    return value


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"Proposal config is missing required field {key!r}.")
    return str(value)


def _optional_string(value: object) -> str | None:
    if value is None or str(value) == "":
        return None
    return str(value)


def _required_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError("Proposal boolean fields must be booleans.")
    return value
