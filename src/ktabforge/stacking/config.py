from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ktabforge.config.loader import load_yaml_file
from ktabforge.config.safety import safe_path_segment
from ktabforge.config.schema import bundled_schema_path, require_loaded_config_valid


@dataclass(frozen=True)
class StackingSelectionConfig:
    strategy: str
    max_pairwise_corr: float | None
    report_top_k_pairs: int
    min_gain: float


@dataclass(frozen=True)
class StackingPreflightConfig:
    experiment_id: str
    competition: str
    artifact_root: Path
    target: str
    id_column: str
    metric_name: str
    candidate_ids: list[str]
    top_n: int | None
    max_parents: int | None
    min_parents: int
    selection: StackingSelectionConfig
    stacker_method: str
    stacker_params: dict[str, Any]
    config_path: Path
    raw: dict[str, object]


def load_stacking_config(config_path: str | Path) -> StackingPreflightConfig:
    path = Path(config_path)
    payload = load_yaml_file(path)
    if not isinstance(payload, dict):
        raise TypeError("Stacking config must be a YAML mapping.")
    require_loaded_config_valid(payload, bundled_schema_path("stacking.schema.json"))

    stacking = payload.get("stacking", {})
    if not isinstance(stacking, dict):
        raise TypeError("Stacking config section 'stacking' must be a mapping.")
    stacker = stacking.get("stacker", {})
    if not isinstance(stacker, dict):
        raise TypeError("Stacking config section 'stacker' must be a mapping when provided.")
    selection = stacking.get("selection", {})
    if not isinstance(selection, dict):
        raise TypeError("Stacking config section 'selection' must be a mapping when provided.")

    min_parents = _optional_int(stacking.get("min_parents"))
    return StackingPreflightConfig(
        experiment_id=safe_path_segment(
            _required_string(stacking, "experiment_id"),
            field="experiment_id",
        ),
        competition=safe_path_segment(
            _required_string(stacking, "competition"),
            field="competition",
        ),
        artifact_root=Path(_required_string(stacking, "artifact_root")),
        target=_required_string(stacking, "target"),
        id_column=str(stacking.get("id_column") or "id"),
        metric_name=str(stacking.get("metric_name") or "roc_auc"),
        candidate_ids=[str(item) for item in stacking.get("candidate_ids", [])],
        top_n=_optional_int(stacking.get("top_n")),
        max_parents=_optional_int(stacking.get("max_parents")),
        min_parents=min_parents if min_parents is not None else 2,
        selection=StackingSelectionConfig(
            strategy=str(selection.get("strategy") or "score_desc"),
            max_pairwise_corr=_optional_probability(selection.get("max_pairwise_corr")),
            report_top_k_pairs=_optional_int(selection.get("report_top_k_pairs")) or 20,
            min_gain=_optional_non_negative_float(selection.get("min_gain")) or 0.0,
        ),
        stacker_method=str(stacker.get("method") or "preflight_only"),
        stacker_params=dict(stacker.get("params") or {}),
        config_path=path,
        raw=payload,
    )


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"Stacking config is missing required field {key!r}.")
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None or str(value) == "":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("Stacking integer options must be greater than 0 when set.")
    return parsed


def _optional_probability(value: object) -> float | None:
    if value is None or str(value) == "":
        return None
    parsed = float(value)
    if parsed < 0 or parsed > 1:
        raise ValueError("Stacking probability options must be within [0, 1] when set.")
    return parsed


def _optional_non_negative_float(value: object) -> float | None:
    if value is None or str(value) == "":
        return None
    parsed = float(value)
    if parsed < 0:
        raise ValueError("Stacking float options must be greater than or equal to 0 when set.")
    return parsed
