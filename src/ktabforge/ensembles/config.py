from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ktabforge.config.loader import load_yaml_file
from ktabforge.config.safety import safe_path_segment
from ktabforge.config.schema import bundled_schema_path, require_loaded_config_valid


@dataclass(frozen=True)
class EnsembleConfig:
    experiment_id: str
    competition: str
    artifact_root: Path
    target: str
    id_column: str
    metric_name: str
    method: str
    candidate_ids: list[str]
    weights: dict[str, float]
    top_n: int | None
    config_path: Path
    raw: dict[str, object]


def load_ensemble_config(config_path: str | Path) -> EnsembleConfig:
    path = Path(config_path)
    payload = load_yaml_file(path)
    if not isinstance(payload, dict):
        raise TypeError("Ensemble config must be a YAML mapping.")
    require_loaded_config_valid(payload, bundled_schema_path("ensemble.schema.json"))
    ensemble = payload.get("ensemble", {})
    if not isinstance(ensemble, dict):
        raise TypeError("Ensemble config section 'ensemble' must be a mapping.")

    return EnsembleConfig(
        experiment_id=safe_path_segment(
            _required_string(ensemble, "experiment_id"),
            field="experiment_id",
        ),
        competition=safe_path_segment(
            _required_string(ensemble, "competition"),
            field="competition",
        ),
        artifact_root=Path(_required_string(ensemble, "artifact_root")),
        target=_required_string(ensemble, "target"),
        id_column=str(ensemble.get("id_column") or "id"),
        metric_name=str(ensemble.get("metric_name") or "roc_auc"),
        method=str(ensemble.get("method") or "simple_average"),
        candidate_ids=[str(item) for item in ensemble.get("candidate_ids", [])],
        weights={
            str(key): float(value)
            for key, value in dict(ensemble.get("weights") or {}).items()
        },
        top_n=_optional_int(ensemble.get("top_n")),
        config_path=path,
        raw=payload,
    )


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"Ensemble config is missing required field {key!r}.")
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None or str(value) == "":
        return None
    return int(value)
