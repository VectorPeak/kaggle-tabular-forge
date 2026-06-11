from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ktabforge.config.loader import load_yaml_file
from ktabforge.config.safety import safe_path_segment
from ktabforge.config.schema import bundled_schema_path, require_loaded_config_valid
from ktabforge.utils.hashing import stable_hash


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    competition: str
    data_dir: Path
    artifact_root: Path
    target: str
    id_column: str
    n_splits: int
    seed: int
    run_mode: str
    feature_set: str | None
    feature_families: list[str]
    model_family: str
    model_preset: str | None
    model_params: dict[str, Any]
    config_path: Path
    config_hash: str
    raw: dict[str, Any]


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    path = Path(config_path)
    payload = load_yaml_file(path)
    if not isinstance(payload, dict):
        raise TypeError("Experiment config must be a YAML mapping.")
    require_loaded_config_valid(payload, bundled_schema_path("experiment.schema.json"))

    experiment = _mapping(payload, "experiment")
    data = _mapping(payload, "data")
    validation = _mapping(payload, "validation")
    features = _mapping(payload, "features")
    model = _mapping(payload, "model")

    seed = int(experiment.get("seed", validation.get("seed", 42)))
    return ExperimentConfig(
        experiment_id=safe_path_segment(
            _required_string(experiment, "experiment_id"),
            field="experiment_id",
        ),
        competition=safe_path_segment(
            _required_string(experiment, "competition"),
            field="competition",
        ),
        data_dir=Path(_required_string(data, "data_dir")),
        artifact_root=Path(_required_string(data, "artifact_root")),
        target=_required_string(data, "target"),
        id_column=_required_string(data, "id_column"),
        n_splits=int(validation.get("n_splits", 5)),
        seed=seed,
        run_mode=str(experiment.get("run_mode", "smoke")),
        feature_set=_optional_string(features.get("feature_set")),
        feature_families=[str(item) for item in features.get("families", [])],
        model_family=_required_string(model, "family"),
        model_preset=_optional_string(model.get("preset")),
        model_params=dict(model.get("params") or {}),
        config_path=path,
        config_hash=stable_hash(payload),
        raw=payload,
    )


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"Experiment config section {key!r} must be a mapping.")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"Experiment config is missing required field {key!r}.")
    return str(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
