from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ktabforge.config.loader import load_yaml_file
from ktabforge.config.safety import safe_path_segment
from ktabforge.config.schema import bundled_schema_path, require_loaded_config_valid
from ktabforge.utils.hashing import stable_hash

_SAFE_AXIS_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$")
_FORBIDDEN_AXIS_KEYS = {
    "data.artifact_root",
    "experiment.competition",
    "experiment.experiment_id",
    "outputs",
}
_FORBIDDEN_AXIS_PREFIXES = ("outputs.",)


@dataclass(frozen=True)
class MatrixConfig:
    factory_id: str
    competition: str
    artifact_root: Path
    continue_on_error: bool
    max_runs: int | None
    base_experiment: dict[str, Any]
    experiment_id_template: str
    axes: dict[str, list[Any]]
    report_metric_name: str
    report_top_n: int
    config_path: Path
    config_hash: str
    raw: dict[str, Any]


def load_matrix_config(config_path: str | Path) -> MatrixConfig:
    path = Path(config_path)
    payload = load_yaml_file(path)
    if not isinstance(payload, dict):
        raise TypeError("Matrix config must be a YAML mapping.")
    require_loaded_config_valid(payload, bundled_schema_path("matrix.schema.json"))

    factory = _mapping(payload, "factory")
    matrix = _mapping(payload, "matrix")
    report = _mapping(payload, "report")
    base_experiment = _mapping(payload, "base_experiment")
    axes = _mapping(matrix, "axes")

    max_runs = _optional_int(factory.get("max_runs"))
    if max_runs is not None and max_runs <= 0:
        raise ValueError("factory.max_runs must be greater than 0 when set")
    report_top_n = _positive_int(report.get("top_n", 20), field="report.top_n")

    return MatrixConfig(
        factory_id=safe_path_segment(_required_string(factory, "factory_id"), field="factory_id"),
        competition=safe_path_segment(
            _required_string(factory, "competition"),
            field="competition",
        ),
        artifact_root=Path(_required_string(factory, "artifact_root")),
        continue_on_error=_optional_bool(factory.get("continue_on_error"), default=True),
        max_runs=max_runs,
        base_experiment=dict(base_experiment),
        experiment_id_template=_required_string(matrix, "experiment_id_template"),
        axes=_axis_mapping(axes),
        report_metric_name=str(report.get("metric_name") or "roc_auc"),
        report_top_n=report_top_n,
        config_path=path,
        config_hash=stable_hash(payload),
        raw=payload,
    )


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"Matrix config section {key!r} must be a mapping.")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"Matrix config is missing required field {key!r}.")
    return str(value)


def _list(value: Any, *, key: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Matrix axis {key!r} must be a non-empty list.")
    return value


def _optional_int(value: object) -> int | None:
    if value is None or str(value) == "":
        return None
    return int(value)


def _positive_int(value: object, *, field: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field} must be greater than 0")
    return parsed


def _optional_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise TypeError("factory.continue_on_error must be a boolean")


def _axis_mapping(axes: dict[str, Any]) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for key, value in axes.items():
        axis_path = str(key)
        _validate_axis_path(axis_path)
        result[axis_path] = _list(value, key=axis_path)
    return result


def _validate_axis_path(axis_path: str) -> None:
    if not _SAFE_AXIS_PATH.match(axis_path):
        raise ValueError(f"Unsafe matrix axis path: {axis_path}")
    if axis_path in _FORBIDDEN_AXIS_KEYS:
        raise ValueError(f"Unsafe matrix axis path: {axis_path}")
    if any(axis_path.startswith(prefix) for prefix in _FORBIDDEN_AXIS_PREFIXES):
        raise ValueError(f"Unsafe matrix axis path: {axis_path}")
