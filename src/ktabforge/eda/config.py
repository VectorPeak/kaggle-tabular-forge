from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ktabforge.config.loader import load_yaml_file
from ktabforge.config.safety import safe_path_segment
from ktabforge.config.schema import bundled_schema_path, require_loaded_config_valid


@dataclass(frozen=True)
class EdaConfig:
    eda_id: str
    competition: str
    artifact_root: Path
    data_dir: Path
    target: str
    id_column: str
    focus: list[str]
    profile_train: bool
    profile_test: bool
    include_drift: bool
    include_leakage_watchlist: bool
    config_path: Path
    raw: dict[str, object]


def load_eda_config(config_path: str | Path) -> EdaConfig:
    path = Path(config_path)
    payload = load_yaml_file(path)
    if not isinstance(payload, dict):
        raise TypeError("EDA config must be a YAML mapping.")
    require_loaded_config_valid(payload, bundled_schema_path("eda.schema.json"))

    eda = _mapping(payload, "eda")
    data = _mapping(payload, "data")
    scan = _mapping(payload, "scan")

    return EdaConfig(
        eda_id=safe_path_segment(_required_string(eda, "eda_id"), field="eda_id"),
        competition=safe_path_segment(
            _required_string(eda, "competition"),
            field="competition",
        ),
        artifact_root=Path(_required_string(eda, "artifact_root")),
        data_dir=Path(_required_string(data, "data_dir")),
        target=_required_string(data, "target"),
        id_column=_required_string(data, "id_column"),
        focus=[str(item) for item in eda.get("focus", [])],
        profile_train=_optional_bool(scan.get("profile_train"), default=True),
        profile_test=_optional_bool(scan.get("profile_test"), default=True),
        include_drift=_optional_bool(scan.get("include_drift"), default=True),
        include_leakage_watchlist=_optional_bool(
            scan.get("include_leakage_watchlist"),
            default=True,
        ),
        config_path=path,
        raw=payload,
    )


def _mapping(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"EDA config section {key!r} must be a mapping.")
    return value


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"EDA config is missing required field {key!r}.")
    return str(value)


def _optional_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise TypeError("EDA scan options must be booleans when provided.")
