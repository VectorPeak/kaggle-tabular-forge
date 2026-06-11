from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ktabforge.artifacts.writers import write_json, write_markdown
from ktabforge.config.loader import load_yaml_file
from ktabforge.config.safety import safe_path_segment
from ktabforge.config.schema import bundled_schema_path, require_loaded_config_valid
from ktabforge.data.io import load_tabular_frames
from ktabforge.eda.profiling import build_feature_backlog, profile_tabular_frames
from ktabforge.eda.report import render_eda_summary
from ktabforge.eda.watchlist import build_leakage_watchlist
from ktabforge.utils.hashing import stable_hash
from ktabforge.utils.time import utc_now_iso


@dataclass(frozen=True)
class EdaScanConfig:
    eda_id: str
    competition: str
    data_dir: Path
    artifact_root: Path
    target: str
    id_column: str
    focus: list[str]
    config_path: Path
    raw: dict[str, Any]


@dataclass(frozen=True)
class EdaScanResult:
    status: str
    eda_id: str
    competition: str
    artifact_dir: Path
    manifest_path: Path
    summary_path: Path
    watchlist_path: Path
    backlog_path: Path


def run_eda_scan_from_config(config_path: str | Path) -> EdaScanResult:
    config = _load_eda_scan_config(config_path)
    artifact_dir = config.artifact_root / "eda_findings" / config.competition / config.eda_id
    if artifact_dir.exists():
        raise FileExistsError(
            f"EDA artifact directory already exists; refusing to overwrite: {artifact_dir}"
        )
    artifact_dir.mkdir(parents=True, exist_ok=False)

    frames = load_tabular_frames(
        data_dir=config.data_dir,
        target=config.target,
        id_column=config.id_column,
    )
    profile = profile_tabular_frames(
        train=frames.train,
        test=frames.test,
        target=config.target,
        id_column=config.id_column,
    )
    watchlist = build_leakage_watchlist(
        train=frames.train,
        test=frames.test,
        target=config.target,
        id_column=config.id_column,
    )
    feature_backlog = build_feature_backlog(profile)

    manifest_path = artifact_dir / "eda_manifest.json"
    summary_path = artifact_dir / "eda_summary.md"
    watchlist_path = artifact_dir / "leakage_watchlist.json"
    backlog_path = artifact_dir / "feature_backlog.json"

    write_json(
        manifest_path,
        {
            "eda_id": config.eda_id,
            "competition": config.competition,
            "data_dir": str(config.data_dir),
            "artifact_root": str(config.artifact_root),
            "target": config.target,
            "id_column": config.id_column,
            "focus": config.focus,
            "config_path": str(config.config_path),
            "config_hash": stable_hash(config.raw),
            "created_at": utc_now_iso(),
            "row_count": profile["row_count"],
            "watchlist_count": len(watchlist),
            "feature_backlog_count": len(feature_backlog),
            "paths": {
                "summary_path": str(summary_path),
                "watchlist_path": str(watchlist_path),
                "feature_backlog_path": str(backlog_path),
            },
        },
    )
    write_json(
        watchlist_path,
        {
            "eda_id": config.eda_id,
            "competition": config.competition,
            "items": watchlist,
        },
    )
    write_json(
        backlog_path,
        {
            "eda_id": config.eda_id,
            "competition": config.competition,
            "items": feature_backlog,
        },
    )
    write_markdown(
        summary_path,
        render_eda_summary(
            eda_id=config.eda_id,
            competition=config.competition,
            focus=config.focus,
            profile=profile,
            watchlist=watchlist,
            feature_backlog=feature_backlog,
        ),
    )

    return EdaScanResult(
        status="completed",
        eda_id=config.eda_id,
        competition=config.competition,
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        summary_path=summary_path,
        watchlist_path=watchlist_path,
        backlog_path=backlog_path,
    )


def _load_eda_scan_config(config_path: str | Path) -> EdaScanConfig:
    loader = _external_loader()
    if loader is not None:
        loaded = loader(config_path)
        return _coerce_loaded_config(loaded, config_path)

    path = Path(config_path)
    payload = load_yaml_file(path)
    if not isinstance(payload, dict):
        raise TypeError("EDA config must be a YAML mapping.")
    schema_path = bundled_schema_path("eda.schema.json")
    if schema_path.exists():
        require_loaded_config_valid(payload, schema_path)

    eda = payload.get("eda", {})
    if not isinstance(eda, dict):
        raise TypeError("EDA config section 'eda' must be a mapping.")
    return EdaScanConfig(
        eda_id=safe_path_segment(_required_string(eda, "eda_id"), field="eda_id"),
        competition=safe_path_segment(
            _required_string(eda, "competition"),
            field="competition",
        ),
        data_dir=Path(_required_string(eda, "data_dir")),
        artifact_root=Path(_required_string(eda, "artifact_root")),
        target=_required_string(eda, "target"),
        id_column=str(eda.get("id_column") or "id"),
        focus=[str(item) for item in eda.get("focus", [])],
        config_path=path,
        raw=payload,
    )


def _external_loader() -> Any:
    try:
        from ktabforge.eda.config import load_eda_config
    except ImportError:
        return None
    return load_eda_config


def _coerce_loaded_config(config: Any, config_path: str | Path) -> EdaScanConfig:
    raw = dict(config.raw)
    resolved_config_path = (
        config.config_path if hasattr(config, "config_path") else Path(config_path)
    )
    return EdaScanConfig(
        eda_id=safe_path_segment(str(config.eda_id), field="eda_id"),
        competition=safe_path_segment(str(config.competition), field="competition"),
        data_dir=Path(config.data_dir),
        artifact_root=Path(config.artifact_root),
        target=str(config.target),
        id_column=str(config.id_column),
        focus=_coerce_focus(config, raw),
        config_path=Path(resolved_config_path),
        raw=raw,
    )


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"EDA config is missing required field {key!r}.")
    return str(value)


def _coerce_focus(config: Any, raw: dict[str, Any]) -> list[str]:
    if hasattr(config, "focus"):
        values = [str(item) for item in list(config.focus)]
        if values:
            return values

    scan = raw.get("scan", {})
    if not isinstance(scan, dict):
        return []

    focus: list[str] = []
    if bool(scan.get("profile_train", True)) or bool(scan.get("profile_test", True)):
        focus.append("profiling")
    if bool(scan.get("include_drift", True)):
        focus.append("drift")
    if bool(scan.get("include_leakage_watchlist", True)):
        focus.append("leakage")
    return focus
