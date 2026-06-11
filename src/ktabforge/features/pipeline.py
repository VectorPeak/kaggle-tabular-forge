from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ktabforge.artifacts.writers import write_json, write_markdown, write_parquet
from ktabforge.config.loader import load_yaml_file
from ktabforge.config.safety import safe_path_segment
from ktabforge.config.schema import bundled_schema_path, require_loaded_config_valid
from ktabforge.data.io import load_tabular_frames
from ktabforge.features.registry import build_transform
from ktabforge.utils.hashing import stable_hash
from ktabforge.utils.time import utc_now_iso


@dataclass(frozen=True)
class FrequencyTransformSpec:
    type: str
    columns: list[str]
    mode: str
    fit_on: str


@dataclass(frozen=True)
class BinningTransformSpec:
    type: str
    columns: list[str]
    bins: int
    strategy: str


@dataclass(frozen=True)
class ArithmeticInteractionsTransformSpec:
    type: str
    pairs: list[dict[str, Any]]


TransformSpec = (
    FrequencyTransformSpec | BinningTransformSpec | ArithmeticInteractionsTransformSpec
)


@dataclass(frozen=True)
class FeatureBuildConfig:
    feature_build_id: str
    competition: str
    artifact_root: Path
    data_dir: Path
    target: str
    id_column: str
    transforms: list[TransformSpec]
    config_path: Path
    raw: dict[str, Any]


@dataclass(frozen=True)
class FeatureBuildResult:
    status: str
    feature_build_id: str
    competition: str
    artifact_dir: Path
    train_features_path: Path
    test_features_path: Path
    manifest_path: Path
    schema_path: Path
    report_path: Path


def load_feature_build_config(config_path: str | Path) -> FeatureBuildConfig:
    path = Path(config_path)
    payload = load_yaml_file(path)
    if not isinstance(payload, dict):
        raise TypeError("Feature build config must be a YAML mapping.")
    require_loaded_config_valid(payload, bundled_schema_path("feature_pipeline.schema.json"))

    feature_build = _mapping(payload, "feature_build")
    data = _mapping(payload, "data")
    raw_transforms = payload.get("transforms", [])
    if not isinstance(raw_transforms, list):
        raise TypeError("Feature build transforms must be a list.")

    transforms = [_load_transform(index, item) for index, item in enumerate(raw_transforms)]
    return FeatureBuildConfig(
        feature_build_id=safe_path_segment(
            _required_string(feature_build, "feature_build_id"),
            field="feature_build_id",
        ),
        competition=safe_path_segment(
            _required_string(feature_build, "competition"),
            field="competition",
        ),
        artifact_root=Path(_required_string(feature_build, "artifact_root")),
        data_dir=Path(_required_string(data, "data_dir")),
        target=_required_string(data, "target"),
        id_column=_required_string(data, "id_column"),
        transforms=transforms,
        config_path=path,
        raw=payload,
    )


def run_feature_build_from_config(config_path: str | Path) -> FeatureBuildResult:
    config = load_feature_build_config(config_path)
    artifact_dir = config.artifact_root / "features" / config.competition / config.feature_build_id
    if artifact_dir.exists():
        raise FileExistsError(
            f"Feature artifact directory already exists; refusing to overwrite: {artifact_dir}"
        )
    artifact_dir.mkdir(parents=True, exist_ok=False)

    frames = load_tabular_frames(
        data_dir=config.data_dir,
        target=config.target,
        id_column=config.id_column,
    )
    train_features = frames.train.copy()
    test_features = frames.test.copy()
    feature_schema: list[dict[str, object]] = []
    transform_summaries: list[dict[str, object]] = []

    for transform_spec in config.transforms:
        transform = build_transform(transform_spec)
        generated_train, generated_test, generated_schema = transform.fit_transform(
            train_features,
            test_features,
        )
        train_features = pd.concat([train_features, generated_train], axis=1)
        test_features = pd.concat([test_features, generated_test], axis=1)
        feature_schema.extend(generated_schema)
        transform_summaries.append(
            {
                "type": transform_spec.type,
                "generated_feature_count": len(generated_schema),
            }
        )

    train_features = _reorder_train_columns(train_features, config.id_column, config.target)
    test_features = _reorder_test_columns(test_features, config.id_column)

    train_features_path = artifact_dir / "train_features.parquet"
    test_features_path = artifact_dir / "test_features.parquet"
    manifest_path = artifact_dir / "feature_build_manifest.json"
    schema_path = artifact_dir / "feature_schema.json"
    report_path = artifact_dir / "feature_build_report.md"

    write_parquet(train_features_path, train_features)
    write_parquet(test_features_path, test_features)
    write_json(
        schema_path,
        {
            "feature_build_id": config.feature_build_id,
            "competition": config.competition,
            "features": feature_schema,
        },
    )
    write_json(
        manifest_path,
        {
            "status": "completed",
            "feature_build_id": config.feature_build_id,
            "competition": config.competition,
            "artifact_dir": str(artifact_dir),
            "config_path": str(config.config_path),
            "config_hash": stable_hash(config.raw),
            "created_at": utc_now_iso(),
            "train_row_count": len(train_features),
            "test_row_count": len(test_features),
            "generated_feature_count": len(feature_schema),
            "transforms": transform_summaries,
            "paths": {
                "train_features_path": str(train_features_path),
                "test_features_path": str(test_features_path),
                "feature_schema_path": str(schema_path),
                "report_path": str(report_path),
            },
        },
    )
    write_markdown(
        report_path,
        _render_feature_build_report(
            feature_build_id=config.feature_build_id,
            competition=config.competition,
            transform_summaries=transform_summaries,
            feature_schema=feature_schema,
        ),
    )

    return FeatureBuildResult(
        status="completed",
        feature_build_id=config.feature_build_id,
        competition=config.competition,
        artifact_dir=artifact_dir,
        train_features_path=train_features_path,
        test_features_path=test_features_path,
        manifest_path=manifest_path,
        schema_path=schema_path,
        report_path=report_path,
    )


def _load_transform(index: int, raw: object) -> TransformSpec:
    if not isinstance(raw, dict):
        raise TypeError(f"transforms.{index} must be a mapping.")
    transform_type = _required_string(raw, "type")
    if transform_type == "frequency":
        return FrequencyTransformSpec(
            type=transform_type,
            columns=[str(item) for item in raw.get("columns", [])],
            mode=_required_string(raw, "mode"),
            fit_on=str(raw.get("fit_on", "train_only")),
        )
    if transform_type == "binning":
        bins = int(raw.get("bins", 0))
        if bins <= 0:
            raise ValueError(f"transforms.{index}.bins must be greater than 0.")
        return BinningTransformSpec(
            type=transform_type,
            columns=[str(item) for item in raw.get("columns", [])],
            bins=bins,
            strategy=_required_string(raw, "strategy"),
        )
    if transform_type == "arithmetic_interactions":
        pairs = raw.get("pairs", [])
        if not isinstance(pairs, list) or not pairs:
            raise ValueError(f"transforms.{index}.pairs must be a non-empty list.")
        return ArithmeticInteractionsTransformSpec(type=transform_type, pairs=list(pairs))
    raise ValueError(f"Unsupported transform type {transform_type!r}.")


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"Feature build config section {key!r} must be a mapping.")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"Feature build config is missing required field {key!r}.")
    return str(value)


def _reorder_train_columns(frame: pd.DataFrame, id_column: str, target: str) -> pd.DataFrame:
    base = [id_column, target]
    rest = [column for column in frame.columns if column not in base]
    return frame[base + rest]


def _reorder_test_columns(frame: pd.DataFrame, id_column: str) -> pd.DataFrame:
    rest = [column for column in frame.columns if column != id_column]
    return frame[[id_column] + rest]


def _render_feature_build_report(
    *,
    feature_build_id: str,
    competition: str,
    transform_summaries: list[dict[str, object]],
    feature_schema: list[dict[str, object]],
) -> str:
    transform_lines = "\n".join(
        f"- `{item['type']}`: {item['generated_feature_count']} features"
        for item in transform_summaries
    )
    feature_lines = "\n".join(f"- `{item['feature_name']}`" for item in feature_schema[:20])
    return (
        "# Feature Build Report\n\n"
        f"- `feature_build_id`: `{feature_build_id}`\n"
        f"- `competition`: `{competition}`\n"
        f"- `generated_feature_count`: {len(feature_schema)}\n\n"
        "## Transforms\n\n"
        f"{transform_lines or '- none'}\n\n"
        "## Generated Features\n\n"
        f"{feature_lines or '- none'}\n"
    )
