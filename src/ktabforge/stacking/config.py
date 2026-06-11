from __future__ import annotations

import math
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


_KNOWN_STACKER_METHODS = frozenset({"preflight_only", "logistic_regression", "ridge_classifier"})
_STACK_RUNTIME_STACKER_METHODS = frozenset({"logistic_regression", "ridge_classifier"})


def load_stacking_config(
    config_path: str | Path,
    *,
    runtime: str | None = None,
) -> StackingPreflightConfig:
    resolved_runtime = runtime or "preflight"
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
    stacker_params = stacker.get("params") or {}
    if not isinstance(stacker_params, dict):
        raise TypeError("Stacking config section 'stacker.params' must be a mapping when provided.")

    top_n = _optional_int(stacking.get("top_n"))
    max_parents = _optional_int(stacking.get("max_parents"))
    min_parents = _optional_int(stacking.get("min_parents"))
    resolved_min_parents = min_parents if min_parents is not None else 2
    _validate_parent_relationships(
        top_n=top_n,
        max_parents=max_parents,
        min_parents=resolved_min_parents,
    )

    stacker_method = _stacker_method(stacker.get("method"), runtime=resolved_runtime)
    _require_finite_numbers(stacker_params, path="stacker.params")
    selection_strategy = str(selection.get("strategy") or "score_desc")
    _validate_selection_relationships(
        strategy=selection_strategy,
        max_pairwise_corr=selection.get("max_pairwise_corr"),
    )

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
        top_n=top_n,
        max_parents=max_parents,
        min_parents=resolved_min_parents,
        selection=StackingSelectionConfig(
            strategy=selection_strategy,
            max_pairwise_corr=_optional_probability(
                selection.get("max_pairwise_corr"),
                field="selection.max_pairwise_corr",
            ),
            report_top_k_pairs=_optional_int(selection.get("report_top_k_pairs")) or 20,
            min_gain=_optional_non_negative_float(
                selection.get("min_gain"),
                field="selection.min_gain",
            )
            or 0.0,
        ),
        stacker_method=stacker_method,
        stacker_params=dict(stacker_params),
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


def _optional_probability(value: object, *, field: str) -> float | None:
    if value is None or str(value) == "":
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or parsed > 1:
        raise ValueError(f"Stacking config field {field} must be finite and within [0, 1].")
    return parsed


def _optional_non_negative_float(value: object, *, field: str) -> float | None:
    if value is None or str(value) == "":
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(
            f"Stacking config field {field} must be a finite float greater than or equal to 0."
        )
    return parsed


def _validate_parent_relationships(
    *,
    top_n: int | None,
    max_parents: int | None,
    min_parents: int,
) -> None:
    if max_parents is not None and max_parents < min_parents:
        raise ValueError(
            "Stacking config field max_parents must be greater than or equal to min_parents."
        )
    if top_n is not None and top_n < min_parents:
        raise ValueError(
            "Stacking config field top_n must be greater than or equal to min_parents."
        )


def _validate_selection_relationships(*, strategy: str, max_pairwise_corr: object) -> None:
    strategy_key = strategy.strip().lower()
    if strategy_key == "diversity_greedy" and max_pairwise_corr in (None, ""):
        raise ValueError(
            "selection.max_pairwise_corr is required when selection.strategy=diversity_greedy."
        )
    if strategy_key == "hill_climb_greedy" and max_pairwise_corr not in (None, ""):
        raise ValueError(
            "selection.max_pairwise_corr is not supported when "
            "selection.strategy=hill_climb_greedy."
        )


def _stacker_method(value: object, *, runtime: str) -> str:
    method = str(value or "preflight_only").strip().lower()
    if method not in _KNOWN_STACKER_METHODS:
        known_methods = ", ".join(sorted(_KNOWN_STACKER_METHODS))
        raise ValueError(
            f"Unsupported stacker.method {method!r}. Known methods: {known_methods}."
        )
    if runtime == "stack" and method not in _STACK_RUNTIME_STACKER_METHODS:
        allowed_methods = ", ".join(sorted(_STACK_RUNTIME_STACKER_METHODS))
        raise ValueError(
            "stacker.method=preflight_only cannot produce a completed stack run. "
            f"Use one of: {allowed_methods}."
        )
    return method


def _require_finite_numbers(value: object, *, path: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Stacking config field {path} must be finite.")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_numbers(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_numbers(item, path=f"{path}[{index}]")
