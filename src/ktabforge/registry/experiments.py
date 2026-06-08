from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

REGISTRY_COLUMNS = [
    "experiment_id",
    "competition",
    "metric_name",
    "oof_score",
    "status",
    "oof_path",
    "test_pred_path",
    "fold_metrics_path",
    "model_family",
    "seed",
    "run_mode",
    "reason",
    "model_preset",
    "feature_set",
    "config_hash",
    "created_at",
    "feature_manifest_hash",
]


def append_experiment_registry(path: str | Path, row: dict[str, Any]) -> Path:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    if registry_path.exists():
        old_frame = pd.read_csv(registry_path)
    else:
        old_frame = pd.DataFrame()

    extra_columns = [
        column
        for column in list(row.keys()) + list(old_frame.columns)
        if column not in REGISTRY_COLUMNS
    ]
    columns = REGISTRY_COLUMNS + list(dict.fromkeys(extra_columns))
    normalized = {column: row.get(column) for column in columns}
    new_frame = pd.DataFrame([normalized], columns=columns)
    frame = pd.concat([new_frame, old_frame.reindex(columns=columns)], ignore_index=True)

    frame.to_csv(registry_path, index=False)
    return registry_path
