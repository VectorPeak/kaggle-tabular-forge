from __future__ import annotations

from pathlib import Path

import pandas as pd


def compare_experiments(
    artifact_root: str | Path,
    competition: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return completed experiments with available OOF evidence, ranked by score."""
    root = Path(artifact_root)
    registry_path = root / "registry" / competition / "experiment_registry.csv"
    if not registry_path.exists():
        return pd.DataFrame()

    registry = pd.read_csv(registry_path)
    if registry.empty:
        return registry

    frame = registry.copy()
    if "status" in frame.columns:
        frame = frame[frame["status"].astype(str).str.lower() == "completed"]
    else:
        frame = frame.iloc[0:0]

    if "oof_path" in frame.columns:
        frame = frame[frame["oof_path"].map(lambda value: _path_exists(value, root))]
    else:
        frame = frame.iloc[0:0]

    if "oof_score" in frame.columns:
        frame = frame.assign(oof_score=pd.to_numeric(frame["oof_score"], errors="coerce"))
        frame = frame.sort_values("oof_score", ascending=False, na_position="last")

    return frame.head(top_n).reset_index(drop=True)


def _path_exists(value: object, artifact_root: Path) -> bool:
    if pd.isna(value):
        return False
    path = Path(str(value))
    candidates = [path] if path.is_absolute() else [path, artifact_root / path]
    return any(candidate.exists() for candidate in candidates)
