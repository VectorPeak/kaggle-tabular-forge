from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FactoryRunResult:
    status: str
    completed_count: int
    failed_count: int
    planned_count: int
    factory_dir: Path
    plan_path: Path
    summary_path: Path
    summary_json_path: Path
    report_path: Path

