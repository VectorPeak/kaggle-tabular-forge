"""Small schema objects shared by data and pipeline workers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TabularSchema:
    """Names of the core columns in a tabular competition."""

    target: str
    id_column: str
