"""Shared CV protocol types."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class FoldBuilder(Protocol):
    """Callable interface for functions that annotate a training frame with folds."""

    def __call__(
        self,
        train: pd.DataFrame,
        target: str,
        n_splits: int,
        seed: int,
    ) -> pd.DataFrame: ...
