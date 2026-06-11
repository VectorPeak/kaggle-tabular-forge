from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ktabforge.features.transforms.base import FeatureSchemaItem


@dataclass(frozen=True)
class BinningTransform:
    columns: list[str]
    bins: int
    strategy: str

    def fit_transform(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[FeatureSchemaItem]]:
        train_features = pd.DataFrame(index=train.index)
        test_features = pd.DataFrame(index=test.index)
        feature_schema: list[FeatureSchemaItem] = []

        for column in self.columns:
            edges = _fit_edges(train[column], bins=self.bins, strategy=self.strategy)
            feature_name = f"{column}__bin"
            train_features[feature_name] = _apply_bins(train[column], edges)
            test_features[feature_name] = _apply_bins(test[column], edges)
            feature_schema.append(
                {
                    "feature_name": feature_name,
                    "transform_type": "binning",
                    "source_columns": [column],
                    "bins": len(edges) - 1,
                    "strategy": self.strategy,
                    "edges": [float(edge) for edge in edges],
                }
            )

        return train_features, test_features, feature_schema


def _fit_edges(series: pd.Series, *, bins: int, strategy: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return np.array([-np.inf, np.inf], dtype=float)

    if strategy != "quantile":
        raise ValueError(f"Unsupported binning strategy {strategy!r}")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(valid, quantiles)
    edges = np.unique(edges.astype(float))
    if len(edges) < 2:
        return np.array([-np.inf, np.inf], dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _apply_bins(series: pd.Series, edges: np.ndarray) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    binned = pd.cut(
        numeric,
        bins=edges,
        labels=False,
        include_lowest=True,
    )
    return binned.astype("Int64").fillna(-1).astype(int)
