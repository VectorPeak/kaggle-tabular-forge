from dataclasses import dataclass
from typing import Protocol

import pandas as pd

FeatureSchemaItem = dict[str, object]


class FeatureTransform(Protocol):
    def fit_transform(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[FeatureSchemaItem]]:
        ...


@dataclass(frozen=True)
class TransformResult:
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    feature_schema: list[FeatureSchemaItem]
