from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ktabforge.features.transforms.base import FeatureSchemaItem


@dataclass(frozen=True)
class ArithmeticInteractionsTransform:
    pairs: list[dict[str, Any]]

    def fit_transform(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[FeatureSchemaItem]]:
        train_features = pd.DataFrame(index=train.index)
        test_features = pd.DataFrame(index=test.index)
        feature_schema: list[FeatureSchemaItem] = []

        for pair in self.pairs:
            left = str(pair["left"])
            right = str(pair["right"])
            operations = [str(item) for item in pair["operations"]]
            for operation in operations:
                feature_name = f"{left}__{operation}__{right}"
                train_features[feature_name] = _apply_operation(
                    train[left],
                    train[right],
                    operation,
                )
                test_features[feature_name] = _apply_operation(
                    test[left],
                    test[right],
                    operation,
                )
                feature_schema.append(
                    {
                        "feature_name": feature_name,
                        "transform_type": "arithmetic_interactions",
                        "source_columns": [left, right],
                        "operation": operation,
                    }
                )

        return train_features, test_features, feature_schema


def _apply_operation(left: pd.Series, right: pd.Series, operation: str) -> pd.Series:
    left_values = pd.to_numeric(left, errors="coerce")
    right_values = pd.to_numeric(right, errors="coerce")
    if operation == "add":
        return left_values + right_values
    if operation == "sub":
        return left_values - right_values
    if operation == "mul":
        return left_values * right_values
    if operation == "div":
        denominator = right_values.replace(0, np.nan)
        return left_values / denominator
    raise ValueError(f"Unsupported arithmetic interaction operation {operation!r}")
