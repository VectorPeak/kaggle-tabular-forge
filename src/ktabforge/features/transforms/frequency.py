from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ktabforge.features.transforms.base import FeatureSchemaItem

_MISSING_TOKEN = "__nan__"


@dataclass(frozen=True)
class FrequencyTransform:
    columns: list[str]
    mode: str
    fit_on: str = "train_only"

    def fit_transform(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[FeatureSchemaItem]]:
        stats_source = train if self.fit_on == "train_only" else pd.concat([train, test], axis=0)
        train_features = pd.DataFrame(index=train.index)
        test_features = pd.DataFrame(index=test.index)
        feature_schema: list[FeatureSchemaItem] = []

        for column in self.columns:
            values = _normalize(stats_source[column])
            counts = values.value_counts(dropna=False)
            if self.mode == "count":
                series_name = f"{column}__count"
                train_features[series_name] = (
                    _normalize(train[column]).map(counts).fillna(0.0).astype(float)
                )
                test_features[series_name] = (
                    _normalize(test[column]).map(counts).fillna(0.0).astype(float)
                )
            else:
                series_name = f"{column}__frequency"
                frequencies = counts / float(len(stats_source))
                train_features[series_name] = (
                    _normalize(train[column]).map(frequencies).fillna(0.0).astype(float)
                )
                test_features[series_name] = (
                    _normalize(test[column]).map(frequencies).fillna(0.0).astype(float)
                )

            feature_schema.append(
                {
                    "feature_name": series_name,
                    "transform_type": "frequency",
                    "source_columns": [column],
                    "mode": self.mode,
                    "fit_on": self.fit_on,
                    "transductive_risk": "medium" if self.fit_on == "train_test" else "low",
                }
            )

        return train_features, test_features, feature_schema


def _normalize(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), _MISSING_TOKEN)
