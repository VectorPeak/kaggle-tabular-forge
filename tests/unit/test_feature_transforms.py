from __future__ import annotations

import math

import pandas as pd

from ktabforge.features.transforms.binning import BinningTransform
from ktabforge.features.transforms.frequency import FrequencyTransform
from ktabforge.features.transforms.interactions import ArithmeticInteractionsTransform


def test_frequency_transform_supports_count_and_frequency_outputs():
    train = pd.DataFrame({"Contract": ["A", "A", "B", None]})
    test = pd.DataFrame({"Contract": ["A", "C", None]})

    count_transform = FrequencyTransform(columns=["Contract"], mode="count")
    train_count, test_count, count_schema = count_transform.fit_transform(train, test)

    assert train_count["Contract__count"].tolist() == [2.0, 2.0, 1.0, 1.0]
    assert test_count["Contract__count"].tolist() == [2.0, 0.0, 1.0]
    assert count_schema[0]["feature_name"] == "Contract__count"

    freq_transform = FrequencyTransform(columns=["Contract"], mode="frequency")
    train_freq, test_freq, _ = freq_transform.fit_transform(train, test)

    assert train_freq["Contract__frequency"].round(6).tolist() == [0.5, 0.5, 0.25, 0.25]
    assert test_freq["Contract__frequency"].round(6).tolist() == [0.5, 0.0, 0.25]


def test_binning_transform_uses_train_fitted_edges_for_test_rows():
    train = pd.DataFrame({"tenure": [1, 2, 3, 4]})
    test = pd.DataFrame({"tenure": [0, 5]})

    transform = BinningTransform(columns=["tenure"], bins=2, strategy="quantile")
    train_features, test_features, schema = transform.fit_transform(train, test)

    assert train_features["tenure__bin"].tolist() == [0, 0, 1, 1]
    assert test_features["tenure__bin"].tolist() == [0, 1]
    assert schema[0]["feature_name"] == "tenure__bin"
    assert schema[0]["bins"] == 2


def test_arithmetic_interactions_transform_builds_requested_operations():
    train = pd.DataFrame(
        {
            "MonthlyCharges": [10.0, 20.0],
            "tenure": [2.0, 0.0],
        }
    )
    test = pd.DataFrame(
        {
            "MonthlyCharges": [5.0],
            "tenure": [4.0],
        }
    )

    transform = ArithmeticInteractionsTransform(
        pairs=[
            {
                "left": "MonthlyCharges",
                "right": "tenure",
                "operations": ["add", "sub", "mul", "div"],
            }
        ]
    )
    train_features, test_features, schema = transform.fit_transform(train, test)

    assert train_features["MonthlyCharges__add__tenure"].tolist() == [12.0, 20.0]
    assert train_features["MonthlyCharges__sub__tenure"].tolist() == [8.0, 20.0]
    assert train_features["MonthlyCharges__mul__tenure"].tolist() == [20.0, 0.0]
    assert train_features["MonthlyCharges__div__tenure"].iloc[0] == 5.0
    assert math.isnan(train_features["MonthlyCharges__div__tenure"].iloc[1])
    assert test_features["MonthlyCharges__mul__tenure"].tolist() == [20.0]
    assert [item["feature_name"] for item in schema] == [
        "MonthlyCharges__add__tenure",
        "MonthlyCharges__sub__tenure",
        "MonthlyCharges__mul__tenure",
        "MonthlyCharges__div__tenure",
    ]
