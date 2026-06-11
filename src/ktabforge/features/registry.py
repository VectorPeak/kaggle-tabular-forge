from __future__ import annotations

from typing import Any

from ktabforge.features.transforms import (
    ArithmeticInteractionsTransform,
    BinningTransform,
    FrequencyTransform,
)
from ktabforge.features.transforms.base import FeatureTransform


def build_transform(transform_spec: Any) -> FeatureTransform:
    if transform_spec.type == "frequency":
        return FrequencyTransform(
            columns=transform_spec.columns,
            mode=transform_spec.mode,
            fit_on=transform_spec.fit_on,
        )
    if transform_spec.type == "binning":
        return BinningTransform(
            columns=transform_spec.columns,
            bins=transform_spec.bins,
            strategy=transform_spec.strategy,
        )
    if transform_spec.type == "arithmetic_interactions":
        return ArithmeticInteractionsTransform(pairs=transform_spec.pairs)
    raise ValueError(f"Unsupported transform type {transform_spec.type!r}")
