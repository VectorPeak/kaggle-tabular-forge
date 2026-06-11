from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ktabforge.candidates import CandidateCompatibilityRejection, CandidateRecord


@dataclass(frozen=True)
class PairwiseCorrelation:
    left_experiment_id: str
    right_experiment_id: str
    correlation: float
    abs_correlation: float


@dataclass(frozen=True)
class SelectionPolicyResult:
    accepted: list[CandidateRecord]
    rejected: list[CandidateCompatibilityRejection]
    pairwise_correlations: list[PairwiseCorrelation]


def apply_selection_policy(
    candidates: list[CandidateRecord],
    *,
    rejected: list[CandidateCompatibilityRejection],
    strategy: str,
    max_pairwise_corr: float | None,
) -> SelectionPolicyResult:
    pairwise_correlations = compute_pairwise_correlations(candidates)
    kept_rejections = list(rejected)
    strategy_key = strategy.strip().lower()
    ranked_candidates = _sort_candidates_by_score(candidates)

    if strategy_key == "score_desc":
        return SelectionPolicyResult(
            accepted=ranked_candidates,
            rejected=kept_rejections,
            pairwise_correlations=pairwise_correlations,
        )

    if strategy_key != "diversity_greedy":
        raise ValueError(
            f"Unsupported selection.strategy {strategy!r}. "
            "Known strategies: score_desc, diversity_greedy."
        )

    if max_pairwise_corr is None:
        raise ValueError(
            "selection.max_pairwise_corr is required when "
            "selection.strategy=diversity_greedy"
        )
    if max_pairwise_corr < 0 or max_pairwise_corr > 1:
        raise ValueError("selection.max_pairwise_corr must be within [0, 1]")

    correlation_lookup = {
        frozenset((pair.left_experiment_id, pair.right_experiment_id)): pair
        for pair in pairwise_correlations
    }
    accepted: list[CandidateRecord] = []
    for candidate in ranked_candidates:
        blocker = _first_blocking_parent(
            candidate,
            accepted=accepted,
            correlation_lookup=correlation_lookup,
            max_pairwise_corr=max_pairwise_corr,
        )
        if blocker is None:
            accepted.append(candidate)
            continue
        parent_id, pair = blocker
        kept_rejections.append(
            CandidateCompatibilityRejection(
                experiment_id=candidate.experiment_id,
                reason=(
                    "pairwise correlation with "
                    f"{parent_id}={pair.abs_correlation:.6f} exceeds "
                    f"max_pairwise_corr={max_pairwise_corr}"
                ),
            )
        )

    return SelectionPolicyResult(
        accepted=accepted,
        rejected=kept_rejections,
        pairwise_correlations=pairwise_correlations,
    )


def compute_pairwise_correlations(candidates: list[CandidateRecord]) -> list[PairwiseCorrelation]:
    rows: list[PairwiseCorrelation] = []
    for index, left in enumerate(candidates):
        left_series = left.oof["prediction"].reset_index(drop=True)
        for right in candidates[index + 1 :]:
            right_series = right.oof["prediction"].reset_index(drop=True)
            corr = _pearson_corr(left_series, right_series)
            rows.append(
                PairwiseCorrelation(
                    left_experiment_id=left.experiment_id,
                    right_experiment_id=right.experiment_id,
                    correlation=corr,
                    abs_correlation=abs(corr),
                )
            )
    return sorted(
        rows,
        key=lambda item: (
            item.abs_correlation,
            item.left_experiment_id,
            item.right_experiment_id,
        ),
        reverse=True,
    )


def _first_blocking_parent(
    candidate: CandidateRecord,
    *,
    accepted: list[CandidateRecord],
    correlation_lookup: dict[frozenset[str], PairwiseCorrelation],
    max_pairwise_corr: float,
) -> tuple[str, PairwiseCorrelation] | None:
    for parent in accepted:
        pair = correlation_lookup.get(frozenset((candidate.experiment_id, parent.experiment_id)))
        if pair is None:
            continue
        if pair.abs_correlation > max_pairwise_corr:
            return parent.experiment_id, pair
    return None


def _pearson_corr(left: pd.Series, right: pd.Series) -> float:
    corr = pd.Series(left, dtype=float).corr(pd.Series(right, dtype=float), method="pearson")
    if pd.isna(corr):
        return 1.0
    return float(corr)


def _sort_candidates_by_score(candidates: list[CandidateRecord]) -> list[CandidateRecord]:
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.oof_score is not None,
            candidate.oof_score if candidate.oof_score is not None else float("-inf"),
        ),
        reverse=True,
    )
