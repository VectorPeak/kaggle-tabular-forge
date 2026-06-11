from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ktabforge.candidates import CandidateCompatibilityRejection, CandidateRecord
from ktabforge.metrics.scoring import metric_higher_is_better, score_predictions

_EPSILON = 1e-12


@dataclass(frozen=True)
class PairwiseCorrelation:
    left_experiment_id: str
    right_experiment_id: str
    correlation: float
    abs_correlation: float


@dataclass(frozen=True)
class HillClimbStep:
    step: int
    experiment_id: str
    ensemble_score: float
    marginal_gain: float | None
    selected_experiment_ids: list[str]


@dataclass(frozen=True)
class SelectionPolicyResult:
    accepted: list[CandidateRecord]
    rejected: list[CandidateCompatibilityRejection]
    pairwise_correlations: list[PairwiseCorrelation]
    hill_climb_trace: list[HillClimbStep]


def apply_selection_policy(
    candidates: list[CandidateRecord],
    *,
    rejected: list[CandidateCompatibilityRejection],
    strategy: str,
    max_pairwise_corr: float | None,
    metric_name: str | None = None,
    target: str | None = None,
    min_gain: float = 0.0,
) -> SelectionPolicyResult:
    pairwise_correlations = compute_pairwise_correlations(candidates)
    kept_rejections = list(rejected)
    strategy_key = strategy.strip().lower()
    resolved_metric_name = metric_name or _infer_metric_name(candidates)
    ranked_candidates = _sort_candidates_by_score(candidates, resolved_metric_name)

    if strategy_key == "score_desc":
        return SelectionPolicyResult(
            accepted=ranked_candidates,
            rejected=kept_rejections,
            pairwise_correlations=pairwise_correlations,
            hill_climb_trace=[],
        )

    if strategy_key == "hill_climb_greedy":
        if target is None:
            raise ValueError("target is required when selection.strategy=hill_climb_greedy")
        if min_gain < 0:
            raise ValueError("selection.min_gain must be greater than or equal to 0")
        return _apply_hill_climb_greedy(
            ranked_candidates,
            rejected=kept_rejections,
            pairwise_correlations=pairwise_correlations,
            metric_name=resolved_metric_name,
            target=target,
            min_gain=min_gain,
        )

    if strategy_key != "diversity_greedy":
        raise ValueError(
            f"Unsupported selection.strategy {strategy!r}. "
            "Known strategies: score_desc, diversity_greedy, hill_climb_greedy."
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
        hill_climb_trace=[],
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


def _apply_hill_climb_greedy(
    candidates: list[CandidateRecord],
    *,
    rejected: list[CandidateCompatibilityRejection],
    pairwise_correlations: list[PairwiseCorrelation],
    metric_name: str,
    target: str,
    min_gain: float,
) -> SelectionPolicyResult:
    accepted: list[CandidateRecord] = []
    trace: list[HillClimbStep] = []
    remaining = list(candidates)
    current_predictions: pd.Series | None = None
    current_score: float | None = None

    while remaining:
        best = _best_hill_climb_candidate(
            remaining,
            accepted=accepted,
            current_predictions=current_predictions,
            metric_name=metric_name,
            target=target,
        )
        candidate, ensemble_predictions, next_score = best
        marginal_gain = None
        if current_score is not None:
            marginal_gain = _improvement_amount(metric_name, current_score, next_score)

        if current_score is not None and (
            marginal_gain is None or marginal_gain <= min_gain + _EPSILON
        ):
            for item in remaining:
                item_score = _candidate_hill_climb_score(
                    item,
                    accepted=accepted,
                    current_predictions=current_predictions,
                    target=target,
                    metric_name=metric_name,
                )
                item_gain = _improvement_amount(metric_name, current_score, item_score)
                rejected.append(
                    CandidateCompatibilityRejection(
                        experiment_id=item.experiment_id,
                        reason=(
                            "hill_climb_greedy stopped because marginal gain "
                            f"{item_gain:.6f} did not exceed min_gain={min_gain}"
                        ),
                    )
                )
            break

        accepted.append(candidate)
        current_predictions = ensemble_predictions
        current_score = next_score
        trace.append(
            HillClimbStep(
                step=len(trace) + 1,
                experiment_id=candidate.experiment_id,
                ensemble_score=next_score,
                marginal_gain=marginal_gain,
                selected_experiment_ids=[item.experiment_id for item in accepted],
            )
        )
        remaining = [
            item for item in remaining if item.experiment_id != candidate.experiment_id
        ]

    return SelectionPolicyResult(
        accepted=accepted,
        rejected=rejected,
        pairwise_correlations=pairwise_correlations,
        hill_climb_trace=trace,
    )


def _best_hill_climb_candidate(
    candidates: list[CandidateRecord],
    *,
    accepted: list[CandidateRecord],
    current_predictions: pd.Series | None,
    metric_name: str,
    target: str,
) -> tuple[CandidateRecord, pd.Series, float]:
    evaluations: list[tuple[CandidateRecord, pd.Series, float]] = []
    for candidate in candidates:
        candidate_predictions = candidate.oof["prediction"].reset_index(drop=True).astype(float)
        if current_predictions is None:
            ensemble_predictions = candidate_predictions
        else:
            ensemble_predictions = (
                current_predictions * len(accepted) + candidate_predictions
            ) / (len(accepted) + 1)
        score = float(
            score_predictions(metric_name, candidate.oof[target], ensemble_predictions)
        )
        evaluations.append((candidate, ensemble_predictions, score))

    higher_is_better = metric_higher_is_better(metric_name)
    if higher_is_better:
        return max(evaluations, key=lambda item: item[2])
    return min(evaluations, key=lambda item: item[2])


def _candidate_hill_climb_score(
    candidate: CandidateRecord,
    *,
    accepted: list[CandidateRecord],
    current_predictions: pd.Series | None,
    target: str,
    metric_name: str,
) -> float:
    candidate_predictions = candidate.oof["prediction"].reset_index(drop=True).astype(float)
    if current_predictions is None:
        ensemble_predictions = candidate_predictions
    else:
        ensemble_predictions = (
            current_predictions * len(accepted) + candidate_predictions
        ) / (len(accepted) + 1)
    return float(score_predictions(metric_name, candidate.oof[target], ensemble_predictions))


def _improvement_amount(metric_name: str, previous_score: float, next_score: float) -> float:
    if metric_higher_is_better(metric_name):
        return next_score - previous_score
    return previous_score - next_score


def _infer_metric_name(candidates: list[CandidateRecord]) -> str:
    if not candidates:
        raise ValueError("candidate pool is empty")
    return candidates[0].prediction_meta.metric_name


def _sort_candidates_by_score(
    candidates: list[CandidateRecord],
    metric_name: str,
) -> list[CandidateRecord]:
    higher_is_better = metric_higher_is_better(metric_name)
    missing_default = float("-inf") if higher_is_better else float("inf")
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.oof_score
            if candidate.oof_score is not None
            else missing_default
        ),
        reverse=higher_is_better,
    )
