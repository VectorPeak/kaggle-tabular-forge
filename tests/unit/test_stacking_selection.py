import pandas as pd

from ktabforge.candidates import (
    CandidateCompatibilityRejection,
    CandidateRecord,
    CVProtocolMeta,
    PredictionArtifactMeta,
)
from ktabforge.stacking.selection import apply_selection_policy


def _candidate(
    experiment_id: str,
    *,
    score: float,
    predictions: list[float],
) -> CandidateRecord:
    target = "Churn"
    id_column = "id"
    oof = pd.DataFrame(
        {
            id_column: [1, 2, 3, 4, 5, 6],
            target: [0, 1, 0, 1, 0, 1],
            "prediction": predictions,
            "fold": [0, 0, 1, 1, 2, 2],
        }
    )
    test = pd.DataFrame(
        {
            id_column: [101, 102],
            target: [0.2, 0.8],
        }
    )
    return CandidateRecord(
        experiment_id=experiment_id,
        model_family="tree",
        oof_score=score,
        row={},
        prediction_meta=PredictionArtifactMeta(
            competition="churn_tiny",
            metric_name="roc_auc",
            prediction_type="probability",
            target=target,
            id_column=id_column,
            oof_row_count=len(oof),
            test_row_count=len(test),
        ),
        cv_protocol=CVProtocolMeta(
            cv_protocol_id="cv-1",
            splitter="StratifiedKFold",
            fold_count=3,
            seed=42,
            oof_safe=True,
        ),
        oof=oof,
        test=test,
    )


def test_apply_selection_policy_diversity_greedy_rejects_high_correlation_candidate():
    candidates = [
        _candidate(
            "candidate-a",
            score=0.91,
            predictions=[0.10, 0.90, 0.20, 0.80, 0.15, 0.85],
        ),
        _candidate(
            "candidate-b",
            score=0.90,
            predictions=[0.11, 0.89, 0.21, 0.79, 0.14, 0.86],
        ),
        _candidate(
            "candidate-c",
            score=0.88,
            predictions=[0.30, 0.78, 0.25, 0.72, 0.45, 0.88],
        ),
    ]

    result = apply_selection_policy(
        candidates,
        rejected=[],
        strategy="diversity_greedy",
        max_pairwise_corr=0.995,
    )

    assert [candidate.experiment_id for candidate in result.accepted] == [
        "candidate-a",
        "candidate-c",
    ]
    rejected_by_id = {item.experiment_id: item.reason for item in result.rejected}
    assert "candidate-b" in rejected_by_id
    assert "max_pairwise_corr" in rejected_by_id["candidate-b"]
    assert result.pairwise_correlations[0].left_experiment_id == "candidate-a"
    assert result.pairwise_correlations[0].right_experiment_id == "candidate-b"


def test_apply_selection_policy_score_desc_preserves_candidates_and_correlation_report():
    candidates = [
        _candidate(
            "candidate-a",
            score=0.91,
            predictions=[0.10, 0.90, 0.20, 0.80, 0.15, 0.85],
        ),
        _candidate(
            "candidate-b",
            score=0.90,
            predictions=[0.11, 0.89, 0.21, 0.79, 0.14, 0.86],
        ),
    ]
    prior_rejections = [CandidateCompatibilityRejection("candidate-z", "not compatible")]

    result = apply_selection_policy(
        candidates,
        rejected=prior_rejections,
        strategy="score_desc",
        max_pairwise_corr=None,
    )

    assert [candidate.experiment_id for candidate in result.accepted] == [
        "candidate-a",
        "candidate-b",
    ]
    assert result.rejected == prior_rejections
    assert len(result.pairwise_correlations) == 1
