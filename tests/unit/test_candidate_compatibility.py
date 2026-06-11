import pandas as pd

from ktabforge.candidates.compatibility import evaluate_candidate_compatibility
from ktabforge.candidates.contracts import CandidateRecord, CVProtocolMeta, PredictionArtifactMeta


def _candidate_record(
    experiment_id: str,
    *,
    prediction_type: str = "probability",
    model_family: str = "lightgbm",
    oof_safe: bool = True,
    include_fold: bool = True,
) -> CandidateRecord:
    payload = {
        "id": [1, 2, 3, 4],
        "Churn": [0, 1, 0, 1],
        "prediction": [0.1, 0.8, 0.2, 0.9],
    }
    if include_fold:
        payload["fold"] = [0, 0, 1, 1]
    oof = pd.DataFrame(payload)
    test = pd.DataFrame(
        {
            "id": [101, 102],
            "Churn": [0.2, 0.7],
        }
    )
    return CandidateRecord(
        experiment_id=experiment_id,
        model_family=model_family,
        oof_score=0.75,
        row={"experiment_id": experiment_id, "model_family": model_family},
        prediction_meta=PredictionArtifactMeta(
            competition="churn_tiny",
            metric_name="roc_auc",
            prediction_type=prediction_type,
            target="Churn",
            id_column="id",
            oof_row_count=4,
            test_row_count=2,
            oof_checksum=f"{experiment_id}-oof",
            test_checksum=f"{experiment_id}-test",
        ),
        cv_protocol=CVProtocolMeta(
            cv_protocol_id="stratified_kfold-5-seed42",
            splitter="stratified_kfold",
            fold_count=2 if include_fold else 0,
            seed=42,
            oof_safe=oof_safe,
        ),
        oof=oof,
        test=test,
    )


def test_candidate_compatibility_rejects_prediction_type_mismatch():
    compatible = _candidate_record("candidate-a", prediction_type="probability")
    incompatible = _candidate_record("candidate-b", prediction_type="raw_score")

    result = evaluate_candidate_compatibility(
        [compatible, incompatible],
        target="Churn",
        id_column="id",
        min_parents=1,
    )

    assert [candidate.experiment_id for candidate in result.accepted] == ["candidate-a"]
    assert len(result.rejected) == 1
    assert result.rejected[0].experiment_id == "candidate-b"
    assert "prediction_type" in result.rejected[0].reason


def test_candidate_compatibility_rejects_non_oof_safe_candidate():
    incompatible = _candidate_record("candidate-b", oof_safe=False)
    compatible = _candidate_record("candidate-a", oof_safe=True)

    result = evaluate_candidate_compatibility(
        [incompatible, compatible],
        target="Churn",
        id_column="id",
        min_parents=1,
    )

    assert [candidate.experiment_id for candidate in result.accepted] == ["candidate-a"]
    rejected = {item.experiment_id: item.reason for item in result.rejected}
    assert "candidate-b" in rejected
    assert "oof_safe" in rejected["candidate-b"]


def test_candidate_compatibility_rejects_candidate_without_fold_column():
    incompatible = _candidate_record("candidate-b", include_fold=False)
    compatible = _candidate_record("candidate-a", include_fold=True)

    result = evaluate_candidate_compatibility(
        [incompatible, compatible],
        target="Churn",
        id_column="id",
        min_parents=1,
    )

    assert [candidate.experiment_id for candidate in result.accepted] == ["candidate-a"]
    rejected = {item.experiment_id: item.reason for item in result.rejected}
    assert "candidate-b" in rejected
    assert "fold" in rejected["candidate-b"]
