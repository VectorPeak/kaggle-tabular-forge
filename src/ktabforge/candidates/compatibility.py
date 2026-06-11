from __future__ import annotations

from dataclasses import dataclass

from ktabforge.candidates.contracts import (
    CandidateCompatibilityRejection,
    CandidateCompatibilityResult,
    CandidateRecord,
)


@dataclass(frozen=True)
class _CompatibilityReference:
    competition: str
    metric_name: str
    prediction_type: str
    target: str
    id_column: str
    cv_protocol_id: str
    fold_count: int
    oof_safe: bool
    oof_ids: list[object]
    target_values: list[object]
    fold_values: list[object] | None
    test_ids: list[object]


def evaluate_candidate_compatibility(
    candidates: list[CandidateRecord],
    *,
    target: str,
    id_column: str,
    min_parents: int = 2,
) -> CandidateCompatibilityResult:
    accepted: list[CandidateRecord] = []
    rejected: list[CandidateCompatibilityRejection] = []
    reference: _CompatibilityReference | None = None

    for candidate in candidates:
        reason = _validate_candidate(candidate, target=target, id_column=id_column)
        if reason is not None:
            rejected.append(
                CandidateCompatibilityRejection(
                    experiment_id=candidate.experiment_id,
                    reason=reason,
                )
            )
            continue

        if reference is None:
            accepted.append(candidate)
            reference = _build_reference(candidate, target=target, id_column=id_column)
            continue

        mismatch = _find_reference_mismatch(
            candidate,
            reference=reference,
            target=target,
            id_column=id_column,
        )
        if mismatch is not None:
            rejected.append(
                CandidateCompatibilityRejection(
                    experiment_id=candidate.experiment_id,
                    reason=mismatch,
                )
            )
            continue
        accepted.append(candidate)

    if min_parents > 0 and len(accepted) < min_parents:
        reason = f"accepted candidate count {len(accepted)} is below min_parents={min_parents}"
        rejected.extend(
            CandidateCompatibilityRejection(
                experiment_id=candidate.experiment_id,
                reason=reason,
            )
            for candidate in accepted
        )
        accepted = []

    return CandidateCompatibilityResult(accepted=accepted, rejected=rejected)


def _validate_candidate(candidate: CandidateRecord, *, target: str, id_column: str) -> str | None:
    meta = candidate.prediction_meta
    protocol = candidate.cv_protocol
    if meta.target != target:
        return f"target mismatch: expected {target}, got {meta.target}"
    if meta.id_column != id_column:
        return f"id_column mismatch: expected {id_column}, got {meta.id_column}"
    if not protocol.oof_safe:
        return "oof_safe must be true"
    if len(candidate.oof) != meta.oof_row_count:
        return "prediction_meta.oof_row_count does not match oof rows"
    if len(candidate.test) != meta.test_row_count:
        return "prediction_meta.test_row_count does not match test rows"

    required_oof = {id_column, target, "prediction"}
    missing_oof = required_oof.difference(candidate.oof.columns)
    if missing_oof:
        return "oof is missing required columns: " + ", ".join(sorted(missing_oof))
    if "fold" not in candidate.oof.columns:
        return "oof is missing required columns: fold"
    if protocol.fold_count <= 0:
        return "fold_count must be greater than 0"

    required_test = {id_column, target}
    missing_test = required_test.difference(candidate.test.columns)
    if missing_test:
        return "test is missing required columns: " + ", ".join(sorted(missing_test))

    return None


def _build_reference(
    candidate: CandidateRecord,
    *,
    target: str,
    id_column: str,
) -> _CompatibilityReference:
    oof = candidate.oof.reset_index(drop=True)
    test = candidate.test.reset_index(drop=True)
    meta = candidate.prediction_meta
    protocol = candidate.cv_protocol
    fold_values = oof["fold"].tolist() if "fold" in oof.columns else None
    return _CompatibilityReference(
        competition=meta.competition,
        metric_name=meta.metric_name,
        prediction_type=meta.prediction_type,
        target=target,
        id_column=id_column,
        cv_protocol_id=protocol.cv_protocol_id,
        fold_count=protocol.fold_count,
        oof_safe=protocol.oof_safe,
        oof_ids=oof[id_column].tolist(),
        target_values=oof[target].tolist(),
        fold_values=fold_values,
        test_ids=test[id_column].tolist(),
    )


def _find_reference_mismatch(
    candidate: CandidateRecord,
    *,
    reference: _CompatibilityReference,
    target: str,
    id_column: str,
) -> str | None:
    meta = candidate.prediction_meta
    protocol = candidate.cv_protocol
    oof = candidate.oof.reset_index(drop=True)
    test = candidate.test.reset_index(drop=True)

    checks: list[tuple[object, object, str]] = [
        (meta.competition, reference.competition, "competition"),
        (meta.metric_name, reference.metric_name, "metric_name"),
        (meta.prediction_type, reference.prediction_type, "prediction_type"),
        (meta.target, reference.target, "target"),
        (meta.id_column, reference.id_column, "id_column"),
        (protocol.cv_protocol_id, reference.cv_protocol_id, "cv_protocol_id"),
        (protocol.fold_count, reference.fold_count, "fold_count"),
        (protocol.oof_safe, reference.oof_safe, "oof_safe"),
    ]
    for value, expected, label in checks:
        if value != expected:
            return f"{label} mismatch: expected {expected}, got {value}"

    if oof[id_column].tolist() != reference.oof_ids:
        return "oof ids do not align"
    if oof[target].tolist() != reference.target_values:
        return "oof targets do not align"

    fold_values = oof["fold"].tolist() if "fold" in oof.columns else None
    if reference.fold_values is not None and fold_values != reference.fold_values:
        return "oof folds do not align"
    if test[id_column].tolist() != reference.test_ids:
        return "test ids do not align"
    return None
