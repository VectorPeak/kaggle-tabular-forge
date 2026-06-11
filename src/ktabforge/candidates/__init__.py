"""Candidate pool utilities for ensemble construction."""

from ktabforge.candidates.compatibility import evaluate_candidate_compatibility
from ktabforge.candidates.contracts import (
    CandidateCompatibilityRejection,
    CandidateCompatibilityResult,
    CandidateRecord,
    CVProtocolMeta,
    PredictionArtifactMeta,
)

__all__ = [
    "CVProtocolMeta",
    "CandidateCompatibilityRejection",
    "CandidateCompatibilityResult",
    "CandidateRecord",
    "PredictionArtifactMeta",
    "evaluate_candidate_compatibility",
]

