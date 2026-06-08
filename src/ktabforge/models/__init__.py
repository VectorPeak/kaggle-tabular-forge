"""Model adapters."""

from ktabforge.models.base import ModelOOFResult
from ktabforge.models.baseline import BaselineResult, run_logistic_oof_baseline
from ktabforge.models.registry import run_model_oof

__all__ = ["BaselineResult", "ModelOOFResult", "run_logistic_oof_baseline", "run_model_oof"]
