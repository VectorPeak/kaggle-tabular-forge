# Stacking and Ensemble

OOF predictions are the central asset for ensemble construction.

## Current Status

Current repository support covers:

1. candidate-pool loading from registry-backed artifacts
2. compatibility filtering for OOF-safe stacking inputs
3. correlation-aware diversity selection
4. OOF-backed greedy hill climbing
5. first-level completed stacking with `logistic_regression` or `ridge_classifier`

Operational constraints already enforced in code:

- `top_n` must be applied after candidate compatibility and selection, not before
- `max_parents` must act as an in-selection cap, not a post-hoc trim
- `hill_climb_greedy` cannot be combined with `max_pairwise_corr`
- stacking config rejects non-finite numeric values and unknown key typos
- selection report / manifest must distinguish configured limits from effective limits

## Candidate Pool Strategy

Target workflow:

1. Generate many candidate models and feature pipelines.
2. Save OOF and test predictions for every candidate.
3. Reject unsafe or unverified candidates.
4. Use OOF score and prediction correlation to select a diverse subset.
5. Build final ensembles from the diverse subset.

The S6E3 pattern was:

- hundreds of potential models
- final subset of about 150
- multiple tree libraries
- many neural architectures
- many feature-engineering pipelines
- multiple seeds and tuned parameter sets

This project should preserve the principle, not hardcode the exact numbers.

## Required Prediction Artifacts

Every ensemble candidate must save OOF and test predictions.

Historical naming convention:

```text
oof_<description>_v<version>.npy
pred_<description>_v<version>.npy
```

Preferred project-native formats:

```text
artifacts/oof/<competition>/<experiment_id>/oof.parquet
artifacts/oof/<competition>/<experiment_id>/oof.npy
artifacts/submissions/<competition>/<experiment_id>/pred.npy
artifacts/submissions/<competition>/<experiment_id>/submission.csv
```

Rules:

- `.npy` is acceptable for large-scale stacking speed.
- `.parquet` is preferred for auditability and schema.
- The manifest must map each prediction file to experiment id, model family,
  feature family, fold strategy, and seed.

## Greedy Hill Climbing

Use OOF-based greedy forward selection:

1. Start with the best single candidate or a stable seed blend.
2. Add one candidate at a time.
3. Keep the candidate only if OOF improves under the competition metric.
4. Track correlation and marginal gain.
5. Stop when no candidate improves or gain falls below threshold.

Rules:

- Hill climbing must use OOF, not public leaderboard.
- Candidate order, metric, weights, and rejected candidates must be logged.
- `min_gain` must be finite and non-negative.

## Logistic Regression Stacking

Use L2-penalized logistic regression or linear/ridge meta-models as stackers
when appropriate.

Purpose:

- prevent any single model from dominating
- handle highly correlated model predictions
- learn which prediction source to trust in different regions of prediction
  space

Rules:

- The stacker itself must be trained with OOF-safe protocol.
- Level 2 and Level 3 stacking must save their own OOF/test predictions.
- Final stacker lineage must list all input experiments.
- `preflight_only` is allowed for `stack-preflight`, but not for `ktab stack`.

## Multi-Level Stacking

Supported conceptual levels:

- Level 0: original feature models.
- Level 1: feature extraction models or base model predictions.
- Level 2: models that consume earlier OOF predictions as additional features.
- Level 3: second-order stackers.
- Final level: logistic/linear meta-learner or final blender.

Rules:

- Never train a stacker on in-sample base predictions.
- Every level must have OOF predictions.
- `_stk` suffix or equivalent metadata should mark models that consume earlier
  OOF predictions.
- Lineage must be queryable from final submission back to all base experiments.

## Diversity Strategy

Model diversity should be designed, not accidental.

Track diversity across:

- model family: XGBoost, LightGBM, CatBoost, YDF, cuML RF, neural models
- tree growth style: depth-wise, leaf-wise, symmetric, bagging
- feature family: snap, TE, digit, radix, KDTree, DAE, TF-IDF, projection
- objective: binary classification, ranking, calibrated logits
- random seed
- hyperparameter search path
- training framework

Experiment Analyst must report:

- OOF score
- fold stability
- OOF correlation with current ensemble
- marginal gain when added
- cost of retaining the model
