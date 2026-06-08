# Leakage and Leaderboard Rules

This project optimizes for trustworthy local evidence. Leakage prevention and
leaderboard discipline matter more than short-term score chasing.

## Anti-Leakage Rules

All agents must follow these rules:

1. Do not use target to construct test features.
2. Target encoding must be out-of-fold.
3. Target-dependent aggregation must be fit inside folds.
4. Group/time problems require group-aware or time-aware CV.
5. Do not fit a target-dependent transformer on full train before generating
   OOF.
6. Do not write public leaderboard feedback into training logic.
7. Do not repeatedly tune weights or thresholds by public leaderboard
   submissions.
8. Keep a `leakage_watchlist` for suspicious columns or transformations.

Risk labels:

```yaml
low:
  - unsupervised preprocessing
  - numeric scaling
  - missing-value indicators
medium:
  - train+test count encoding
  - test distribution alignment
  - pseudo labeling
high:
  - target encoding
  - time aggregations
  - entity historical stats
  - adversarial-validation-driven feature removal
forbidden:
  - test target usage
  - public leaderboard parameter search
  - manual probing by repeated submissions
```

## Target Encoding

Target encoding rules:

- For base OOF predictions, fit target encoders inside each training fold only.
- For nested target encoding, use an inner fold loop inside each outer CV fold.
- Do not fit a target encoder on full train and then create OOF predictions.
- Derived categorical crosses that use target statistics still need explicit
  review, even if raw categorical handling is delegated to CatBoost.

## External or Original Data

External/reference data can be powerful and dangerous.

Required review fields:

- source URL or provenance
- license or competition legality
- relationship to train/test generation
- whether labels are present
- whether labels are used
- nearest-neighbor columns
- scaling and distance metric
- risk label
- reviewer decision

Legal external priors must be distinguished from test target leakage.

## Anti-Leaderboard-Overfitting Rules

- Public leaderboard is a weak external signal, not the optimization target.
- OOF/CV is the primary decision signal.
- Do not override OOF decline because of one public LB gain.
- Ensemble weights must be primarily OOF-derived.
- If CV and LB diverge repeatedly, start a `debug_gap` goal instead of
  continuing blind submissions.

Each submission record must include:

```yaml
submission_id: string
experiment_id: string
local_oof: number
public_lb: number | null
private_lb: unknown
reason: string
expected_risk: low | medium | high
decision_after_result: keep | ignore | investigate
```

## Pre-Submission Check

No automatic Kaggle submission without human approval.

Required fields:

```yaml
experiment_id: string
oof_exists: true
fold_metrics_exist: true
inference_reproducible: true
sample_submission_format_valid: true
no_leakage_blocker: true
human_approved: true
```
