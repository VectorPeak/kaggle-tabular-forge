# KGMON Methodology

This project is inspired by the S6E3 KGMON-style workflow: a human-led,
agent-expanded, OOF-verified tabular competition system.

## Mission

`kaggle-tabular-forge` exists to systematize the workflow used in strong Kaggle
tabular solutions:

1. Understand the data deeply through EDA.
2. Build honest baselines.
3. Mine synthetic-data fingerprints when the competition data is generated.
4. Generate many feature and model hypotheses.
5. Run experiments with strict CV and OOF discipline.
6. Preserve every useful and failed experiment.
7. Select diverse models from a larger candidate pool.
8. Combine them through OOF-backed hill climbing, stacking, and final blending.
9. Produce submissions only after review and human approval.

## Non-Goals

- Do not build a generic AutoML platform.
- Do not optimize directly against public leaderboard feedback.
- Do not treat LLM-generated code or claims as trusted without CV/OOF evidence.
- Do not require GPU for the MVP.
- Do not hide experiments in notebooks or chat logs only.
- Do not let agents overwrite historical artifacts.
- Do not auto-submit to Kaggle without human approval.

## Core Pattern to Preserve

The S6E3 writeup highlights a repeatable pattern:

- Create a large candidate pool, such as hundreds of models.
- Select a smaller, diverse final set.
- Build hundreds of feature-engineering variants; feature engineering is the
  primary performance driver.
- Let most final models share a strong common feature core, then vary
  specialized techniques around that core.
- Ask multiple LLM agents to run independent EDA, especially to discover the
  relationship between large synthetic train data, smaller original data, and
  target behavior.
- Convert discoveries into testable feature ideas, not unverified claims.

The exact numbers are not the product. The disciplined search process is the
product.

## MVP Requirements

The MVP should make one tabular Kaggle competition reproducible from EDA to the
first ensemble.

Required capabilities:

- Standard project layout for data, configs, experiments, OOF, submissions,
  reports, and LLM proposals.
- A `goal.yaml` or equivalent structured goal file per competition.
- Data sanity checks:
  - train/test/sample submission shape
  - id alignment
  - target distribution
  - missing values
  - categorical cardinality
  - train/test drift
  - obvious leakage risks
- Validation design:
  - KFold
  - StratifiedKFold
  - GroupKFold
  - time-aware split where needed
- Baselines:
  - linear or logistic baseline
  - one GBDT baseline: LightGBM, XGBoost, or CatBoost
- Feature engineering:
  - numeric transforms
  - categorical encoding
  - count/frequency encoding
  - target encoding with out-of-fold safety
  - cross features
  - synthetic fingerprint features when applicable
- Experiment registry:
  - one unique `experiment_id` per run
  - config path
  - command
  - data version
  - fold strategy
  - metric
  - model family
  - feature set
  - params
  - OOF path
  - test prediction path
  - fold metrics
  - runtime/hardware metadata
- OOF discipline:
  - all ensemble candidates must have OOF predictions
  - OOF rows must align with train rows
  - test predictions must align with sample submission
  - fold metrics must be saved
- Combination:
  - simple average
  - weighted average
  - rank average
  - hill climbing
  - first-level stacking
- Reports:
  - EDA report
  - validation report
  - experiment summary
  - ensemble report
  - submission review
- AI Council records:
  - hypothesis
  - expected lift
  - implementation notes
  - leakage risk
  - validation plan
  - result
  - decision

## Enhanced Requirements

Later versions should support the high-throughput pattern described in the
NVIDIA/KGMON workflow:

- Hundreds of experiment configurations.
- Experiment queue with priority, retries, and failure logging.
- Model diversity analysis using OOF correlation.
- Multi-level stacking:
  - feature extraction level
  - base model level
  - intermediate stacker level
  - final logistic/linear stacker level
- Pseudo-labeling with strict provenance and risk labels.
- GPU feature engineering with cuDF/cuML when available.
- GPU model training with XGBoost, PyTorch, cuML, or compatible libraries.
- Adversarial validation.
- Synthetic train vs original data relationship mining.
- Original-data lookup and nearest-neighbor anchoring when legal and available.
- Automated agent proposal backlog.
- Competition report generation.
- Kaggle API integration for download and optional human-approved submission.

## Agentic EDA Requirement

For synthetic tabular competitions, EDA should explicitly study:

- synthetic train vs original/reference data
- test distribution vs train distribution
- target behavior around reconstructed original anchors
- row-level generator artifacts
- feature interactions suggested by domain equations
- high-risk leakage hypotheses that must be rejected or reviewed

Every EDA finding should be converted into one of:

- feature proposal
- validation concern
- leakage watchlist item
- model family suggestion
- experiment backlog item
- rejection note with rationale
