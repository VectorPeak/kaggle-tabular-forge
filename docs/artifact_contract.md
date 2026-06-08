# Artifact Contract

All generated outputs should live under `artifacts/`. Raw data and generated
artifacts should not be committed to git.

## Recommended Structure

```text
artifacts/
  experiments/<competition>/<experiment_id>/
    config.yaml
    metrics.json
    params.json
    feature_manifest.json
    model_manifest.json
    environment.json
    git.json
    notes.md
    logs/
  oof/<competition>/<experiment_id>/
    oof.parquet
    oof.npy
    fold_metrics.csv
    schema.json
  submissions/<competition>/<experiment_id>/
    pred.npy
    submission.csv
    submission_review.md
    public_lb.json
    checksum.txt
  reports/<competition>/<experiment_id>/
    eda.md
    validation_report.md
    error_analysis.md
    leaderboard_notes.md
  llm_proposals/<competition>/<proposal_id>/
    prompt.md
    response.md
    proposal.json
    accepted_changes.md
    review.md
  eda_findings/<competition>/
  candidate_selection/<competition>/
```

## Experiment Id

Format:

```text
YYYYMMDD-HHMM-<short_name>
```

Examples:

```text
20260606-1430-lgbm-baseline-v1
20260607-0915-cuml-features-xgb
20260608-2210-stack-lgbm-cat-xgb
```

Rules:

- Use a new experiment id for every run that writes artifacts.
- Do not overwrite old experiment directories.
- Keep experiment ids stable across OOF, predictions, reports, and submission
  review.

## File Type Guidance

Use YAML for human-edited intent:

- competition config
- experiment config
- paths config
- agents config
- stacking presets

Use JSON for machine-written records:

- `metrics.json`
- `params.json`
- `environment.json`
- `git.json`
- `proposal.json`
- `run_manifest.json`
- `model_manifest.json`
- `feature_manifest.json`
- `public_lb.json`

Use Parquet for auditable tabular outputs:

- `oof.parquet`
- `test_predictions.parquet`
- `feature_cache.parquet`
- `experiment_registry.parquet`
- `candidate_pool.parquet`
- EDA profiling tables
- fold prediction details

Use `.npy` when stacking speed or storage efficiency matters:

- `oof.npy`
- `pred.npy`
- large prediction matrices

Use Markdown for human review:

- `eda.md`
- `validation_report.md`
- `submission_review.md`
- `leaderboard_notes.md`
- `feature_catalog.md`
- `model_zoo.md`
- `stacking.md`
- `environment_notes.md`
- `leakage_review.md`

## OOF Alignment Requirements

Every OOF artifact must record:

- competition name
- experiment id
- id column
- target column
- prediction column
- row count
- fold column if applicable
- metric and metric mode
- feature family list
- model family
- seed
- checksum

Every test prediction artifact must align with the sample submission id order or
record the transformation used to restore that order.
