# Goal Contract

This document is the canonical contract for `/goal`.

`/goal` should be easy for a human to write and strict for the system to
validate. The human entry point stays small; the internal acceptance record is
evidence-heavy.

Machine-readable starting point:

- `configs/goal.example.yaml`
- `configs/schemas/goal.schema.json`

## Minimal User Input

The human should provide four fields whenever possible:

```text
/goal objective=<objective> competition=<competition> request="<intent>" budget=<budget>
```

Required fields:

| Field | Purpose | Example |
| --- | --- | --- |
| `objective` | Goal type | `eda`, `build_baseline`, `feature_test`, `improve_cv`, `ensemble`, `submit`, `debug_gap` |
| `competition` | Kaggle slug or local competition id | `playground-series-s6e3` |
| `request` | Human intent in one sentence | `build first honest LightGBM baseline` |
| `budget` | Coarse cost boundary | `static_review`, `smoke`, `single_fold`, `full_cv`, `24h`, `gpu=1xA100` |

Optional fields:

- `metric`
- `target`
- `baseline_experiment`
- `run_mode`
- `risk_tolerance`
- `submission_budget`
- `model`
- `feature_family`
- `method`
- `submit=false`

## Automatic Discovery

The system should discover these fields before asking the human:

| Field | Preferred source |
| --- | --- |
| `metric` and `metric_mode` | competition config |
| `target` and `id_column` | competition config, train columns, sample submission |
| data paths | paths config, competition config, standard directories |
| task type | competition config or target distribution |
| validation strategy | competition config or safe defaults |
| current best experiment | experiment registry or OOF artifacts |
| OOF candidates | `artifacts/oof/<competition>/` and registry |
| model and feature registries | `configs/` and competition overrides |
| `experiment_id` | generated as `YYYYMMDD-HHMM-<short_name>` |
| artifact paths | artifact contract defaults |
| submission policy | human approval required by default |

Only ask the human when missing context affects safety, comparability, or
submission risk.

## Required Follow-Up Questions

Ask a concise follow-up when any of these cannot be discovered:

- `objective`
- `competition`
- `metric`
- `target`
- data paths
- group/time/entity validation constraints
- `baseline_experiment` for `feature_test`, `improve_cv`, or `debug_gap`
- candidate list for `ensemble` or `submit`
- external/original data legality
- explicit human approval for a Kaggle submission

Ask at most three questions at once.

## Objective-Specific Context

| Objective | Extra context |
| --- | --- |
| `eda` | focus areas such as leakage, drift, target distribution, synthetic fingerprints, external/original data |
| `build_baseline` | model preference, run mode, safe baseline only or submission candidate |
| `feature_test` | feature hypothesis, baseline experiment, risk tolerance, promotion rule |
| `improve_cv` | current best, editable surface, budget, stability vs score preference |
| `ensemble` | candidate source, filtering rules, method, stacking depth |
| `submit` | submission candidate, OOF evidence, submission budget, human approval |
| `debug_gap` | local CV, public LB, submission ids, gap direction, candidate lineage |

## Acceptance Record

Every `/goal` must end with an acceptance record. A completed goal means the
evidence chain is closed for that objective, not merely that a model score was
printed.

```yaml
goal_acceptance:
  schema_version: goal_acceptance.v0
  goal:
    goal_id: string
    created_at: string
    status: parsed | planned | running | completed | rejected | inconclusive | blocked
    objective: eda | build_baseline | improve_cv | feature_test | ensemble | submit | debug_gap
    competition: string
    metric: string
    metric_mode: maximize | minimize
    target: string | unknown
    constraints:
      wall_time: string | null
      gpu_budget: string | null
      submission_budget: integer | null

  discovery:
    metric:
      value: string | null
      source: path | inferred | missing
      confidence: high | medium | low | missing
    target:
      value: string | null
      source: path | inferred | missing
      confidence: high | medium | low | missing
    data_paths:
      train: path | null
      test: path | null
      sample_submission: path | null
    provenance:
      configs:
        - path
      artifacts:
        - path

  required_artifacts:
    experiment_id: string | null
    reports:
      - path
    configs:
      - path
    oof:
      path: path | null
      row_count: integer | null
      checksum: string | null
    predictions:
      path: path | null
      checksum: string | null
    submission:
      path: path | null
      checksum: string | null
    manifests:
      feature_manifest: path | null
      model_manifest: path | null
      environment: path | null
      git: path | null
      lineage: path | null
    failure_record: path | null

  evidence:
    commands:
      - string
    metrics:
      baseline_experiment_id: string | null
      baseline_oof: number | null
      current_oof: number | null
      delta: number | null
      fold_metrics:
        - fold: integer
          score: number
      seed_metrics:
        - seed: integer
          score: number
    cv_protocol:
      cv_protocol_id: string
      splitter: string
      fold_count: integer
      seed: integer | string
      group_column: string | null
      time_column: string | null
      oof_safe: boolean
    comparison:
      conclusion: improved | no_gain | regressed | signal_only | not_applicable
      reason: string

  safety_gates:
    no_leakage_blocker: boolean
    leakage_watchlist_reviewed: boolean
    target_encoding_oof_safe: boolean | not_applicable
    no_full_train_target_dependent_oof: boolean
    external_data_reviewed: boolean | not_applicable
    no_public_lb_optimization: boolean
    no_auto_submit: boolean
    oof_alignment_verified: boolean | not_applicable
    submission_format_valid: boolean | not_applicable
    human_approved: boolean | not_required

  reproducibility:
    git_commit: string | null
    git_dirty: boolean | unknown
    environment_recorded: boolean
    config_recorded: boolean
    command_recorded: boolean
    seeds_recorded: boolean
    artifact_overwrite_check: passed | failed | unknown

  decision:
    status: completed | rejected | inconclusive | blocked
    recommendation: accept | reject | revise | investigate | run_experiment | request_human_approval
    allowed_claims:
      - string
    forbidden_claims:
      - string
    reason: string
    next_actions:
      - action: string
      owner: human | agent
      priority: P0 | P1 | P2 | P3

  errors:
    - string
  warnings:
    - string
```

## Objective Acceptance Rules

`eda` can complete without OOF if it produces a data report, leakage watchlist,
CV recommendation, and feature hypotheses.

`build_baseline` must produce a baseline config, command, fold metrics, OOF
artifact, environment record, and validation report.

`feature_test` must produce a feature manifest, risk label, baseline comparison,
and fold-stability decision. Single-fold signal is not a completed improvement.

`improve_cv` must compare against a baseline experiment with full CV OOF unless
the explicit goal was only exploratory.

`ensemble` must use only candidates with aligned OOF and test predictions. It
must save candidate selection, weights or stacker lineage, and rejected reasons.

`submit` must pass submission format checks, link to OOF-backed experiments, and
record explicit human approval.

`debug_gap` must record the CV/LB gap, candidate lineage, suspected causes, and
investigation outcome. It does not need to produce a better model.

## Inconclusive Conditions

A goal is `inconclusive`, not `completed`, when:

- it has only smoke or single-fold evidence but claims improvement
- it lacks OOF artifact, fold metrics, checksum, or experiment id
- OOF or sample-submission alignment cannot be verified
- CV protocol is missing or incompatible
- target-dependent features lack leakage review
- public leaderboard gain is the only positive evidence
- ensemble candidates lack aligned OOF
- submission lacks human approval
- artifacts were not written or may have overwritten old artifacts
- command, config, environment, or git state is insufficient for reproduction
