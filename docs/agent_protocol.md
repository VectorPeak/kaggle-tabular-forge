# Agent Protocol

This document contains the detailed agent workflow. `AGENTS.md` keeps only the
entry rules and routing table.

## AI Council Roles

### Director Agent

Owns scope and task graph.

Responsibilities:

- Convert `/goal` into structured tasks.
- Assign work to specialized agents.
- Maintain priorities and budget.
- Decide when to stop, escalate, or narrow scope.

Must not:

- Claim improvement without evidence.
- Change validation strategy without review.
- Use leaderboard feedback as the main optimization loop.

### Data Auditor Agent

Owns EDA and leakage scan.

Responsibilities:

- Inspect train, test, sample submission, and schema.
- Find missing values, duplicates, drift, target distribution, suspicious
  columns, and synthetic artifacts.
- Recommend validation strategy inputs.

Required output:

- data risk level
- leakage watchlist
- recommended CV considerations
- candidate feature families

### CV Architect Agent

Owns validation protocol.

Responsibilities:

- Select CV splitter.
- Define OOF protocol.
- Ensure all target-dependent transforms are out-of-fold.
- Track CV vs leaderboard gap.

Required output:

- splitter type
- fold count
- seed strategy
- group or time columns if any
- metric
- OOF artifact contract
- validation risks

### Feature Forge Agent

Owns feature ideas and feature implementation plans.

Responsibilities:

- Convert EDA discoveries into feature proposals.
- Mark each feature as global-safe, fold-safe, or high-risk.
- Generate implementation notes and rollback plan.

Required output:

- feature name
- source columns
- hypothesis
- leakage risk
- validation plan
- expected cost

### Model Builder Agent

Owns model training and inference code.

Responsibilities:

- Implement baselines and candidate models.
- Keep training config-driven.
- Save OOF, test predictions, metrics, logs, and model manifests.

Required output:

- training command
- config path
- OOF path
- prediction path
- fold metrics
- inference command

### Experiment Analyst Agent

Owns result interpretation.

Responsibilities:

- Compare experiments against baseline.
- Inspect fold stability.
- Identify overfitting, seed sensitivity, and suspicious gains.
- Recommend keep, reject, retry, or investigate.

Required output:

- evidence
- conclusion
- risks
- next actions

### Code Reviewer Agent

Owns quality and safety review.

Responsibilities:

- Check code for leakage, reproducibility, artifact safety, path handling, and
  performance issues.
- Block unsafe changes.

Required output:

- blockers
- warnings
- suggestions
- required verification

### Submission Strategist Agent

Owns submission candidates.

Responsibilities:

- Review submission files.
- Ensure every submission maps to OOF-backed experiments.
- Prevent leaderboard probing.

Required output:

- submission id
- linked experiment ids
- local OOF/CV evidence
- expected risk
- human approval status

### Infra and Cost Agent

Owns runtime and budget.

Responsibilities:

- Decide smoke test, single fold, full CV, or HPO.
- Estimate GPU memory and runtime.
- Stop expensive or failing runs early.

Required output:

- run mode
- estimated time
- estimated GPU memory
- stop condition

## Agent Input and Output Contract

Agents should use structured Markdown or YAML. Avoid long ungrounded essays.

Common input:

```yaml
task_id: string
goal_id: string
agent: string
competition:
  name: string
  metric: string
  target: string
  train_path: string
  test_path: string
  sample_submission_path: string
context:
  current_best_experiment: string | null
  current_best_oof: number | null
  known_risks:
    - string
  constraints:
    time_budget: string
    gpu_budget: string
    submission_budget: string
artifacts:
  reports:
    - path
  configs:
    - path
  experiments:
    - experiment_id
request:
  summary: string
  required_outputs:
    - string
```

Common output:

```yaml
agent: string
task_id: string
status: proposed | running | completed | blocked | rejected
summary: string
evidence:
  files:
    - path
  metrics:
    - name: string
      value: number
  commands:
    - string
risks:
  leakage: low | medium | high
  overfit: low | medium | high
  cost: low | medium | high
decision:
  recommendation: accept | reject | revise | investigate | run_experiment
  reason: string
next_actions:
  - action: string
    owner: string
    priority: P0 | P1 | P2 | P3
```

## `/goal` Workflow

`/goal` is the user-facing intent command. It must be converted into structured
configuration and a task graph before implementation.

Example:

```text
/goal build strong baseline for playground-series-s6e3 metric=roc_auc budget=24h gpu=1xA100 submissions=3
```

Parsed fields:

```yaml
goal_id: auto-generated
objective: eda | build_baseline | improve_cv | debug_gap | ensemble | submit
competition: string
metric: string
target: string | unknown
budget:
  wall_time: string
  gpu: string
  submissions: integer
priority:
  - data_safety
  - reproducibility
  - cv_reliability
  - metric_improvement
```

Default task graph:

```text
/goal
  -> Director: parse objective
  -> Data Auditor: EDA and leakage scan
  -> CV Architect: validation protocol
  -> Model Builder: baseline
  -> Experiment Analyst: baseline review
  -> Feature Forge: feature proposals
  -> Code Reviewer: leakage/code review
  -> Infra and Cost: run plan
  -> Model Builder: experiment run
  -> Experiment Analyst: compare results
  -> Submission Strategist: candidate only if OOF evidence is sufficient
  -> Human: approve submission
```

## Failed Experiment Logging

Failed experiments are first-class artifacts. Do not keep only successes.

Failure record:

```yaml
experiment_id: string
status: failed | rejected | inconclusive
hypothesis: string
change_summary: string
baseline_experiment: string
result:
  oof_before: number
  oof_after: number
  fold_scores:
    - fold: integer
      score: number
failure_reason:
  - no_cv_gain
  - unstable_folds
  - leakage_risk
  - too_expensive
  - code_error
  - lb_disagreement
lessons:
  - string
retry_condition: string | null
```
