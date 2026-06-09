# P4 候选实验工厂

P4 把 P2 的单实验 runner 包装成候选实验工厂：通过 matrix 配置生成多个可信 P2 实验配置，顺序执行，记录成功和失败，并产出候选报告供 P3 ensemble 使用。

P4 不是 HPO 系统，也不是并发调度器。MVP 只做小规模、串行、可审计的候选生产。

## 目标

P4 要回答：

- 本轮候选实验来自哪个 matrix 配置？
- 展开了多少个候选？
- 每个候选的 `experiment_id`、config hash、artifact path 是什么？
- 哪些候选成功，哪些失败，失败原因是什么？
- 哪些 completed candidates 可交给 P3 ensemble？

## CLI

先做 dry-run：

```powershell
uv run ktab factory --config configs\matrices\p04_churn_candidate_factory.example.yaml --dry-run
```

执行小矩阵：

```powershell
uv run ktab factory --config configs\matrices\p04_churn_candidate_factory.example.yaml --max-runs 2
```

`--max-runs` 会在调用 P2 runner 之前生效。

## Artifact

P4 生成：

```text
artifacts/factory/<competition>/<factory_id>/
  matrix_config.json
  matrix_plan.csv
  expanded_configs/
    <experiment_id>.yaml
  run_summary.csv
  run_summary.json
  candidate_report.md
```

单个 completed experiment 仍然使用 P2 artifact 契约：

```text
artifacts/experiments/<competition>/<experiment_id>/
artifacts/oof/<competition>/<experiment_id>/oof.parquet
artifacts/submissions/<competition>/<experiment_id>/submission.csv
artifacts/registry/<competition>/experiment_registry.csv
```

失败实验不伪装成 completed candidate，只进入 factory summary/report。

## Safety

P4 MVP 安全门：

- matrix 展开顺序确定。
- `experiment_id` 必须唯一。
- `experiment_id` 不允许空字符串、路径分隔符、`..` 或不安全字符。
- matrix axis 只用于实验变量，不允许覆盖 `data.artifact_root`、`experiment.competition`、`experiment_id` 或 `outputs.*`。
- factory 顶层的 `competition` 和 `artifact_root` 是生成 artifact 路径的权威来源。
- `max_runs` 在执行前截断计划。
- `dry_run` 不训练模型，不写 OOF/submission。
- P2 runner 异常会被记录到 `run_summary.csv`。
- `continue_on_error=false` 时，失败后的未执行候选会记录为 `skipped`。
- 默认串行执行，不做并发写 registry。
- 不自动触发 Kaggle submission。

## Deferred

P4 暂缓：

- 并发执行。
- HPO/search algorithm。
- GPU 队列。
- 自动 retry。
- 自动清理失败 artifact。
- 自动触发 P3 ensemble。
- 复杂 matrix DSL。
- public leaderboard 反馈。

## Acceptance

P4 可验收条件：

- matrix config 可以展开成多个 P2 experiment configs。
- duplicate experiment id 会被拒绝。
- dry-run 写出 plan/report，但不写 OOF/submission。
- factory runner 能调用 P2 runner。
- 成功和失败都有 summary。
- candidate report 包含 recommended candidate IDs。
- `ktab factory --config`、`--dry-run`、`--max-runs` 可用。
- `uv run pytest -q` 和 `uv run ruff check .` 通过。
