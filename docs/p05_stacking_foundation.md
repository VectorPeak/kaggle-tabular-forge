# P5 Stacking / Selection 基础能力

P5 把仓库从 “P3 轻量 ensemble + P4 候选工厂” 往前推了一步，补上了候选筛选、OOF 驱动 hill climbing、以及一层 OOF-safe stacking 的基础闭环。

这个阶段的重点不是追求最强 Kaggle 分数，而是先把一层 stacking 做到可配置、可审计、可回溯，并把最容易误导人的选择语义收紧。

## 目标

P5 要回答：

- 哪些 completed candidate 可以成为 stacking parent？
- 它们是否满足 OOF-safe、CV protocol 兼容、目标列与 id 对齐？
- 候选筛选到底是按分数直排、按相关性去重，还是按 hill climbing 逐步吸收？
- 一层 stacker 用了什么方法、哪些 parent、哪些限制条件？
- selection report、stacking manifest、registry lineage 是否足够诚实，能让人回放整个过程？

## 已落地能力

当前仓库中，P5 已经具备以下能力：

1. registry-backed candidate pool 读取。
2. stacking parent 的 compatibility gate。
3. `score_desc`、`diversity_greedy`、`hill_climb_greedy` 三种选择策略。
4. OOF-backed greedy hill climbing。
5. 一层 completed stacking，支持：
   - `logistic_regression`
   - `ridge_classifier`
6. `ktab stack-preflight --config ...`
7. `ktab stack --config ...`

## CLI

### 1. 仅做 preflight

```powershell
uv run ktab stack-preflight --config configs\stacking.example.yaml
```

这个命令会：

- 读取 stacking config
- 选择 compatible parents
- 生成 `stack_oof.parquet` 和 `stack_test.parquet`
- 写出 selection report 和 stacking manifest
- 不训练最终 stacker

### 2. 完成一层 stacking

```powershell
uv run ktab stack --config configs\stacking.example.yaml
```

这个命令会：

- 复用 preflight 的选择逻辑
- 训练 OOF-safe 一层 stacker
- 写出 stacked OOF、submission、fold metrics、manifest 和 registry row

## Selection 策略

P5 当前支持三种策略：

### `score_desc`

按单模型 OOF 分数排序，保留前若干个 compatible parents。

用途：

- 做最保守的 baseline stack
- 验证 artifact / lineage / stacker runtime 是否连通

### `diversity_greedy`

按单模型 OOF 分数排序，但候选加入 accepted 集合前，要经过 pairwise correlation 约束。

用途：

- 去掉高度重复的 parent
- 保持 parent 集合可解释

要求：

- 必须配置 `selection.max_pairwise_corr`

### `hill_climb_greedy`

从当前最优候选出发，逐步把会提升 ensemble OOF 分数的 candidate 加进来。

用途：

- 在 OOF 证据上做一层 forward selection
- 给后续 stacker 输入一个更小、更强的 parent 子集

要求：

- 必须配置 target
- `selection.min_gain` 必须是有限且非负的数
- 不允许和 `selection.max_pairwise_corr` 混用

## 关键约束

P5 这次专门把几个容易“看起来合理、实际上会误导”的约束收紧了：

1. `top_n` 不允许在 compatibility / selection 之前过早截断。
2. `max_parents` 不允许先完整选择、再事后裁剪；它必须作为 selection 过程内的上限约束参与求解。
3. 显式 `candidate_ids` 不能因为 `top_n` 影响而被误报成 `not found in registry`。
4. `hill_climb_greedy` 不能和 `max_pairwise_corr` 混用。
5. selection report / stacking manifest 必须区分：
   - configured limit
   - effective limit

## 配置安全门

P5 当前已经拦住这些高风险配置问题：

- `selection.max_pairwise_corr` / `selection.min_gain` / `stacker.params` 中的 `NaN`、`Inf`
- `stacking` / `selection` / `stacker` 层级的未知字段 typo
- 非法的 `stacker.method`
- `top_n < min_parents`
- `max_parents < min_parents`
- `hill_climb_greedy + max_pairwise_corr`

另外，`preflight_only` 只允许用于 `stack-preflight`，不能用于 `ktab stack`。

## Artifact

P5 预期产物包括：

```text
artifacts/experiments/<competition>/<experiment_id>/
  selection_report.md
  stacking_manifest.json
  fold_metrics.csv
  metrics.json
  run_manifest.json

artifacts/oof/<competition>/<experiment_id>/
  stack_oof.parquet
  oof.parquet

artifacts/submissions/<competition>/<experiment_id>/
  stack_test.parquet
  submission.csv
```

其中：

- `stack_oof.parquet` / `stack_test.parquet` 反映 parent-level stacked matrix
- `oof.parquet` / `submission.csv` 反映 stacker 最终输出
- `stacking_manifest.json` 记录 selection strategy、accepted parents、rejected reasons、pairwise correlations、hill climb trace、effective limits

## Registry / Lineage

P5 的 registry row 需要能回答：

- 这个 stacking experiment 来自哪些 parent experiment？
- 它用的是哪一种 stacker method？
- 它的 OOF 分数是多少？
- selection strategy 是什么？
- 相关 report / manifest / output 文件在哪里？

因此，stacking row 至少要记录：

- `model_family: stacking`
- `model_preset: <stacker_method>`
- `parent_experiment_ids`
- `parent_count`
- `selection_report_path`
- `stacking_manifest_path`

## 测试与回归

P5 当前已经覆盖了几类关键测试：

- config schema 与 loader 的约束校验
- `score_desc` / `diversity_greedy` / `hill_climb_greedy` 单测
- `top_n` 后移语义
- `max_parents` 作为 in-selection 约束的语义
- `hill_climb_greedy + max_pairwise_corr` 拒绝路径
- stacking preflight 与 completed runtime 集成测试

## 非目标

P5 当前不做：

- public-LB-driven selection
- 多层 stacking
- pseudo-label teacher selection
- original / external data 驱动的 stacking 决策
- 自动 Kaggle submission
- 大规模并发 stacking search

这些内容应当进入后续阶段，而不是让 P5 先背上复杂度。

## 与 P6 的边界

P5 已经解决的是：

- “如何从 candidate pool 中安全地选择 parent，并完成一层 stacking”

P6 更应该解决的是：

- “如何从 EDA、feature proposal、feature build、feature compare 这条线上更系统地产生更强的 candidate”

也就是说，P5 负责 selection / stacking 的闭环，P6 负责更强的 candidate source。
