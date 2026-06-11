# P6 Feature Workbench

P6 的目标不是“盲目多造一些特征”，而是把更强 candidate 的生产过程收口成一个可配置、可审计、可回放的闭环。

如果说 P5 解决的是“如何从候选池里安全地选、再去叠”，那么 P6 解决的是“如何更系统地造出更强、而且仍然 OOF-safe 的候选”。

## 边界

P6 与相邻阶段的边界要先说清楚：

- `P5` 已完成的是 candidate selection、hill climbing 和一层 stacking。
- `P6` 要新增的是 `EDA -> feature proposal -> fold-safe feature build -> OOF compare -> candidate promotion`。
- `P7` 才进入 pseudo-label、extra training、final retrain 和 submission review pack。

一句话概括：

- `P5` 负责 selection / stacking。
- `P6` 负责更强的 candidate source。
- `P7` 负责 final candidate workflow。

## 目标

P6 要回答这些问题：

- 这轮 EDA 发现了什么结构、drift、泄漏风险和可疑列关系？
- 这些发现如何沉淀成结构化的 feature backlog，而不是只留在聊天记录里？
- 哪些特征是 `global_safe`，哪些是 `outer_fold_safe`，哪些必须 `nested_fold_safe`？
- 特征构建后的 train / test matrix 是否和 OOF、id、fold 对齐？
- 哪些 feature bundle 真的带来了稳定 OOF 收益，哪些只是高风险噪声？

## MVP

P6 MVP 建议收得很克制，只做四件事：

### 1. 可执行 EDA

新增一条结构化 EDA 入口：

```powershell
uv run ktab eda scan --config configs\eda\playground-series-s6e3.example.yaml
```

这一层至少产出：

- `eda_summary.md`
- `eda_manifest.json`
- `leakage_watchlist.json`
- `feature_backlog.json`

首批关注：

- 列类型、缺失、重复、基数
- train/test drift
- synthetic/original 关系线索
- target 相关的可疑列与风险提醒

### 2. 结构化 feature proposal

Agent 可以提出想法，但想法必须落到 proposal/config 上：

```powershell
uv run ktab proposal validate --config configs\proposals\freq_count.example.yaml
uv run ktab proposal register --config configs\proposals\freq_count.example.yaml
```

P6 里 proposal 的作用是：

- 记录假设
- 记录风险
- 记录输入列与预期产物
- 让后续 feature build 有清晰边界

### 3. fold-safe feature build

真正的特征构建应由 `ktab` 中央执行：

```powershell
uv run ktab feature build --config configs\features\freq_bundle.example.yaml
```

MVP 首批只建议支持三类低风险家族：

- `frequency/count`
- `binning`
- `arithmetic interactions`

`target encoding` 可以在 P6 先留接口，但默认应走显式安全门，且要求 nested / OOF-safe。

### 4. 接入现有 P2 / P4 / compare 流水线

P6 不另起训练入口，而是接入现有闭环：

- `ktab run --config ...`
- `ktab factory --config ...`
- `ktab compare --artifact-root ...`

这样 feature bundle 的价值最终还是回到 OOF 证据，而不是停在“看起来很聪明的特征故事”。

## 分层设计

P6 适合拆成四层：

### 1. EDA 层

职责：

- 扫描数据
- 输出 watchlist
- 提炼候选 feature family
- 给出 CV 和风险提醒

建议源码位置：

- `src/ktabforge/eda/config.py`
- `src/ktabforge/eda/profiling.py`
- `src/ktabforge/eda/watchlist.py`
- `src/ktabforge/eda/report.py`

### 2. Feature Backlog / Spec 层

职责：

- 把 feature idea 变成可审计的 backlog item
- 把 backlog item 落成明确的 feature spec

`feature_backlog` 至少应包含：

- `feature_id`
- `feature_family`
- `hypothesis`
- `source_columns`
- `requires_target`
- `fold_safety`
- `leakage_risk`
- `transductive_risk`
- `data_sources`
- `validation_plan`
- `status`
- `review_required`
- `owner`

`feature_spec` 至少应包含：

- `feature_id`
- `output_columns`
- `params`
- `fit_scope`
- `transform_scope`
- `cv_protocol_compatibility`
- `transductive_mode`
- `artifact_expectations`
- `cache_key_fields`
- `review_fields`

### 3. Feature Build 层

职责：

- 按 feature spec 真正构建 train/test feature matrix
- 管控 fold-safe 与 nested-safe 语义
- 写出 feature manifest 和缓存

建议执行模式：

- `global_safe`
- `outer_fold_safe`
- `nested_fold_safe`

建议源码位置：

- `src/ktabforge/features/registry.py`
- `src/ktabforge/features/pipeline.py`
- `src/ktabforge/features/transforms/base.py`
- `src/ktabforge/features/transforms/frequency.py`
- `src/ktabforge/features/transforms/binning.py`
- `src/ktabforge/features/transforms/interactions.py`

### 4. Review / Promotion 层

职责：

- 比较 feature bundle 对实验结果的影响
- 记录接受、拒绝和待观察原因
- 把通过验证的候选晋升给 P5 使用

这一层不要由 agent 直接改写 registry 结果，而是让 `ktab compare` 和 `ktab review-pack` 负责最终收口。

## CLI 与配置

P6 推荐新增这些入口：

```powershell
uv run ktab eda scan --config <eda.yaml>
uv run ktab proposal validate --config <proposal.yaml>
uv run ktab proposal register --config <proposal.yaml>
uv run ktab proposal review --proposal-id <proposal_id> --status accept|reject|revise
uv run ktab feature build --config <feature_build.yaml>
uv run ktab feature inspect --feature-build-id <id>
uv run ktab review-pack --competition <competition> --factory-id <factory_id>
```

`compare` 建议扩展：

- `--group-by experiment|proposal|feature_build|factory`
- `--include-review-status`
- `--output markdown|csv|json`

建议新增配置目录：

```text
configs/
├── eda/
├── proposals/
├── features/
└── schemas/
    ├── eda.schema.json
    └── feature_pipeline.schema.json
```

## 产物与 Registry

P6 的关键原则是：agent 负责提案与审查，`ktab` 负责执行、落盘和注册。

建议产物：

```text
artifacts/eda_findings/<competition>/<eda_id>/
  eda_manifest.json
  eda_summary.md
  leakage_watchlist.json
  feature_backlog.json

artifacts/features/<competition>/<feature_build_id>/
  feature_build_manifest.json
  feature_schema.json
  feature_build_report.md
  train_features.parquet
  test_features.parquet

artifacts/reports/<competition>/<review_pack_id>/
  review_pack_manifest.json
  decision_context.md
```

建议在现有 experiment registry 里补这些字段：

- `eda_id`
- `proposal_ids`
- `feature_build_ids`
- `feature_bundle_id`
- `factory_id`
- `review_status`
- `review_pack_id`
- `owner_agent`

如果后面需要更强追踪，再拆出：

- `proposal_registry`
- `feature_build_registry`

## 多 Agent 分工

P6 适合把“发散型认知工作”和“中央执行工作”分开。

适合交给 agents 的部分：

- 读 EDA 报告并提出假设
- 起草 feature proposal
- 风险标签初判
- 对 compare 结果做解释
- 起草 review pack 的人类可读结论

必须由 `ktab` 中央执行的部分：

- config schema 校验
- id 分配与合法性检查
- feature build
- OOF / id / fold 对齐检查
- artifact 路径生成
- manifest 写入
- registry 追加
- compare 排名口径
- review-pack 打包

## 安全门

P6 文档里必须把这些规则写死：

1. target encoding 必须 OOF-safe；默认要求 nested inner fold。
2. target-dependent aggregation 不允许先 fit full train，再生成 OOF。
3. train+test count/frequency 属于 transductive 特征，必须显式标记。
4. external/original data 必须记录来源、合法性、是否含标签、是否使用标签。
5. feature build 必须绑定 `cv_protocol_id`。
6. train/test feature matrix 必须分别与 train rows、sample submission id 顺序对齐。
7. `leakage_watchlist` 必须是 artifact，不是口头提醒。

## 测试与验收

P6 MVP 可验收条件建议如下：

- `ktab eda scan --config ...` 能稳定产出 EDA artifact。
- proposal config 可以通过 schema 校验并注册。
- `ktab feature build --config ...` 能写出 feature manifest 和 feature cache。
- 低风险特征家族可以接入现有 `ktab run` / `ktab factory`。
- compare 能按 feature bundle 或 proposal 维度输出对比结果。
- 明确阻断 target leakage、未标记 transductive 和 OOF 不安全路径。

## Deferred To P7

这些内容不要混进 P6：

- pseudo-label teacher policy
- pseudo-label 样本筛选与混合比例
- extra training / final retrain
- final shortlist freeze
- submission review pack 的最终审批闭环

P6 先把“证据充足的 candidate source”打稳，P7 再去处理“如何把候选推进成最终提交件”。
