# P0 项目基座

P0 定义项目的地基：不可违反的实验规则、证据契约、验收标准和渐进式披露入口。后续 P1/P2/P3 都必须服从这里记录的边界。

本阶段的核心目标是：

> 先把可审计的实验规则和证据契约建好，再建设高吞吐模型工厂。

P0 不是完整 Kaggle 训练框架，也不声称训练流水线已经实现。它的职责是让 `/goal`、artifact、泄漏审查、README 和 Agent 路由都能被追踪和验收。

## 记录的 `/goal`

本阶段对应的项目级 goal：

```text
/goal objective=build_evidence_pipeline competition=_project request="build P0 project foundation contracts, docs, and README" budget=static_review
```

结构化形式：

```yaml
goal:
  goal_id: 20260606-p00-project-foundation
  objective: build_evidence_pipeline
  competition: _project
  request: build P0 project foundation contracts, docs, and README
  budget:
    run_mode: static_review
    wall_time: null
    gpu: null
    submissions: 0
```

## P0 要回答的问题

P0 必须让未来每次实验都能回答这些问题：

1. 这项工作来自哪个 `/goal`？
2. 哪个 config、数据版本、CV 协议和命令生成了结果？
3. OOF 与 test prediction 在哪里？
4. train 与 sample submission 的行是否对齐？
5. 泄漏风险和 public LB 风险是否审查过？
6. 结果能否复现？
7. 最终决策是什么：接受、拒绝、重试、调查，还是请求人类批准？

## P0 交付物

### 1. Goal Contract

规范文档：

- `docs/goal_contract.md`

必须记录：

- 最小 `/goal` 输入。
- 自动发现规则。
- 必要追问。
- 不同 objective 需要的上下文。
- 验收 schema。
- inconclusive 条件。

### 2. Competition Workspace Contract

未来比赛建议使用：

```text
competitions/<competition>/
  goal.yaml
  configs/
    competition.yaml
    paths.yaml
    experiments/
    ensembles/
  reports/
  proposals/
```

P0 只记录这个契约。完整脚手架可以留到后续 CLI 任务实现。

### 3. CV Protocol Contract

每个真实实验必须引用一个 `cv_protocol_id`，并记录：

- splitter。
- fold 数量。
- seed。
- group 或 time column，如果比赛需要。
- metric 与 metric mode。
- OOF safety statement。

所有 ensemble candidate 必须使用同一套 CV 协议，或者明确记录兼容性说明。

### 4. Artifact Contract

规范文档：

- `docs/artifact_contract.md`

每个实验最终都应该写出：

- `config.yaml`
- `run_manifest.json`
- `metrics.json`
- `fold_metrics.csv`
- `oof.parquet` 或 `oof.npy` 及其 schema
- `pred.npy` 或 `test_predictions.parquet`
- `feature_manifest.json`
- `model_manifest.json`
- `environment.json`
- `git.json`
- `notes.md`

P0 记录契约和验收门，后续阶段再逐步实现 writer。

### 5. Experiment Registry Contract

最小 registry 字段：

```yaml
experiment_id: string
competition: string
goal_id: string
cv_protocol_id: string
data_version: string
feature_manifest_hash: string
model_family: string
params_hash: string
seed: integer
run_mode: static_review | smoke | single_fold | full_cv | hpo
oof_path: path | null
test_pred_path: path | null
fold_metrics_path: path | null
metric_name: string
oof_score: number | null
fold_mean: number | null
fold_std: number | null
prediction_type: probability | logit | rank | raw | not_applicable
lineage:
  parents:
    - experiment_id
leakage_risk: low | medium | high | blocker
status: keep | reject | failed | inconclusive
reason: string
```

### 6. Leakage And Submission Gates

规范文档：

- `docs/leakage_rules.md`

P0 安全门：

- 生成 OOF 前，不允许在 full train 上训练目标相关变换。
- target encoding 必须 OOF-safe，或在 fold 内做 nested 编码。
- external/original data 必须审查来源和合法性。
- public LB 不能决定 weights、thresholds、features 或训练逻辑。
- Kaggle submission 必须有人类明确批准。

### 7. README 与 AGENTS

README 应该让人类在五分钟内理解：

- 项目是什么。
- 项目不是什么。
- 为什么 OOF/artifact 重要。
- `/goal` 如何工作。
- 当前 P0/P1/P2/P3 阶段是什么。
- 下一步该读哪里。

AGENTS.md 是唯一 Agent 总纲，只放最高优先级规则和任务路由；细节放入 `docs/`。

## P0 非目标

P0 不实现：

- 完整训练框架。
- 完整 neural model zoo。
- RAPIDS/cuDF/cuML 默认安装。
- HPO scheduler。
- Web UI。
- 数据库服务。
- 自动 Kaggle submission。
- public-LB-driven weight search。
- 多层 stacking 实现。

## P0 验收

P0 完成条件：

- README 存在，并指向渐进式披露文档。
- `/goal` 输入与验收规则已记录。
- 项目基座范围已记录。
- AGENTS 导航指向正确文档。
- artifact、leakage、environment 文档仍是各自规范 owner。
- 链接可以解析。
- Markdown code fence 平衡。
- 没有失败 patch、merge marker 或编码乱码。
- 生成数据和 artifact 不进入 git。

P0 不完成的情况：

- `/goal` 验收只存在于聊天里，没有落文档。
- README 复制深层文档，而不是导航到深层文档。
- P0 暗示完整训练框架已经存在。
- 安全门隐藏在入口路径之外。
- 原始数据或生成 artifact 被提交。
